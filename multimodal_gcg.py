import copy
import gc
import json
import logging

from dataclasses import dataclass, asdict
import random
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
import wandb

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax

from PIL import Image
import pdb


from torchtune.rlhf.loss import DPOLoss

import torch.nn.functional as F


def preprocess_llama2(tokenizer, convs: list) -> list: # tokenize and concat the coversations
    input_ids = None

    if convs[0]['role'] == 'system':
        sysmsg = convs[0]['content']
        convs.pop(0)
    else:
        sysmsg = ''
        
    conv_str = f'<s>[INST] <<SYS>>\n{sysmsg}\n<</SYS>>\n\n'


    for ind, conv in enumerate(convs):
        if ind % 2 == 0: # human
            h = conv['content'].strip()
            h = f"<s> [INST] {h} [/INST] " # [1, 29871, 518, 25580, 29962, 29871, human, 518, 29914, 25580, 29962, 29871]
            cur_input_ids = tokenizer(h, add_special_tokens=False, truncation=True).input_ids
            # cur_input_ids = tokenizer_image_token(prompt=h, return_tensors=return_tensors)
            
            conv_str +=  h
            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])

        else: # gpt
            g = conv['content']
            if g is not None:
                g = f'{g}</s>'
                cur_input_ids = tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
                input_ids = torch.cat([input_ids, cur_input_ids])
            conv_str += g

    return conv_str

def preprocess_vicuna(tokenizer, convs: list) -> list: # tokenize and concat the coversations
    input_ids = None

    if convs[0]['role'] == 'system':
        sysmsg = convs[0]['content']
        convs.pop(0)
    else:
        sysmsg = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    
    # no sys msg!!!
    sysmsg = ''

    conv_str = f'{sysmsg} '

    for ind, conv in enumerate(convs):
        if ind % 2 == 0: # human
            h = conv['content'].strip()
            h = f"USER: {h} "
            cur_input_ids = tokenizer(h, add_special_tokens=False, truncation=True).input_ids
            # cur_input_ids = tokenizer_image_token(prompt=h, return_tensors=return_tensors)
            
            conv_str +=  h
            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])

        else: # gpt
            g = conv['content']
            if g is not None:
                g = f"ASSISTANT: {g}</s>"
                cur_input_ids = tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
                input_ids = torch.cat([input_ids, cur_input_ids])
            conv_str += g

    return conv_str

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class MMGCGConfig:
    num_steps_per_sample: int = 250
    # optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    loss_type: str = 'sigmoid'
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    loss_threshold: float = 0.15
    dpo_loss_weight: float = 1.
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


@dataclass
class MMGCGInput:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]



class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    # tensor.repeat(n1 ,n2, n3, ...), where n's are number of times to repeat in each dimension
    original_ids = ids.repeat(search_width, 1) # (search_width, suflen)


    # find the tokens with largest negative grads
    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices # (suflen, topk)

    # (search_width, n_replace), range from 0 to suflen-1
    sampled_ids_chosen = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    
    # pick the elements from dim=2
    # (search_width, 1, 1)
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_chosen], # (search_width, n_replace, topk)
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device) # (search_width, n_replace, 1)
    )
    sampled_ids_val = sampled_ids_val.squeeze(2) # (search_width, 1)

    # (search_width, suflen)
    new_ids = original_ids.scatter_(dim=1, index=sampled_ids_chosen, src=sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)

class MMGCG:
    def __init__(
        self, 
        args,
        model: transformers.PreTrainedModel,
        tokenizer : transformers.PreTrainedTokenizer,
        config: MMGCGConfig,
        image_processor = None,
    ):
        self.model = model

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.args = args

        # init modules
        self.clip_model = model.vision_tower
        self.projector = model.multi_modal_projector
        self.language_model = model.language_model


        self.embedding_layer = self.language_model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None

        self.stop_flag = False
        self.buffer_initialized = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

        self.init_training_state()

    def init_training_state(self,):
        self._global_step = 1

    def get_global_step(self):
        return self._global_step
    
    def step(self):
        self._global_step += 1

    def get_sample_idx(self):
        return self._sample_idx

    def set_sample_idx(self, idx, offset):
        self._sample_idx =  idx + offset


    @torch.no_grad()
    def get_clip_output(
        self,
        images, 
        selected_layer=-2, select_stg='patch', pooling_stg = 'mean'
    ):
        # images = [Image.open(image).convert('RGB') for image in images]
        inputs = self.image_processor(images, return_tensors='pt').to(self.model.device) # (1, 3, 336, 336)
        
        emb = self.clip_model(**inputs, output_hidden_states=True)

        emb = emb.hidden_states[selected_layer] # (bsz, 577, 1024)
        if select_stg == 'patch':
            emb = emb[:, 1:] # (bsz, 576, 1024)
        else:
            raise NotImplementedError
        
        if pooling_stg == 'none':
            return emb 
        elif pooling_stg == 'mean':
            return emb.mean(1) # (bsz, 1024)
        elif pooling_stg == 'max':
            return emb.max(1) # (bsz, 1024)
        
        raise NotImplementedError

    
    @torch.no_grad()
    def get_projector_output(
        self,
        images, 
        selected_layer=-2, select_stg='patch', pooling_stg = 'none'
    ):
        img_feat = self.get_clip_output(
            images,
            pooling_stg=pooling_stg, 
            selected_layer=selected_layer, 
            select_stg=select_stg,
        ) # (bsz, 576, 1024)

        out = self.projector(img_feat)

        return out # (1, 576, 4096)

    def run_single(
        self,
        # messages: Union[str, List[dict]],
        raw_message: dict,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        # if isinstance(messages, str):
        #     messages = [{"role": "user", "content": messages}]
        # else:
        #     messages = copy.deepcopy(messages)

        image = raw_message.get('image', None)
        if image:
            image = Image.open(image).convert('RGB')
        messages = [{"role": "user", "content": raw_message['prompt']}]

        target_chosen = raw_message['chosen']
        target_rejected = raw_message['rejected']
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        pv = self.args.prompt_version
        if 'llama-2' in pv:
            template = preprocess_llama2(tokenizer=tokenizer, convs=messages) 
        elif 'llava-v1.5' in pv:
            template = preprocess_vicuna(tokenizer, messages)
        else:
            template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        
        
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if any([_ in pv for _ in ['llama-2', 'llava-v1.5']]):
            pass
        else:
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")

        before_str, after_str = template.split("{optim_str}")

        # target = " " + target if config.add_space_before_target else target
        target_chosen = " " + target_chosen if config.add_space_before_target else target_chosen
        target_rejected = " " + target_rejected if config.add_space_before_target else target_rejected

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_chosen_ids = tokenizer([target_chosen], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_rejected_ids = tokenizer([target_rejected], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer

        # before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]
        before_embeds, after_embeds, target_chosen_embeds, target_rejected_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_chosen_ids, target_rejected_ids)]
        
        '''
        todo: image emb here! concat it with before_embeds
        '''
        if image is not None:
            image_embeds = self.get_projector_output(image)
            # before_embeds = torch.cat([before_embeds, image_embeds], dim=1)
            before_embeds = torch.cat([image_embeds, before_embeds], dim=1)

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        

        # self.target_ids = target_ids
        # self.target_chosen_ids = target_chosen_ids
        # self.target_rejected_ids = target_rejected_ids

        # self.before_embeds = before_embeds
        # self.after_embeds = after_embeds
        # # self.target_embeds = target_embeds
        # self.target_chosen_embeds = target_chosen_embeds
        # self.target_rejected_embeds = target_rejected_embeds

        # Initialize the attack buffer for the first run
        if not self.buffer_initialized:
            buffer = self.init_buffer(
                after_embeds=after_embeds,
                target_chosen_embeds=target_chosen_embeds,
                target_rejected_embeds=target_rejected_embeds,
                target_chosen_ids=target_chosen_ids,
                target_rejected_ids=target_rejected_ids,
                before_embeds=None, 
            ) 
            self.buffer = buffer
        else:
            buffer = self.buffer
        optim_ids = buffer.get_best_ids() # (1, suflen)

        # losses = []
        losses = {'losses': [], 'chosen_losses': [], 'rejected_losses':[]}
        optim_strings = []
        
        for _ in (range(config.num_steps_per_sample)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(
                optim_ids,
                after_embeds=after_embeds,
                target_embeds=(target_chosen_embeds, target_rejected_embeds),
                target_ids=(target_chosen_ids, target_rejected_ids),
                before_embeds=before_embeds,
            ) # (1, suflen, vocab_size)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0] # some ids may be filtered

                # pdb.set_trace()
                # before_embeds.shape, after_embeds.shape, target_chosen_embeds.shape, target_rejected_embeds.shape
                # (torch.Size([1, 629, 4096]), torch.Size([1, 1, 4096]), torch.Size([1, 1, 4096]), torch.Size([1, 1, 4096]))

                # Compute loss on all candidate sequences 
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_chosen_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_chosen_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                    input_rejected_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_rejected_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_chosen_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_chosen_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                    input_rejected_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_rejected_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)



                loss, chosen_nll_loss, rejected_nll_loss = self.compute_loss(
                    search_batch_size = batch_size,
                    chosen_emb=input_chosen_embeds,
                    rejected_emb=input_rejected_embeds,
                    target_chosen_ids = target_chosen_ids,
                    target_rejected_ids = target_rejected_ids,
                )

                # loss_chosen = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_chosen_embeds, self.target_chosen_ids)
                # loss_rejected = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_rejected_embeds, self.target_rejected_ids)
                # loss = loss_chosen - loss_rejected


                current_loss = loss.min().item()
                min_idx = loss.argmin()
                current_chosen_loss = chosen_nll_loss.min().item()
                current_rejected_loss = rejected_nll_loss.min().item()

                optim_ids = sampled_ids[min_idx].unsqueeze(0)

                # Update the buffer based on the loss
                # losses.append(current_loss)
                losses['losses'].append(current_loss)
                losses['chosen_losses'].append(current_chosen_loss)
                losses['rejected_losses'].append( current_rejected_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

                # print(raw_message['type'], current_loss, current_chosen_loss, current_rejected_loss)


            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            # print(optim_str)
            # print(f'loss: {current_loss}')
            # print('')

            buffer.log_buffer(tokenizer)     

            if self.args.report_to == 'wandb':
                wandb.log({'step_loss': current_loss, 'step': self.get_global_step()})
                wandb.log({'step_chosen_loss': current_chosen_loss, 'step': self.get_global_step()})
                wandb.log({'step_rejected_loss': current_rejected_loss, 'step': self.get_global_step()})
                # wandb.log({'suffix': optim_str}, step = self.get_global_step())
                # if self.get_global_step() % 2 == 0:
                #     # print(self.get_global_step(), self.args.wandb_table)
                #     self.args.wandb_table.add_data(self.get_global_step(), optim_str)
                #     wandb.log({'suffix': self.args.wandb_table, })
                    
            
            self.step() # increase counter


            if current_loss < self.config.loss_threshold:
                break
            # if current_chosen_loss < current_rejected_loss:
            #     break

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        # min_loss_index = losses.index(min(losses)) 
        min_loss_index = losses['losses'].index(min(losses['losses'])) 
        if self.args.report_to == 'wandb':
            # wandb.log({'sample_loss': losses[min_loss_index], 'step': self.get_global_step()})
            wandb.log({'sample_loss': losses['losses'][min_loss_index], 'step': self.get_global_step()})
            wandb.log({'sample_chosen_loss': losses['chosen_losses'][min_loss_index], 'step': self.get_global_step()})
            wandb.log({'sample_rejected_loss': losses['rejected_losses'][min_loss_index], 'step': self.get_global_step()})
            # wandb.log({'suffix': optim_str}, step = self.get_global_step())

        with open(self.args.output_pth, 'a') as f:
            line = json.dumps(
                {'step': self.get_global_step(), 
                 'sample_idx': self.get_sample_idx(),
                 'sample_id': raw_message['id'],
                 'type': raw_message['type'],
                 'sample_loss': losses['losses'][min_loss_index],
                 'sample_chosen_loss': losses['chosen_losses'][min_loss_index],
                 'sample_rejected_loss': losses['rejected_losses'][min_loss_index],
                 'suffix': optim_strings[min_loss_index]},
                 ensure_ascii=False
            )
            f.write(line + '\n')

        result = GCGResult(
            # best_loss=losses[min_loss_index],
            best_loss=losses['losses'][min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )
        

        return result


    def run_batch(
        self,
        # messages: list[dict],
        # # target: str,
        # targets_chosen: list[str],
        # targets_rejected: list[str],
        batch,
        epoch_idx,
    ):
        '''
        run for each sample in a batch.
        '''
        pbar = tqdm(range(len(batch)))


        for sample_idx in pbar:
            self.set_sample_idx(sample_idx, epoch_idx*len(batch))
            sample = batch[sample_idx]
            # message, target_chosen, target_rejected = sample['message'], sample['chosen'], sample['rejected']
            result = self.run_single(
                raw_message=sample,
            )
            best_loss = result.best_loss
            best_string = result.best_string
            pbar.set_postfix_str(f'Loss: {best_loss:.3f}, Suf: |>{best_string}<|' )



    def init_buffer(
            self,
            after_embeds,
            target_chosen_embeds,
            target_rejected_embeds,
            target_chosen_ids,
            target_rejected_ids,
            before_embeds=None, 
    ) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_chosen_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                after_embeds.repeat(true_buffer_size, 1, 1),
                target_chosen_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
            init_buffer_rejected_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                after_embeds.repeat(true_buffer_size, 1, 1),
                target_rejected_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_chosen_embeds = torch.cat([
                before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                after_embeds.repeat(true_buffer_size, 1, 1),
                target_chosen_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
            init_buffer_rejected_embeds = torch.cat([
                before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                after_embeds.repeat(true_buffer_size, 1, 1),
                target_rejected_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)



        init_buffer_losses, chosen_nll_loss, rejected_nll_loss = self.compute_loss(
            search_batch_size=true_buffer_size,
            chosen_emb=init_buffer_chosen_embeds,
            rejected_emb=init_buffer_rejected_embeds,
            target_chosen_ids=target_chosen_ids,
            target_rejected_ids=target_rejected_ids,
        )

        # init_buffer_chosen_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_chosen_embeds, self.target_chosen_ids)
        # init_buffer_rejected_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_rejected_embeds, self.target_rejected_ids)
        # init_buffer_losses = init_buffer_chosen_losses - init_buffer_rejected_losses



        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        
        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        self.buffer_initialized = True
        
        return buffer



    def compute_loss(
        self,
        search_batch_size,
        chosen_emb, # (bsz, seqlen, dim)
        rejected_emb, # (bsz, seqlen, dim)
        target_chosen_ids, # (bsz, trglen, dim)
        target_rejected_ids, # (bsz, trglen', dim)
    ):
        
        # dpo loss
        # https://github.com/huggingface/trl/blob/b8c9d9c7bc999d06f2a48cba5b688de6d8e8beab/trl/trainer/dpo_trainer.py#L863
        # input: all_chosen_emb, all_rej_emb, target_chosen_ids, target_rejected_ids
        # forward twice to get two logits
        # aggregated = logsoftmax(logits, dim=-1).sum(-1)
        all_loss = []
        prefix_cache_batch = []

        current_batch_size = chosen_emb.shape[0]

        if self.prefix_cache:
            if not prefix_cache_batch or current_batch_size != search_batch_size:
                prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

            chosen_logits = self.model(inputs_embeds=chosen_emb, past_key_values=prefix_cache_batch).logits
            rejected_logits = self.model(inputs_embeds=rejected_emb, past_key_values=prefix_cache_batch).logits
        else:
            chosen_logits = self.model(inputs_embeds=chosen_emb).logits
            rejected_logits = self.model(inputs_embeds=rejected_emb).logits

        # compute nll loss
        tmp = chosen_emb.shape[1] - target_chosen_ids.shape[1]
        shift_logits = chosen_logits[..., tmp-1:-1, :].contiguous()
        shift_labels = target_chosen_ids.repeat(current_batch_size, 1)
        chosen_nll_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
        chosen_nll_loss = chosen_nll_loss.view(current_batch_size, -1).mean(dim=-1)

        tmp = rejected_emb.shape[1] - target_rejected_ids.shape[1]
        shift_logits = rejected_logits[..., tmp-1:-1, :].contiguous()
        shift_labels = target_rejected_ids.repeat(current_batch_size, 1)
        rejected_nll_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
        rejected_nll_loss = rejected_nll_loss.view(current_batch_size, -1).mean(dim=-1)

        if self.config.dpo_loss_weight > 0:
            # get target logits
            chosen_trg_len = target_chosen_ids.shape[1]
            chosen_logps = chosen_logits[..., -chosen_trg_len:, :].mean(1).log_softmax(-1)

            rejected_trg_len = target_rejected_ids.shape[1]
            rejected_logps = rejected_logits[..., -rejected_trg_len:, :].mean(1).log_softmax(-1)

            logratios = chosen_logps - rejected_logps

            # add ref model probs here
            logits = logratios
        

            if self.config.loss_type == "sigmoid":
                beta = 1
                label_smoothing = 0
                dpo_loss = (
                    -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                    - F.logsigmoid(-beta * logits) * label_smoothing
                ).mean(-1) # (bsz, vocab) -> (bsz,)
                nll_loss = chosen_nll_loss / (chosen_nll_loss + rejected_nll_loss)
                loss = nll_loss + self.config.dpo_loss_weight * dpo_loss

        else:
            loss = chosen_nll_loss / (chosen_nll_loss + rejected_nll_loss)

        loss.shape, chosen_nll_loss.shape, rejected_nll_loss.shape

        return loss, chosen_nll_loss, rejected_nll_loss

    
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
        after_embeds: Tensor,
        target_embeds: tuple,
        target_ids: tuple,

        before_embeds: Tensor = None,

    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        target_chosen_embeds, target_rejected_embeds = target_embeds
        target_chosen_ids, target_rejected_ids = target_ids


        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            # input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            input_chosen_embeds = torch.cat([optim_embeds, after_embeds, target_chosen_embeds], dim=1)
            input_rejected_embeds = torch.cat([optim_embeds, after_embeds, target_rejected_embeds], dim=1)
            output_chosen = model(inputs_embeds=input_chosen_embeds, past_key_values=self.prefix_cache)
            output_rejected = model(inputs_embeds=input_rejected_embeds, past_key_values=self.prefix_cache)
        else:
            input_chosen_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_chosen_embeds], dim=1)
            input_rejected_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_rejected_embeds], dim=1)
            output_chosen = model(inputs_embeds=input_chosen_embeds)
            output_rejected = model(inputs_embeds=input_rejected_embeds)
        


        logits_chosen = output_chosen.logits
        logits_rejected = output_rejected.logits

        losses = []

        for (input_embeds, target_ids, logits) in zip( (input_chosen_embeds, input_rejected_embeds), (target_chosen_ids, target_rejected_ids), (logits_chosen, logits_rejected) ):
            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
            shift_labels = target_ids


            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
            else:
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses.append(loss)

        # loss here
        # loss = losses[0] - losses[1]
        # loss = torch.sigmoid(losses[0] - losses[1])
        loss = losses[0] / (losses[0] + losses[1])

        # if self.args.report_to == 'wandb':
        #     wandb.log({'chosen_loss': losses[0], 'step': self.get_global_step()})
        #     wandb.log({'rejected_loss': losses[1], 'step': self.get_global_step()})
                

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad


    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
        target_ids: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits
                pdb.set_trace()

                tmp = input_embeds.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)



                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run_mmgcg(
    args,
    model: transformers.PreTrainedModel,
    # messages: Union[str, List[dict]],
    # target_chosen: str,
    # target_rejected: str,
    data: list[dict],
    tokenizer: transformers.PreTrainedTokenizer ,
    config: Optional[MMGCGConfig] = None, 
    image_processor = None,
) -> GCGResult:
    """Generates a single optimized string using GCG. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    
    if config is None:
        config = MMGCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = MMGCG(
        args=args,
        model=model, 
        tokenizer=tokenizer, 
        config=config,
        image_processor=image_processor,
    )
    # result = gcg.run(messages, target)
    # result = gcg.run(messages, target_chosen, target_rejected)

    for epoch in range(args.num_epochs):
        random.shuffle(data)
        result = gcg.run_batch(data, epoch)
    return result
    