import argparse
import torch
import json

import wandb
import pdb

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration

from dataclasses import asdict

import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    # Set the seed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed the built-in random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Configure cuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If using the Hugging Face Transformers library
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass  # Transformers library is not installed




parser = argparse.ArgumentParser(description='mmgcg')
parser.add_argument('--model_dir', default='')
parser.add_argument('--data_path', default='')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--experiment_name', required=True)
parser.add_argument('--prompt_version', required=True)
parser.add_argument('--report_to', default='')

parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--num_steps_per_sample', default=500, type=int)
parser.add_argument('--search_width', default=64, type=int)
parser.add_argument('--optim_str_init', default='x x x x x x x x x x x x x x x x x x x x', type=str)
parser.add_argument('--topk', default=64, type=int)
parser.add_argument('--seed', default=4, type=int)
parser.add_argument('--verbosity', default='WARNING', type=str)
parser.add_argument('--use_mellowmax', action='store_true')

parser.add_argument('--loss_threshold', default=0.1, type=float, help='learning rate')
parser.add_argument('--dpo_loss_weight', default=1., type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
args.output_pth = os.path.join(args.output_dir, f'{args.experiment_name}.jsonl')

seed_everything(args.seed)
##############################################

from multimodal_gcg import MMGCGConfig, run_mmgcg


def load_model_and_processor():
    model_id = args.model_dir

    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer, image_processor = processor.tokenizer, processor.image_processor

    return model, tokenizer, image_processor



def load_data():

    pth = args.data_path
    data = json.load(open(pth))
    return data




if __name__ == '__main__':


    # load data
    data = load_data()

    # load model
    model, tokenizer, image_processor = load_model_and_processor()

    # load_config
    config = MMGCGConfig(
        num_steps_per_sample=args.num_steps_per_sample,
        search_width=args.search_width,
        topk=args.topk,
        seed=args.seed,
        loss_threshold=args.loss_threshold,
        verbosity=args.verbosity,
        use_mellowmax=args.use_mellowmax,
        optim_str_init=args.optim_str_init,
    )


    if args.report_to == 'wandb':
        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project="mmgcg",
            name=args.experiment_name,
            # Track hyperparameters and run metadata
            config=asdict(config),
        )
        args.wandb_table = wandb.Table(columns=['step', 'suffix'])
        wandb.log({'suffix': args.wandb_table})

    try:
        result = run_mmgcg(
            args = args,
            model=model, 
            tokenizer=tokenizer, 
            image_processor=image_processor, 
            data=data,
            config=config,
        )
    except KeyboardInterrupt:
        if args.report_to == 'wandb':
            run.finish()


    print(result)