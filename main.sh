
export CUDA_VISIBLE_DEVICES=0


###############
model_name="llava-v1.5-7b"
model_dir="/home/ghc240000/models/llava-1.5-7b-hf"
###############


postfix="-v4-main"


log_name="mmgcg-$model_name$postfix" 
# a looser condition for each step compared to v1.3
# num_step/sample: 30 -> 20


log_path="./train_logs/$log_name"



nohup \
python -u ./train_src/train_mm.py \
    --model_dir $model_dir \
    --num_epochs 1 \
    --optim_str_init "x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x" \
    --data_path ./data/mixed_ratio0.20.json \
    --search_width 64 \
    --loss_threshold 0.3 \
    --num_steps_per_sample 40 \
    --topk 64 \
    --dpo_loss_weight 0. \
    --prompt_version $model_name \
    --output_dir ./outputs \
    --experiment_name "$log_name" \
    --report_to wandb > $log_path 2>&1 &
    # --report_to none

tail -f $log_path

# "McBatchagh SavCombine Bernard WatsonTheanson Du altres texture W extrem la Baltimore Houston],[urm"