#!/bin/bash

experiment_name=$1
model_directory=$2
num_gpu_devices=$3
cluster_name=$4

# old parameters
# #num_workers=$((num_gpu_devices*4))
# num_workers=0
# batch_size_per_device=64
# batch_size=$((num_gpu_devices*batch_size_per_device))

# printing values
# echo "experiment_name: $experiment_name"
# echo "data directory:  $data_directory"
# echo "experiment_id:   $experiment_id"
# echo "num_gpu_devices: $num_gpu_devices"
# echo "num_workers:     $num_workers"
# echo "batch_size:      $batch_size"
# echo "batch_size_per_device: $batch_size_per_device"

echo "checking nvidia-smi:"
nvidia-smi
#echo "checking nvidia-smi nvlink:"
#nvidia-smi nvlink --status
#echo "checking nvcc version:"
#nvcc --version
#echo "checking shared memory (/dev/shm):"
#df -h
#echo "running system check:"
#/usr/bin/python3 src/utils/system_check.py

#export NETRC=/staging/jaenmarquez/.config/wandb/.netrc
#wandb login --relogin b6cf381756cfeeb8e1d5a61ad946302465b56ad1


# copy model weights from /staging to local workspace
echo "copying model files:"
model_orig=$model_directory/darwin-7b_v2
model_dest=./models/darwin-7b_v2
mkdir -p $model_dest
#rsync -ah --progress /staging/jaenmarquez/data/imagenet/ILSVRC2012* ./models
cp -r $model_orig $model_dest

echo "running training job:"
export WANDB_API_KEY=b6cf381756cfeeb8e1d5a61ad946302465b56ad1

torchrun  --nproc_per_node=$num_gpu_devices --master_port=1212 train.py \
    --model_name_or_path $model_orig \
    --data_path datasets/drug_discovery/tabular/BROAD_REPURPOSING_DRUGS.json \
    --bf16 True \
    --output_dir $model_dest \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False

# ALTERNATIVELY: run in local mode
# docker run --gpus '"device=6,7"' ivanjaenm/llm4bio
# export CUDA_VISIBLE_DEVICES=6,7
# export DS_SKIP_CUDA_CHECK=1
# export WANDB_API_KEY=b6cf381756cfeeb8e1d5a61ad946302465b56ad1

# torchrun  --nproc_per_node=2 --master_port=1212 train.py \
#     --model_name_or_path models/darwin-7b_v2 \
#     --data_path datasets/drug_discovery/tabular/BROAD_REPURPOSING_DRUGS.json \
#     --bf16 True \
#     --output_dir models/darwin-7b_v2_finetuned \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 False