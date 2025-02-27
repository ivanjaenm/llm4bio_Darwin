#!/bin/bash

# activate environment
#conda activate bias4ml

# run a single experiment
#python src/train.py experiment=exp_mlp-mnist.yaml

# run all imagenet experiments inside configs/experiment
# using the same chtc cluster - this is to avoid copy/extraction/preprocesing 
#python src/train.py --multirun 'experiment=glob(exp_*-imagenet)'

experiment_name=$1
model_directory=$2
num_gpu_devices=$3
cluster_name=$4
# #num_workers=$((num_gpu_devices*4))
# num_workers=0
# batch_size_per_device=64
# batch_size=$((num_gpu_devices*batch_size_per_device))

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

#python src/train.py --multirun 'experiment=glob(*imagenet*)'

echo "running training job:"

# /usr/bin/python3 src/train.py \
#                  experiment=$experiment_name \
#                  paths.data_dir=$model_directory \
#                  data.num_workers=$num_workers \
#                  data.pin_memory=false \
#                  data.batch_size=$batch_size \
#                  trainer=ddp \
#                  trainer.devices=$num_gpu_devices \
#                  trainer.max_epochs=50 \
#                  trainer.num_nodes=1

export DS_SKIP_CUDA_CHECK=1
export WANDB_API_KEY=b6cf381756cfeeb8e1d5a61ad946302465b56ad1

torchrun  --nproc_per_node=$num_gpu_devices --master_port=1212 train.py \
    --model_name_or_path $model_directory/darwin-7b_v2 \
    --data_path datasets/drug_discovery/tabular/BROAD_REPURPOSING_DRUGS.json \
    --bf16 True \
    --output_dir $model_directory/darwin-7b_v2_finetuned \
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
