#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tempr1
# ============ 清理环境（关键！） ============
pkill -9 -u $USER ray &> /dev/null
pkill -9 -u $USER vllm &> /dev/null
sleep 5
rm -rf /tmp/ray_$USER* ~/.ray &> /dev/null


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1,2,3  
export DATA_DIR='data/multitq_search2'
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# 内存和通信优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1                  # ✅ 关键：启用 P2P
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1                 # 阻塞等待
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0   
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 替代废弃的 NCCL_ASYNC_ERROR_HANDLING
export NCCL_LAUNCH_MODE=PARALLEL 
export NCCL_TIMEOUT=3600         # ✅ 关键：并行启动

# ✅ 新增：CPU 线程控制（避免过度竞争）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=1200
export RAY_object_store_memory=8000000000 
export RAY_BACKEND_LOG_LEVEL=info  # ✅ 改为 info（debug 太冗长）
export RAY_scheduler_spread_threshold=1.0  # ✅ 新增：更激进的任务调度
# Note: Ray worker nice value is set to 0 in main_ppo_format.py via _system_config

# ============ 实验配置 ============
WAND_PROJECT='Search-R1-MultiTQ-eval'
export BASE_MODEL='/verl_checkpoints/xxx'
export EXPERIMENT_NAME=multitq-grpo-eval

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn 

# ============ 训练脚本 ============
PYTHONUNBUFFERED=1 nice -n 0 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train_all.parquet \
    data.val_files=$DATA_DIR/test_all.parquet \
    data.train_data_num=null \
    data.val_data_num=500 \
    data.train_batch_size=60 \
    data.val_batch_size=500 \
    data.max_prompt_length=6200 \
    data.max_response_length=5000 \
    data.max_start_length=2000 \
    data.max_obs_length=2400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=10 \
    actor_rollout_ref.actor.ppo_micro_batch_size=5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=30 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log