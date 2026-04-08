#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tempr1

# ============ 环境配置 ============
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3,4,5
export DATA_DIR='data/multitq_search2'
export WANDB_MODE=offline

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL配置
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============ 实验配置 ============
WAND_PROJECT='Search-R1-MultiTQ-Curriculum'

# ⚠️ 从Stage 1的checkpoint继续训练
STAGE1_CHECKPOINT='verl_checkpoints/xxx'
export BASE_MODEL=$STAGE1_CHECKPOINT
export EXPERIMENT_NAME=multitq-grpo-stage2

export VLLM_ATTENTION_BACKEND=XFORMERS

# ============ 训练脚本 ============
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_all.parquet \
    data.val_files=$DATA_DIR/test_all.parquet \
    data.train_data_num=null \
    data.val_data_num=30 \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_start_length=1500 \
    data.max_obs_length=1500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb','console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    +trainer.resume_step=40 \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=20 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log