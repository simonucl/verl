#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=FLASH_ATTN


# Create output directory
OUTPUT_DIR="./dpo_qwen_2.5_instruct_3b"
mkdir -p $OUTPUT_DIR

# Run DPO training
python verl/trainer/main_dpo.py \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=100 \
  trainer.save_every_n_steps=500 \
  trainer.val_generations_to_log_to_wandb=10 \
  trainer.logger=["wandb"] \
  trainer.remove_previous_ckpt_in_save=true \
  trainer.resume_mode="disable" \
  \
  actor_rollout.hybrid_engine=true \
  actor_rollout.model.path="Qwen/Qwen2.5-3B-Instruct" \
  actor_rollout.actor.use_fast_tokenizer=true \
  actor_rollout.actor.dpo_beta=0.1 \
  actor_rollout.actor.dpo_top_k_pairs=1 \
  actor_rollout.actor.dpo_micro_batch_size_per_gpu=4 \
  actor_rollout.actor.max_grad_norm=1.0 \
  \
  actor_rollout.actor.optim.lr=5e-7 \
  actor_rollout.actor.optim.weight_decay=0.0 \
  actor_rollout.actor.optim.scheduler_type="cosine" \
  actor_rollout.actor.optim.warmup_steps=100 \
  \
  actor_rollout.rollout.temperature=0.7 \
  actor_rollout.rollout.top_p=0.9 \
  actor_rollout.rollout.max_new_tokens=1024 \
  actor_rollout.rollout.n=8 \
  \
  data.train_files=["/root/data/iter_ultrafb/train/train.parquet"] \
  data.prompt_key="prompt" \
  data.max_prompt_length=1024 \
  data.train_batch_size=16 \
  data.shuffle=true \
  data.seed=42 \
  data.return_raw_chat=false \
  \
  reward_model.enable=true \
  reward_model.model.path="RLHFlow/ArmoRM-Llama3-8B-v0.1" \
  reward_model.model.trust_remote_code=true \
  \
  wandb.project="qwen-2.5-dpo-training" \
  wandb.name="qwen-2.5-instruct-3b-dpo"