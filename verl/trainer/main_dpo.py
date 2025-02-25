# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP DPO Trainer with Ray-based single controller.
This trainer supports model-agnostic initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional, List, Callable, Any, Tuple
from copy import deepcopy

import ray
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
import wandb
import transformers
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker

def generate_pairwise_data(data: DataProto, score_key: str = 'token_level_scores', top_k: int = 1) -> DataProto:
    """
    Generate pairwise data for DPO training based on scores.
    For each prompt, select the top-k and bottom-k responses to create pairs.
    
    Args:
        data: DataProto containing batch data
        score_key: Key for scores in the batch
        top_k: Number of top/bottom responses to pair
        
    Returns:
        DataProto with pairwise data
    """
    batch = data.batch
    non_tensor_batch = data.non_tensor_batch
    
    # Get unique prompts and their indices
    prompts = non_tensor_batch.get('prompts', [])
    if not prompts:
        # If prompts not available, use input_ids up to input_token_len as proxy
        if 'input_token_len' in batch:
            prompts = []
            for i, length in enumerate(batch['input_token_len']):
                prompt_ids = batch['input_ids'][i, :length].tolist()
                prompts.append(tuple(prompt_ids))  # Use tuple for hashability
    
    # Group indices by prompt
    unique_prompts = {}
    for i in range(len(prompts)):
        prompt = prompts[i]
        if prompt not in unique_prompts:
            unique_prompts[prompt] = []
        unique_prompts[prompt].append(i)
    
    # Calculate sequence-level scores by summing token-level scores
    if score_key in batch:
        scores = batch[score_key].sum(dim=-1)  # (batch_size,)
    else:
        # Fallback if token_level_scores not available
        scores = batch.get('rewards', torch.zeros(len(prompts), device=batch['input_ids'].device))
    
    # Create pairs based on scores
    chosen_indices = []
    rejected_indices = []
    
    for prompt, indices in unique_prompts.items():
        if len(indices) < 2:
            continue
            
        # Sort indices by score
        sorted_indices = sorted(indices, key=lambda i: scores[i].item(), reverse=True)
        
        # Create pairs: top-k with bottom-k
        for i in range(min(top_k, len(sorted_indices) // 2)):
            chosen_idx = sorted_indices[i]
            rejected_idx = sorted_indices[-(i+1)]
            
            chosen_indices.append(chosen_idx)
            rejected_indices.append(rejected_idx)
    
    # Create new batch with chosen/rejected pairs
    pair_data = DataProto()
    
    # Add chosen data
    chosen_data = data.select(chosen_indices)
    for k, v in chosen_data.batch.items():
        pair_data.batch[f'chosen_{k}'] = v
    
    # Add rejected data
    rejected_data = data.select(rejected_indices)
    for k, v in rejected_data.batch.items():
        pair_data.batch[f'rejected_{k}'] = v
    
    # Add metadata
    pair_data.non_tensor_batch['pair_ids'] = list(zip(chosen_indices, rejected_indices))
    
    return pair_data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def compute_timing_metrics(batch, timing_raw):
    """Compute timing metrics per token"""
    # Calculate total tokens
    if 'attention_mask' in batch.batch:
        num_tokens = torch.sum(batch.batch['attention_mask']).item()
    else:
        # For pairwise data
        chosen_mask = batch.batch.get('chosen_attention_mask', None)
        rejected_mask = batch.batch.get('rejected_attention_mask', None)
        num_tokens = 0
        if chosen_mask is not None:
            num_tokens += torch.sum(chosen_mask).item()
        if rejected_mask is not None:
            num_tokens += torch.sum(rejected_mask).item()
    
    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens 
            for name in timing_raw.keys() if num_tokens > 0
        },
    }


def reduce_metrics(metrics: dict):
    """Reduce metrics by taking the mean"""
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """
    Compute DPO loss as described in the paper "Direct Preference Optimization: 
    Your Language Model is Secretly a Reward Model"
    
    Args:
        policy_chosen_logps: Log probs from policy model for chosen responses
        policy_rejected_logps: Log probs from policy model for rejected responses
        reference_chosen_logps: Log probs from reference model for chosen responses
        reference_rejected_logps: Log probs from reference model for rejected responses
        beta: Temperature parameter for the DPO loss
        
    Returns:
        DPO loss and advantages
    """
    # Compute the log ratios between policy and reference model
    chosen_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_ratio = policy_rejected_logps - reference_rejected_logps
    
    # Compute the implied reward
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # Compute the DPO loss (negative log sigmoid of the logits)
    losses = -F.logsigmoid(logits)
    
    return losses.mean(), logits

class RayDPOTrainer(object):
    """
    DPO Trainer using Ray for distributed training.
    """
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please use only one.")

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch."
            )

        print("[validate_config] Configuration checks passed!")

    def _create_dataloader(self):
        # Create training dataloader
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        
        # Use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        # Skip validation dataloader creation since we don't have validation data
        self.val_dataset = None
        self.val_dataloader = None

        assert len(self.train_dataloader) >= 1
        # assert len(self.val_dataloader) >= 1  # Remove this assertion

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        # print(f'Size of val dataloader: {len(self.val_dataloader)}')  # Remove this line

        # Inject total_training_steps to actor optim_config
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout.actor.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb
        
        Major reason is to keep track of validation generations from base model"""
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        """Run validation and return metrics"""
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Skip validation for model-based reward models
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # Pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # Unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # Evaluate using reward_function or reward model
            if self.use_rm:
                # Pad to be divisible by reward model world size
                test_batch_padded, rm_pad_size = pad_dataproto_to_divisor(test_batch, self.rm_wg.world_size)
                scored_batch_padded = self.rm_wg.compute_rm_score(test_batch_padded)
                scored_batch = unpad_dataproto(scored_batch_padded, pad_size=rm_pad_size)
                reward_tensor = scored_batch.batch['token_level_scores']
            else:
                # Evaluate using reward_function
                reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # Evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # Reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def init_workers(self):
        """Initialize resource pool and worker groups"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # Create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # Create a reward model if needed
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # Initialize WorkerGroups
        all_wg = {}
        self.wg_dicts = []
        # Remove any resource pools with empty class dictionaries
        empty_pools = [pool for pool, class_dict in self.resource_pool_to_cls.items() if not class_dict]
        for pool in empty_pools:
            self.resource_pool_to_cls.pop(pool)

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f'class_dict: {class_dict}')
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # Keep the reference of WorkerDict to support ray >= 2.31
            self.wg_dicts.append(wg_dict)

        if self.use_reference_policy:
            self.ref_policy_wg: DPOActorRolloutWorker = all_wg['ref']
            self.ref_policy_wg.init_model()
            self.ref_policy_wg.eval_model()

        if self.use_rm:
            self.rm_wg: RewardModelWorker = all_wg['rm']
            self.rm_wg.init_model()

        # Create actor_rollout at the end for better memory estimation
        self.actor_rollout_wg: DPOActorRolloutWorker = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        """Save model checkpoints"""
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # Save dataloader state
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # Latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """Load model checkpoints"""
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # Load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # Find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
                    
        print(f'Load from checkpoint folder: {global_step_folder}')
        # Set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        
        # Load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # Load dataloader
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def compute_dpo_metrics(self, batch):
        """Compute metrics for DPO training"""
        metrics = {}
        
        # Extract chosen and rejected logprobs
        chosen_logprobs = batch.batch.get('chosen_logprobs', None)
        rejected_logprobs = batch.batch.get('rejected_logprobs', None)
        
        if chosen_logprobs is not None and rejected_logprobs is not None:
            # Calculate policy advantage
            policy_advantages = chosen_logprobs - rejected_logprobs
            
            # Calculate reference advantage
            ref_chosen_logprobs = batch.batch.get('ref_chosen_logprobs', None)
            ref_rejected_logprobs = batch.batch.get('ref_rejected_logprobs', None)
            
            if ref_chosen_logprobs is not None and ref_rejected_logprobs is not None:
                ref_advantages = ref_chosen_logprobs - ref_rejected_logprobs
                
                # Calculate KL divergence from reference policy
                chosen_kl = ref_chosen_logprobs - chosen_logprobs
                rejected_kl = ref_rejected_logprobs - rejected_logprobs
                
                # Add metrics
                metrics['dpo/policy_advantage'] = policy_advantages.mean().item()
                metrics['dpo/ref_advantage'] = ref_advantages.mean().item()
                metrics['dpo/chosen_kl'] = chosen_kl.mean().item()
                metrics['dpo/rejected_kl'] = rejected_kl.mean().item()
                metrics['dpo/avg_kl'] = (chosen_kl.mean() + rejected_kl.mean()).item() / 2
                
                # Calculate accuracy (how often policy prefers the chosen response)
                accuracy = (policy_advantages > 0).float().mean().item()
                metrics['dpo/accuracy'] = accuracy
        
        return metrics

    def train(self):
        """Main training loop for DPO"""
        self.init_workers()
        
        # Initialize global step counter
        self.global_steps = 0
        
        # Load checkpoint if needed
        self._load_checkpoint()
        
        # Training loop
        for epoch in range(self.config.trainer.total_epochs):
            print(f"Starting epoch {epoch}")
            
            for batch_idx, batch_data in enumerate(self.train_dataloader):
                # Skip steps that were already processed
                if self.global_steps < self.config.trainer.skip_steps:
                    self.global_steps += 1
                    continue
                
                # Check if we've reached the total training steps
                if self.global_steps >= self.total_training_steps:
                    print(f"Reached total training steps {self.total_training_steps}, stopping training")
                    return
                
                # Initialize metrics and timing
                metrics = {}
                timing_raw = {}
                
                # Convert batch data to DataProto
                with _timer('data_preparation', timing_raw):
                    batch = DataProto.from_single_dict(batch_data)
                    
                    # Balance batch for better performance
                    self._balance_batch(batch, metrics)
                
                # Generate sequences using actor model
                with _timer('generate_sequences', timing_raw):
                    # Prepare batch for generation
                    gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'do_sample': True,
                        'temperature': self.config.actor_rollout.rollout.temperature,
                        'top_p': self.config.actor_rollout.rollout.top_p,
                        'max_new_tokens': self.config.actor_rollout.rollout.max_new_tokens,
                        'recompute_log_prob': False,
                    }
                    
                    # Pad batch to be divisible by dp_size
                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    
                    # Generate sequences
                    output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                    
                    # Unpad the output
                    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
                    
                    # Merge original batch with generated outputs
                    batch = batch.union(output_gen_batch)
                
                # Compute rewards/scores for generated sequences
                with _timer('compute_rewards', timing_raw):
                    if self.use_rm:
                        # Use reward model to score sequences
                        rm_batch = deepcopy(batch)
                        rm_batch_padded, pad_size = pad_dataproto_to_divisor(rm_batch, self.rm_wg.world_size)
                        scored_batch_padded = self.rm_wg.compute_rm_score(rm_batch_padded)
                        scored_batch = unpad_dataproto(scored_batch_padded, pad_size=pad_size)
                        batch = batch.union(scored_batch)
                    else:
                        # Use reward function to score sequences
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                
                # Generate pairwise data for DPO training
                with _timer('generate_pairwise_data', timing_raw):
                    pair_batch = generate_pairwise_data(
                        batch, 
                        score_key='token_level_scores',
                        top_k=self.config.actor_rollout.actor.get('dpo_top_k_pairs', 1)
                    )
                
                # Compute reference policy log probabilities if using reference model
                with _timer('compute_ref_logprobs', timing_raw):
                    if self.use_reference_policy:
                        with torch.no_grad():
                            ref_batch = deepcopy(pair_batch)
                            ref_batch_padded, pad_size = pad_dataproto_to_divisor(ref_batch, self.ref_policy_wg.world_size)
                            
                            # Compute log probs for chosen responses
                            chosen_ref_batch = ref_batch_padded.select_keys_with_prefix('chosen_')
                            chosen_ref_batch.rename_keys_with_prefix('chosen_', '')
                            ref_chosen_batch_padded = self.ref_policy_wg.compute_logprobs(chosen_ref_batch)
                            ref_chosen_batch = unpad_dataproto(ref_chosen_batch_padded, pad_size=pad_size)
                            
                            # Compute log probs for rejected responses
                            rejected_ref_batch = ref_batch_padded.select_keys_with_prefix('rejected_')
                            rejected_ref_batch.rename_keys_with_prefix('rejected_', '')
                            ref_rejected_batch_padded = self.ref_policy_wg.compute_logprobs(rejected_ref_batch)
                            ref_rejected_batch = unpad_dataproto(ref_rejected_batch_padded, pad_size=pad_size)
                            
                            # Add reference log probs to pair batch
                            pair_batch.batch['ref_chosen_logprobs'] = ref_chosen_batch.batch['logprobs']
                            pair_batch.batch['ref_rejected_logprobs'] = ref_rejected_batch.batch['logprobs']
                
                # Update policy using DPO
                with _timer('dpo_update', timing_raw):
                    # Prepare batch for DPO update
                    dpo_batch_padded, pad_size = pad_dataproto_to_divisor(pair_batch, self.actor_rollout_wg.world_size)
                    
                    # Update policy
                    dpo_metrics_padded = self.actor_rollout_wg.dpo_update(dpo_batch_padded)
                    dpo_metrics = unpad_dataproto(dpo_metrics_padded, pad_size=pad_size)
                    
                    # Add DPO metrics
                    for k, v in dpo_metrics.batch.items():
                        if isinstance(v, torch.Tensor):
                            metrics[f'dpo/{k}'] = v.mean().item()
                
                # Compute additional metrics
                with _timer('compute_metrics', timing_raw):
                    dpo_metrics = self.compute_dpo_metrics(pair_batch)
                    metrics.update(dpo_metrics)
                    
                    # Add timing metrics
                    timing_metrics = compute_timing_metrics(batch, timing_raw)
                    metrics.update(timing_metrics)
                
                # Log metrics
                if self.global_steps % self.config.trainer.log_every_n_steps == 0:
                    print(f"Step {self.global_steps}/{self.total_training_steps}, Metrics: {metrics}")
                    
                    # Log to wandb if configured
                    if 'wandb' in self.config.trainer.logger:
                        wandb.log(metrics, step=self.global_steps)
                
                # Skip validation since we don't have validation data
                # if self.global_steps % self.config.trainer.val_check_interval == 0:
                #     print("Running validation...")
                #     val_metrics = self._validate()
                #     
                #     # Log validation metrics
                #     if 'wandb' in self.config.trainer.logger:
                #         wandb.log(val_metrics, step=self.global_steps)
                
                # Save checkpoint
                if self.global_steps % self.config.trainer.save_every_n_steps == 0:
                    print(f"Saving checkpoint at step {self.global_steps}")
                    self._save_checkpoint()
                
                # Increment global step counter
                self.global_steps += 1
            
            # Save checkpoint at the end of each epoch
            self._save_checkpoint()
        
        # Final checkpoint
        self._save_checkpoint()
        print("Training completed!")

class DPOActorRolloutWorker(ActorRolloutRefWorker):
    """
    Worker implementation for DPO training that handles both actor model updates and sequence generation.
    This would be the implementation of role_worker_mapping[Role.ActorRollout].
    """
    def __init__(self, config, role='actor_rollout'):
        super().__init__(config, role)
        self.config = config
        self.role = role
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.beta = config.actor.get('dpo_beta', 0.1)  # KL penalty coefficient
        
    def compute_logprobs(self, batch):
        """Compute log probabilities for sequences"""
        input_ids = batch.batch['input_ids']
        attention_mask = batch.batch['attention_mask']
        
        # Forward pass with no loss computation
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Extract token-level log probabilities
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get the log probs of the next tokens
            shifted_input_ids = input_ids[:, 1:].clone()
            shifted_log_probs = log_probs[:, :-1].gather(
                dim=-1, 
                index=shifted_input_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create mask for response tokens only
            response_mask = torch.zeros_like(shifted_log_probs)
            
            # For each sequence, identify response tokens
            for i, length in enumerate(batch.batch.get('input_token_len', [])):
                if length < shifted_log_probs.shape[1]:
                    response_mask[i, length:] = 1.0
            
            # If input_token_len not provided, use attention mask
            if 'input_token_len' not in batch.batch:
                response_mask = attention_mask[:, 1:].float()
            
            # Mask out prompt tokens, keeping only response tokens
            masked_log_probs = shifted_log_probs * response_mask
            
            # Sum log probs for the entire sequence
            seq_log_probs = masked_log_probs.sum(dim=1)
            
            # Add to batch
            batch.batch['logprobs'] = seq_log_probs
            
            return batch
    
    def dpo_update(self, batch):
        """
        Update model using DPO loss with pairwise data.
        This is the implementation of the method called in the training loop.
        """
        self.model.train()
        metrics = DataProto()
        
        # Get chosen and rejected sequences
        chosen_batch = batch.select_keys_with_prefix('chosen_')
        chosen_batch.rename_keys_with_prefix('chosen_', '')
        
        rejected_batch = batch.select_keys_with_prefix('rejected_')
        rejected_batch.rename_keys_with_prefix('rejected_', '')
        
        # Compute policy log probs
        chosen_batch = self.compute_logprobs(chosen_batch)
        rejected_batch = self.compute_logprobs(rejected_batch)
        
        policy_chosen_logps = chosen_batch.batch['logprobs']
        policy_rejected_logps = rejected_batch.batch['logprobs']
        
        # Get reference model log probs if available
        if 'ref_chosen_logprobs' in batch.batch and 'ref_rejected_logprobs' in batch.batch:
            ref_chosen_logps = batch.batch['ref_chosen_logprobs']
            ref_rejected_logps = batch.batch['ref_rejected_logprobs']
        else:
            # If reference log probs are not provided, use the policy's own log probs
            # This effectively disables the KL penalty
            ref_chosen_logps = policy_chosen_logps.detach()
            ref_rejected_logps = policy_rejected_logps.detach()
        
        # Compute DPO loss
        loss, advantages = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.beta
        )
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.actor.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.actor.max_grad_norm
            )
        
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Log metrics
        metrics.batch['loss'] = loss.detach()
        metrics.batch['advantages'] = advantages.detach()
        metrics.batch['policy_chosen_logps'] = policy_chosen_logps.detach()
        metrics.batch['policy_rejected_logps'] = policy_rejected_logps.detach()
        
        # Calculate accuracy (how often policy prefers chosen over rejected)
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()
        metrics.batch['accuracy'] = accuracy.detach()
        
        return metrics

@hydra.main(config_path='config', config_name='dpo_trainer', version_base=None)
def main(config):
    """Entry point for DPO training"""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
        Role.RewardModel: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec, mapping)
    
    # Define worker types for each role
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(DPOActorRolloutWorker),
        Role.RefPolicy: ray.remote(DPOActorRolloutWorker)
    }
    
    # Add reward model if needed
    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.actor_rollout.model.path,
        use_fast=config.actor_rollout.actor.use_fast_tokenizer,
        padding_side='left'
    )
    
    # Create trainer
    trainer = RayDPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager
    )
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()