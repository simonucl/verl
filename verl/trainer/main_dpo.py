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

import ray
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from codetiming import Timer
import wandb
import transformers
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.trainer.dpo.ray_trainer import RayDPOTrainer
from verl.utils.import_utils import import_external_libs
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from omegaconf import DictConfig, open_dict

class DPORefActorRolloutWorker(ActorRolloutRefWorker):
    def __init__(self, config, role='actor_rollout'):
        super().__init__(config, role)
        self.config = config
        self.role = role
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.beta = config.actor.get('dpo_beta', 0.1)  # KL penalty coefficient

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self.config.ref.model.path is not None:
            model_path = self.config.ref.model.path
            model_config = self.config.ref.model
        else:
            model_path = self.config.model.path
            model_config = self.config.model
        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=model_path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=model_config.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=model_config.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        torch.cuda.empty_cache()
        
        
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
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def dpo_update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_dpo_policy', logger=None) as timer:
                metrics = self.actor.update_dpo_policy(data=data)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            for key, value in metrics.items():
                if isinstance(value, list):
                    metrics[key] = np.mean(value)
            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output
        

@hydra.main(config_path='config', config_name='dpo_trainer', version_base=None)
def main(config):
    run_dpo(config)

def run_dpo(config):
    """Entry point for DPO training"""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    
    ray.get(main_task.remote(config))

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
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
        Role.RefPolicy: ray.remote(DPORefActorRolloutWorker)
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