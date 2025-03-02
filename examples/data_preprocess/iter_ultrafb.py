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
Preprocess the RLHFlow/iterative-prompt-v1-iter1-20K dataset for training.
"""
import argparse
import os

import pandas as pd
from datasets import load_dataset, concatenate_datasets

from tqdm.auto import tqdm

from verl.utils.fs import copy, makedirs

def generate_rl_dataset(target_hdfs_path_dir, local_dir='~/data/iter_ultrafb/rl', total_iter=6):
    datasets = []
    for iter in range(1, total_iter + 1):
        dataset = load_dataset(f'RLHFlow/iterative-prompt-v1-iter{iter}-20K')
        train_dataset = dataset['train']

        data_source = f'RLHFlow/iterative-prompt-v1-iter{iter}-20K'

        def make_map_fn(split):
            def process_fn(example, idx):
                # The context_messages is already in the format [{"content": "xxx", "role": "user"}]
                prompt = example['context_messages']
                
                data = {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": "alignment",
                    "reward_model": {
                        "style": "model",
                        "ground_truth": ""  # should not be used
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        datasets.append(train_dataset)

    train_dataset = concatenate_datasets(datasets)

    print(f"Length of train_dataset: {len(train_dataset)}")
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    local_path = os.path.join(local_dir, 'train.parquet')
    train_dataset.to_parquet(local_path)

    if target_hdfs_path_dir is not None:
        hdfs_dir = target_hdfs_path_dir + '/' + 'train.parquet'
        makedirs(hdfs_dir)

        copy(local_path, hdfs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default='~/data/iter_ultrafb')
    parser.add_argument('--hdfs_dir', type=str, required=False, default=None)
    parser.add_argument('--total_iter', type=int, default=6)

    args = parser.parse_args()


    generate_rl_dataset(args.hdfs_dir, os.path.join(args.local_dir, 'train'), args.total_iter)
