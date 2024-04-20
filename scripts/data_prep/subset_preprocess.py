import random, os, string
from datetime import datetime
import numpy as np
from tqdm import tqdm

def create_sub_dataset_local(full_data_json, subset_size, seq_len, shuffle=True, return_json=False, dataset_path=None):
    random.seed(datetime.now().timestamp())
    rnd = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    json_path = f'/mnt/beegfs/alistgrp/mnikdan/tmpc4/{rnd}.jsonl'
    
    search_multiplier = 10
    indices = np.sort(np.random.choice(subset_size * search_multiplier, subset_size, replace=False))

    if dataset_path is None:
        dataset_path = f'/mnt/beegfs/alistgrp/mnikdan/tmpc4/{rnd}'
    train_split_path = os.path.join(dataset_path, 'train')

    print(f'creating a new sub dataset with size {len(indices)}...')

    print(f'subsetting the dataset...')
    lines = []
    idx_id = 0
    # print(f'looking for {indices[idx_id]}')
    with open(full_data_json, 'r') as f:
        for i, line in tqdm(enumerate(f), total=subset_size*search_multiplier):
            if i == indices[idx_id]:
                # print(f'found {indices[idx_id]}')
                idx_id += 1
                # print(f'now looking for {indices[idx_id]}')
                lines.append(line)
                if idx_id == len(indices):
                    break
            i += 1

    if shuffle:
        print(f'shuffling the dataset...')
        random.shuffle(lines)
    
    print(f'writing the new json file to {json_path}...')
    with open(json_path, 'w') as f:
        f.writelines(lines)
    
    if return_json:
        return json_path

    print(f'converting to the streaming format...')
    os.makedirs(train_split_path, exist_ok=True)
    command = f"/nfs/scistore19/alistgrp/mnikdan/miniconda3/envs/google/bin/python convert_dataset_json.py --path {json_path} --out_root {train_split_path} --concat_tokens {seq_len} --split train --tokenizer meta-llama/Llama-2-70b-hf --compression zstd"
    os.system(command)
    
    print(f'dataset created!')

    return dataset_path


seq_len = 4096
model_size = 125 # million parameters

num_tokens_needed = int(model_size / 125) * 2500000000
num_samples_needed = num_tokens_needed // 256 # assuming samples are 256 tokens long on average
print(f'num_tokens_needed: {num_tokens_needed}')
print(f'num_samples_needed: {num_samples_needed}')

create_sub_dataset_local(
    # full_data_json='/mnt/beegfs/alistgrp/mnikdan/DATA/c4_full_json/train-10M.jsonl',
    full_data_json='/mnt/beegfs/alistgrp/mnikdan/DATA/c4_full_json/train.jsonl',
    subset_size=num_samples_needed,
    seq_len=seq_len,
    return_json=False,
    dataset_path=f'/mnt/beegfs/alistgrp/mnikdan/chinch_cas_c4/llama70b_concat{seq_len}/'
)

# print(dataset_path)
