from datasets import load_dataset
import os

target = '/mnt/beegfs/alistgrp/mnikdan/DATA/'
dataset = load_dataset("squad")
for split, split_dataset in dataset.items():
    split_dataset.to_json(os.path.join(target, f"squad-{split}.jsonl"))