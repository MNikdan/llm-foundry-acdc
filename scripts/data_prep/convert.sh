export DATASET=squad
python convert_dataset_json.py \
  --path /mnt/beegfs/alistgrp/mnikdan/DATA/${DATASET}-train.jsonl \
  --out_root /mnt/beegfs/alistgrp/mnikdan/DATA/${DATASET} --split train \
  --tokenizer bert-base-uncased \
  --compression zstd
