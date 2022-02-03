#!/bin/sh

python run_mlm.py --train_file ../data/twitter/UK/2019/08/train.txt \
                        --validation_file ../data/twitter/UK/2019/08/dev.txt \
                        --model_name_or_path vinai/bertweet-base\
                        --num_train_epochs 1 \
                        --output_dir saved_output \
                        --save_steps 1 \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 16 \
                        --max_train_samples 1000 \
                        --max_eval_samples 500 \
                        --line_by_line True 