python run_classifier.py --train_file data/reddit/sample/train.json \
                        --validation_file data/reddit/sample/valid.json \
                        --test_file data/reddit/sample/test.json \
                        --model_name_or_path bert-base-uncased \
                        --num_train_epochs 5 \
                        --output_dir saved_output \
                        --save_steps 10 \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 16 