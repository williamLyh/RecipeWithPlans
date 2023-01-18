python3 stage_planner.py --preprocessed_data_path='/home/yinhong/Documents/datasets/recipe1m+/preprocessed_data/' \
                        --model_saving_path='model-checkpoint/planner_results/' \
                        --train=True \
                        --lr=8e-5 \
                        --l2_decay=0.01 \
                        --epoch=2 \
                        --batch_size=256 \
                        --save_steps=2000 \
                        --eval_steps=2000 \
                        --warmup_steps=200