#! /bin/bash
TB_PATH=/home/zihan/lab/late_fusion_exp/runs_baseline_seed_272
SAVE_PATH=/home/zihan/lab/late_fusion_exp/ckpt_baseline_seed_272
export PYTHON_PATH=$PYTHON_PATH:./client:./dataset:./model:./server
python main.py --round 50 --local-epoch 5 \
                 --num-client 15 --num-thread 1 \
                 --csv-path /home/zihan/lab/dataset/clients \
                 --data-path /home/zihan/lab/dataset \
                 --num-workers 32 \
                 --use-tensorboard --tensorboard-path $TB_PATH \
                 --ckpt-path $SAVE_PATH \
                 --data-parallel \
                 --gpu-ids 1,2,3 \
                 
