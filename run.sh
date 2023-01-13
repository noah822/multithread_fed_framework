#! /bin/sh
export PYTHON_PATH=$PYTHON_PATH:./client:./dataset:./model:./server
python fed_train.py --round 100 --local-epoch 5 \
                 --num-client 10 --num-thread 10 \
                 --mode resume --resume-ckpt-path . \
                 --user-tensorboard

