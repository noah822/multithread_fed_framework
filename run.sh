#! /bin/sh
export PYTHON_PATH=$PYTHON_PATH:./client:./dataset:./model:./server
python main.py --round 1 --local-epoch 5 \
                 --num-client 15 --num-thread 5 \
		 --csv-path /home/zihan/lab/dataset/clients \
		 --data-path /home/zihan/lab/dataset \
                 --use-tensorboard

