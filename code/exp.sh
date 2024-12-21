#!/bin/bash

python main.py --run_name impacts --cuda 3 --epochs 100 
python main.py --run_name random --cuda 3 --epochs 100 --random_mode 
python main.py --run_name baseline --cuda 3 --epochs 100 --drop_ratio 1

