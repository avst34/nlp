#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --gres=gpu:1

source /cs/usr/aviramstern/lab/venvs/local/bin/activate.csh

python3 /cs/usr/aviramstern/nlp_prod/process_tuner_result.py /cs/usr/aviramstern/lab/results.csv
