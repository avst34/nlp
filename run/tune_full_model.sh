#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

#source /cs/usr/aviramstern/lab/venvs/$1/bin/activate

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

export PYTHONPATH=$SRC_BASE
while true; do
    export RESULTS_PATH=$SRC_BASE/../full_model2.csv
    /cs/usr/aviramstern/lab/python_phoenix/bin/python3 -u  $SRC_BASE/run/lstm_mlp_baseline.py  $1
done
