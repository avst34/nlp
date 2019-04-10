#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

#source /cs/usr/aviramstern/lab/venvs/$1/bin/activate

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

export PYTHONPATH=$SRC_BASE
/cs/usr/aviramstern/lab/python_phoenix/bin/python3 -u  $SRC_BASE/run/predict_boknilev_supersenses.py
