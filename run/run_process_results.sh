#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SRC_BASE
#while true; do
/cs/labs/oabend/aviramstern/python_phoenix/bin/python3 -u $SRC_BASE/process_tuner_result.py $1 $2 $3 $4 $5
#done
