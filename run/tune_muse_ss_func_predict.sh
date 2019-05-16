#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

#source /cs/usr/aviramstern/lab/venvs/$1/bin/activate

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SRC_BASE
while true; do
    export RESULTS_PATH=$SRC_BASE/../chinese_results.csv
    /cs/usr/aviramstern/lab/python_phoenix/bin/python3 -u $SRC_BASE/run/muse_pss_func_predict.py $SRC_BASE/../chinese_results.csv
done
