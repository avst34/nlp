#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

source /cs/usr/aviramstern/lab/venvs/$1/bin/activate

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SRC_BASE
while true; do
    export RESULTS_PATH=$SRC_BASE/../muse_chinese_results.csv
    $2 -u $SRC_BASE/run/muse_eval_chinese.py $SRC_BASE/../muse_chinese_results.csv
done
