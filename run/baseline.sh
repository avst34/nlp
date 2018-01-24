#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --gres=gpu:1

source /cs/usr/aviramstern/lab/venvs/local/bin/activate

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while true; do
    #    #python3 $DIR/../main.py > ~/baseline.out
    # python3 $DIR/../main.py --dynet-gpu --dynet-autobatch > ~/baseline.out
    python3 /cs/usr/aviramstern/nlp_prod/main.py
done
