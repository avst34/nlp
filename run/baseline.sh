#!/bin/bash
#_SBATCH --mem=400m
#_SBATCH -c4
#_SBATCH --time=2:0:0
#_SBATCH --gres=gpu:3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#python3 $DIR/../main.py > ~/baseline.out 2>~/baseline.err
python3 $DIR/../main.py --dynet-gpu