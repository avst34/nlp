#!/bin/sh
#SBATCH --mem=4096m
#SBATCH -c2
#SBATCH --time=2-0
##SBATCH --gres=gpu:1

SRC_BASE=/cs/usr/aviramstern/lab/nlp_prod

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SRC_BASE
while true; do
    #    #python3 $DIR/../run/tune_hcpd_model.py > ~/baseline.out
    # python3 $DIR/../main.py --dynet-gpu --dynet-autobatch > ~/baseline.out
    /cs/usr/aviramstern/lab/python_phoenix/bin/python3 -u $SRC_BASE/run/tune_hcpd_model_streusle_boknilev.py $SRC_BASE/../hcpd_tuning_streusle_boknilev_pipeline.csv
done
