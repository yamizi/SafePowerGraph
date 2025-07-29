#!/bin/bash
ulimit -n 8192
export CUDA_VISIBLE_DEVICES=0
export COMET_WORKSPACE="hgnn"

CASE="case118"
SCALE=0
PROJECT="perf_opf_weightingRebuttalV2"
DATASET="y_OPF"
MUTATIONS="load_relative"
BATCH_TRAIN=512
HP=10
CV=0
NB_TRAIN=8000
NB_VAL=2000
CLS=${1:-"gat"}
LR=0.001
DLR=0.5
AGG="mean"
HC="128:128"
EPOCHS=1000
OPF=1
RAY=1
SEED=${3:-20}
NB_WORKERS=0
VAL_MUTATIONS="${2:-line_nminus1}"

DEVICE="cpu"
RAY=1

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 1 --use_physical_loss "0_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
#python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 2 --use_physical_loss "0_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
#python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "0_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
