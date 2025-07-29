#!/bin/bash
ulimit -n 8192
export CUDA_VISIBLE_DEVICES=0
export COMET_WORKSPACE="hgnn"

CASE="case9"
SCALE=0
PROJECT="perf_opf_weightingRebuttal"
DATASET="y_OPF"
MUTATIONS="load_relative"
BATCH_TRAIN=512
HP=10
CV=0
NB_TRAIN=8000
NB_VAL=2000
CLS=${1:-"gps"}
LR=0.001
DLR=0.5
AGG="mean"
HC="128:128"
EPOCHS=1000
OPF=0
RAY=1
SEED=${2:-20}
NB_WORKERS=0


VAL_MUTATIONS="line_nminus1"

DEVICE="cpu"
RAY=1
#python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cuda"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "0_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "0_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "0_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS


VAL_MUTATIONS="load_relative"

VAL_MUTATIONS="cost"
