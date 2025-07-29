#!/bin/bash
ulimit -n 8192
export COMET_WORKSPACE="hgnn"

CASE="case118"
SCALE=0
PROJECT="perf_opf_weightingRebuttal"
DATASET="y_OPF"
MUTATIONS="load_relative"
#VAL_MUTATIONS="load_relative+line_nminus1+cost"
BATCH_TRAIN=128
HP=10
CV=0
NB_TRAIN=800
NB_VAL=200
CLS=${1:-"sage"}
LR=0.0001
DLR=0.5
AGG="mean"
HC="128:128"
EPOCHS=1000
OPF=1
RAY=1
SEED=${2:-20240625}
NB_WORKERS=0



### TOP

VAL_MUTATIONS="line_nminus1"
DEVICE="cpu"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

DEVICE="cpu"
VAL_MUTATIONS="cost"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS


VAL_MUTATIONS="load_relative"
DEVICE="cpu"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

### BOTTOM

VAL_MUTATIONS="line_nminus1"
DEVICE="cpu"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS 

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS 

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_0.5_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS 
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_2" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

DEVICE="cpu"
VAL_MUTATIONS="cost"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_0.5_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS 
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_2" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS


VAL_MUTATIONS="load_relative"
DEVICE="cpu"
RAY=1
python experiments/build_db.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

RAY=0
DEVICE="cpu"

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "2_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "random" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "uniform" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "3_2" --weighting "relative" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS

python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS  
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_0.5_1" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS 
python experiments/test_performances.py --cases $CASE --cls $CLS --hidden_channels $HC --epochs $EPOCHS --ray $RAY --cv_ratio $CV --num_samples $HP --mutations $MUTATIONS  --validation_mutations $VAL_MUTATIONS --scale $SCALE --device $DEVICE --dataset_type $DATASET --comet_name $PROJECT --opf $OPF --nb_train $NB_TRAIN --nb_val $NB_VAL --clamp_boundary 3 --use_physical_loss "1_2_0_1_2" --weighting "relative sup2ssl" --seed $SEED  --batch_train $BATCH_TRAIN --num_workers_train $NB_WORKERS
