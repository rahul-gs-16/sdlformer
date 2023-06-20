MODEL='kiki_prop_net_dc'
# BASE_PATH=<Parent folder of dataset> # Fill and uncomment before running 
DATASET_TYPE='coronal_pd_h5'
# DATASET_TYPE='coronal_pd_fs_h5'
# DATASET_TYPE='axial_t2_h5'

BATCH_SIZE=1
NUM_EPOCHS=1
DEVICE='cuda:0'
ACC_FACTOR='4x'

NC=1
MASK_TYPE='cartesian'
# EXP_DIR=<Location to save the weights and output> # Fill the experiment directory and uncomment it before running
TRAIN_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'
SELFSUP_MASK=${BASE_PATH}'/selfsup_masks_1/'

echo python train_selfsup.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC} --selfsup-mask-path ${SELFSUP_MASK}
python train_selfsup.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC} --selfsup-mask-path ${SELFSUP_MASK}

# Validation
CHECKPOINT=${EXP_DIR}'/best_model.pt'
OUT_DIR=${EXP_DIR}'/results'
mkdir -p ${OUT_DIR}
DATA_PATH=${VALIDATION_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}


# Evaluate
TARGET_PATH=${VALIDATION_PATH}
PREDICTIONS_PATH=${OUT_DIR}
REPORT_PATH=${EXP_DIR}
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 


# Measure
TARGET_PATH=${VALIDATION_PATH}
PREDICTIONS_PATH=${OUT_DIR}
REPORT_PATH=${EXP_DIR}
python measures_csv_pd_pdfs.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}



