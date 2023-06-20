MODEL='kikinet'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
# DATASET_TYPE='coronal_pd_h5'
# DATASET_TYPE='coronal_pd_fs_h5'
DATASET_TYPE='axial_t2_h5'

# <<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
MASK_TYPE='cartesian'
ACC_FACTOR='4x'
NC=1

EXP_DIR='/home/fahim/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC}

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC}

BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
MASK_TYPE='cartesian'
ACC_FACTOR='5x'
NC=1

EXP_DIR='/home/fahim/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC}

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC}