MODEL='kikinet_50-50-split'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
DATASET_TYPE='coronal_pd_h5'
# DATASET_TYPE='coronal_pd_fs_h5'
# DATASET_TYPE='axial_t2_h5'

BATCH_SIZE=1
DEVICE='cuda:0'
MASK_TYPE='cartesian'
ACC_FACTOR='4x'
NC=1

CHECKPOINT='/media/Data/MRI/fahim_experiments/mc_ocucrn_selfsup/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/nc'${NC}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/Data/MRI/fahim_experiments/mc_ocucrn_selfsup/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/nc'${NC}'/'${MODEL}'/results'
DATA_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

EXP_DIR='/media/Data/MRI/fahim_experiments/mc_ocucrn_selfsup/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/nc'${NC}'/'${MODEL}

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}