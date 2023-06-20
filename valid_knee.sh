MODEL='kikinet'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
# DATASET_TYPE='coronal_pd_h5'
# DATASET_TYPE='coronal_pd_fs_h5'
DATASET_TYPE='axial_t2_h5'

BATCH_SIZE=1
DEVICE='cuda:0'
MASK_TYPE='cartesian'
ACC_FACTOR='4x'
CHECKPOINT='/media/Data/MRI/fahim_experiments/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/Data/MRI/fahim_experiments/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}'/hf_results'
DATA_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}

BATCH_SIZE=1
DEVICE='cuda:0'
MASK_TYPE='cartesian'
ACC_FACTOR='5x'
CHECKPOINT='/media/Data/MRI/fahim_experiments/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/Data/MRI/fahim_experiments/mc_kikinet/exp/'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'/'${MODEL}'/hf_results'
DATA_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}