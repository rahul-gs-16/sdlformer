# SDLFormer
Official pytorch implementation of SDLFormer: A Sparse and Dense Locality-enhanced Transformer for Accelerated MR Image Reconstruction accepted at MICCAI workshop MILLanD 2023 Medical Image Learning with noisy and Limited Data (https://arxiv.org/abs/2308.04262).

Run the train_valid_eval_knee_selfsuper.sh script file to Train, and evalueate the model in self supervised manner.
Run the train_valid_eval_knee_super.sh script file to Train, and evalueate the model in fully supervised manner.
In the script files fill the BASE_PATH, and EXP_DIR with parent folder of the dataset and the place where the results must be saved respectively.

Download weights of Uformer from https://github.com/ZhendongWang6/Uformer and put in the weights folder
