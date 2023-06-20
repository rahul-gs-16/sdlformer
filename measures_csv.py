import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import pandas as pd 
from data import transforms as T
import torch
from torch.nn import functional as F

def complex_img_pad(im_crop, shape):

    _, h1, w1, _ = im_crop.shape 
    h2, w2 = shape

    
    h_dif = h2 - h1 
    w_dif = w2 - w1

    # how this will work for odd data

    h_dif_half = h_dif // 2
    w_dif_half = w_dif // 2

    im_crop_pad = F.pad(im_crop,[0,0,w_dif_half, w_dif_half, h_dif_half, h_dif_half])

    return im_crop_pad


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(gt,pred,data_range=gt.max())


def low_or_high_freq_recover(recons,mask,sens,low=False):
    
    lf_mask = torch.zeros(mask[:,:,0].unsqueeze(0).shape)
    lf_n = 12 #8 for mc_knee multicoil complex
    lf_mask[0,:lf_n,:lf_n] = 1.
    lf_mask[0,:lf_n,-lf_n:] = 1.
    lf_mask[0,-lf_n:,:lf_n] = 1.
    lf_mask[0,-lf_n:,-lf_n:] = 1.
    lf_mask = lf_mask.unsqueeze(-1)
    hf_mask = 1. - lf_mask
    recons = T.complex_multiply(recons[...,0].unsqueeze(0), recons[...,1].unsqueeze(0), 
                               sens[...,0], sens[...,1])
    recons_kspace = T.fft2(recons)
    if low:
        recons_lf_kspace = lf_mask * recons_kspace #F
        recons_lf = T.ifft2(recons_lf_kspace)
        recons_lf = T.complex_multiply(recons_lf[...,0], recons_lf[...,1], 
                                sens[...,0], 
                               -sens[...,1]).sum(dim=0)  
        target = recons_lf #F
    else:
        recons_hf_kspace = hf_mask * recons_kspace #F
        recons_hf = T.ifft2(recons_hf_kspace)
        recons_hf = T.complex_multiply(recons_hf[...,0], recons_hf[...,1], 
                                sens[...,0], 
                               -sens[...,1]).sum(dim=0) 
        target = recons_hf #F
        
    return target


def evaluate(args, recons_key,metrics_info):

    for tgt_file in tqdm(args.target_path.iterdir()):
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:
#             target = target[recons_key][:]
#             target = T.complex_abs(torch.from_numpy(target)).numpy()
            sensitivity = torch.from_numpy(target['sensitivity'][:])#F
            target = torch.from_numpy(target[recons_key][:])
            target_2 = T.complex_center_crop(target,(320,320)).unsqueeze(0)#F
            target_2 = complex_img_pad(target_2,(640,484)).squeeze(0)#368 for all except axial_t2_h5 for which it's 484 #F
            target = low_or_high_freq_recover(target_2,target_2,sensitivity)#F
            target = T.complex_abs(T.complex_center_crop(target,(320,320))).numpy()
            recons = recons['reconstruction'][:]
            recons = np.transpose(recons,[1,2,0])
            if len(target.shape) ==2 :
                target = np.expand_dims(target,2)

            no_slices = target.shape[-1]

            for index in range(no_slices):
                target_slice = target[:,:,index]
                recons_slice = recons[:,:,index]
                mse_slice  = round(mse(target_slice,recons_slice),5)
                nmse_slice = round(nmse(target_slice,recons_slice),5)
                psnr_slice = round(psnr(target_slice,recons_slice),2)
                ssim_slice = round(ssim(target_slice,recons_slice),4)

                metrics_info['MSE'].append(mse_slice)
                metrics_info['NMSE'].append(nmse_slice)
                metrics_info['PSNR'].append(psnr_slice)
                metrics_info['SSIM'].append(ssim_slice)
                metrics_info['VOLUME'].append(tgt_file.name)
                metrics_info['SLICE'].append(index)
        #break

    return metrics_info

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')

    args = parser.parse_args()

    recons_key = 'img_gt'

    metrics_info = {'VOLUME':[],'SLICE':[],'MSE':[],'NMSE':[],'PSNR':[],'SSIM':[]}

    metrics_info = evaluate(args,recons_key,metrics_info)
    csv_path     = args.report_path / 'hf_metrics.csv'
    df = pd.DataFrame(metrics_info)
    df.to_csv(csv_path)


