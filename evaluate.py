import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm
from dataset import KneeDataDev
from data import transforms as T
import torch
from tqdm import tqdm
from sewar.full_ref import vifp
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

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


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
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return structural_similarity(gt,pred,multichannel=True, data_range=gt.max())

def vif_p(gt, pred):

    return vifp(gt, pred)



METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn,
    VIF=vif_p
    
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


    
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



def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in tqdm(args.target_path.iterdir()):
        #print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
#             sensitivity = torch.from_numpy(target['sensitivity'][:])
            target = torch.from_numpy(target[recons_key][:])
#             target_2 = T.complex_center_crop(target,(320,320)).unsqueeze(0)
#             target_2 = complex_img_pad(target_2,(640,484)).squeeze(0)#368 for all except axial_t2_h5 for which it's 484 
#             target = low_or_high_freq_recover(target_2,target_2,sensitivity)
            target = T.complex_abs(T.complex_center_crop(target,(320,320))).numpy()
            recons = recons['reconstruction'][:]
            recons = np.transpose(recons,[1,2,0])

            if len(target.shape) == 2: 
                target = np.expand_dims(target,2) # added for knee dataset 
            metrics.push(target, recons)
            
    return metrics


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
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()
    print (metrics_report)

    with open(args.report_path / 'report.txt','w') as f:
        f.write(metrics_report)

    #print(metrics)
