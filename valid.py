import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import KneeDataDev
from architecture import KIKInet
import h5py
from tqdm import tqdm
from data import transforms as T
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

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def create_data_loaders(args):

    data = KneeDataDev(args.data_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,)

    return data_loader


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    n_iter = 1

    model = KIKInet(n_iter=n_iter).to(args.device)
    model.load_state_dict(checkpoint['model'])
    
    return model


def low_or_high_freq_recover(recons,mask,sens,low=False):
    
    lf_mask = torch.zeros(mask[0,:,:,0].unsqueeze(0).shape).cuda()
    lf_n = 12 #8 for mc_knee multicoil complex
    lf_mask[0,:lf_n,:lf_n] = 1.
    lf_mask[0,:lf_n,-lf_n:] = 1.
    lf_mask[0,-lf_n:,:lf_n] = 1.
    lf_mask[0,-lf_n:,-lf_n:] = 1.
    lf_mask = lf_mask.unsqueeze(-1).unsqueeze(0)
    hf_mask = 1. - lf_mask
    recons = T.complex_multiply(recons[...,0].unsqueeze(1), recons[...,1].unsqueeze(1), 
                               sens[...,0], sens[...,1])
    recons_kspace = T.fft2(recons)
    if low:
        recons_lf_kspace = lf_mask * recons_kspace #F
        recons_lf = T.ifft2(recons_lf_kspace)
        recons_lf = T.complex_multiply(recons_lf[...,0], recons_lf[...,1], 
                                sens[...,0], 
                               -sens[...,1]).sum(dim=1)  
        target = recons_lf #F
    else:
        recons_hf_kspace = hf_mask * recons_kspace #F
        recons_hf = T.ifft2(recons_hf_kspace)
        recons_hf = T.complex_multiply(recons_hf[...,0], recons_hf[...,1], 
                                sens[...,0], 
                               -sens[...,1]).sum(dim=1) 
        target = recons_hf #F
        
    return target


def run_unet(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            _,_,rawdata_und,masks,sensitivity,fnames = data

            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
            sensitivity = sensitivity.to(args.device)
            
            output = model(rawdata_und,masks,sensitivity)
#             output = T.complex_center_crop(img_und,(320,320))
#             output = complex_img_pad(output,(640,368))#368 for all except axial_t2_h5 for which it's 484
#             output = low_or_high_freq_recover(output,output,sensitivity)
#             output = T.complex_center_crop(output,(320,320))
            recons = T.complex_abs(output).to('cpu')
            
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append(recons[i].numpy())

        reconstructions = {
            fname: np.stack([pred for pred in sorted(slice_preds)])
            for fname, slice_preds in reconstructions.items()
        }
            
    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=1, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
