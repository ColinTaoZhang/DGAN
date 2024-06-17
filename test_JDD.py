import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.MIRNet_model import MIRNet
from models import *
from dataloaders.data_rgb import get_validation_data, get_test_data
import utils
from skimage import img_as_ubyte
import scipy.io as scio
import h5py

parser = argparse.ArgumentParser(description='RAW JDD evaluation')
parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/test/', type=str, help='Directory for results')
parser.add_argument('--weights_J', default='./checkpoints/JDD/models/Unet_G_JDD/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

# test_dataset = get_validation_data(args.input_dir)
test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=1, drop_last=False)



model_restoration = U_Net_G(ps=False)
model_demosaic = U_Net_G()

checkpoint = torch.load(args.weights_J)
model_restoration.load_state_dict(checkpoint["state_dict_DN"])
model_demosaic.load_state_dict(checkpoint["state_dict_DM"])
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_demosaic.cuda()

model_restoration=nn.DataParallel(model_restoration)
model_demosaic=nn.DataParallel(model_demosaic)

model_restoration.eval()
model_demosaic.eval()


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]
        rgb_denoised, rgb_denoised_green = model_restoration(rgb_noisy)
        rgb_restored, rgb_restored_green = model_demosaic(rgb_denoised, rgb_denoised_green)

        ####### compute
        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()/10
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()/10
        psnr, ssim = utils.batch_metric(rgb_restored, rgb_gt)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                # utils.save_img(args.result_dir + filenames[batch][-8:-4] + '_restored.png', denoised_img)
                utils.save_img(args.result_dir + filenames[batch].split('/')[-1][:-4] + '.png', denoised_img)
        
mean_psnr = sum(psnr_val_rgb)/len(psnr_val_rgb)
mean_ssim = sum(ssim_val_rgb)/len(ssim_val_rgb)
with open(args.result_dir + 'metric.txt', 'a') as fd:
    fd.write('\nPSNR: {}, SSIM: {}\n'.format(mean_psnr, mean_ssim))
print("PSNR: %.4f, SSIM: %.4f" %(mean_psnr, mean_ssim))
