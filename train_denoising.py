import os
import cv2
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from models import *

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler

# from thop import profile
from torchstat import stat

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

######### Model ###########
model_restoration = U_Net_G(ps=True)

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-start_epoch+1, eta_min=1e-6)
else:
    warmup = True

######### Scheduler ###########
if warmup:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion = torch.nn.L1Loss().cuda()

######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

train_dataset = get_training_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

val_dataset = get_validation_data(val_dir, img_options_train)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):    

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda().float()
        input_unpack = data[4].cuda().float()

        # if epoch>5:
        #     target, input_ = mixup.aug(target, input_)

        restored, restored_green = model_restoration(input_)
        restored = torch.clamp(restored,0,10)  
        
        loss = criterion(restored, target)
        loss = criterion(restored, target) + (30-epoch*27.0/100.0)*criterion(restored_green, torch.cat((target[:,1:2], target[:,3:4]), dim=1)) ##G
    
        loss.backward()
        # nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        epoch_loss +=loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            if save_images and epoch%5 == 0 and i==eval_now:
                utils.mkdir(result_dir + '/%d/%d'%(epoch,i))
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda().float()
                    filenames = data_val[2]
                    input_unpack = data_val[4].cuda().float()

                    restored, restored_green = model_restoration(input_) 
                    restored = torch.clamp(restored,0,10) 
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                    if save_images and epoch%5 == 0 and i==eval_now:
                        target = utils.processing(target.permute(0, 2, 3, 1).cpu().detach().numpy())
                        input_ = utils.processing(input_.permute(0, 2, 3, 1).cpu().detach().numpy())
                        restored = utils.processing(restored.permute(0, 2, 3, 1).cpu().detach().numpy())

                        noisy = np.concatenate((input_[:,:,:,:1], 0.5*(input_[:,:,:,1:2]+input_[:,:,:,3:]), input_[:,:,:,2:3]), axis=3)
                        if target.shape[3] == 4:
                            target = np.concatenate((target[:,:,:,:1], 0.5*(target[:,:,:,1:2]+target[:,:,:,3:]), target[:,:,:,2:3]), axis=3)
                            restored = np.concatenate((restored[:,:,:,:1], 0.5*(restored[:,:,:,1:2]+restored[:,:,:,3:]), restored[:,:,:,2:3]), axis=3)
                        else:
                            target = target[:,::2,::2,:]
                            restored = restored[:,::2,::2,:]
                        
                        for batch in range(input_.shape[0]):
                            # temp = np.concatenate((noisy[batch,:,:,:]*255, restored[batch,:,:,:]*255, target[batch,:,:,:]*255),axis=1)
                            temp = np.concatenate((restored[batch,:,:,:]*255, target[batch,:,:,:]*255),axis=1)
                            utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))
                            # cv2.imwrite(os.path.join(result_dir, str(epoch), str(i), 'input'+filenames[batch][:-4] +'.jpg'), input_[batch,0]*255)
                            # cv2.imwrite(os.path.join(result_dir, str(epoch), str(i), 'target'+filenames[batch][:-4] +'.jpg'), target[batch,0]*255)
                            # cv2.imwrite(os.path.join(result_dir, str(epoch), str(i), 'resored'+filenames[batch][:-4] +'.jpg'), restored[batch,0]*255)

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            
            model_restoration.train()

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    # it_total+=1

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%5 == 0 and epoch > 0:
        torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

