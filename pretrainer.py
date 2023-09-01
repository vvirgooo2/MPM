import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from model.MPMLift import MPMmask, MPMmask2
from Augpart.aug_infernce import inference_gan
from Augpart.gan_preparation import get_poseaug_model, change_poseaug_model
from data.dataloader_getter import get_loader
from trainutils.pretrain_epoch import pretrain_epoch, pretrain_val_epoch

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    
if __name__ == '__main__':
    # pre-work
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            

    actions = define_actions(opt.actions, 'any')
    print(torch.cuda.is_available())
    model = {}
    if opt.shuffle==0:
        model['MAE'] =   nn.DataParallel(MPMmask(opt)).cuda()
    else:
        model['MAE'] =   nn.DataParallel(MPMmask2(opt)).cuda()
    
    print('poseaug', opt.poseaug)
    print('n_joints', opt.n_joints)
    print('shuffle', opt.shuffle)
    print('testaug', opt.test_augmentation)
    # define model and load weight
    model_params = 0
    for parameter in model['MAE'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if opt.MAEreload == 1:
        model_dict = model['MAE'].state_dict()
        MAE_path = opt.MAEcp
        pre_dict = torch.load(MAE_path)
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        print("Load layers: ", len(state_dict.keys()))
        prop = "Load layers: " + str(len(state_dict.keys()))
        
        model['MAE'].load_state_dict(model_dict)

    # define optimizer
    all_param = []
    lr = opt.lr
    all_param += list(model['MAE'].named_parameters())
    
    # -frozen
    frozen_count = 0
    frozen_List = []
    if opt.frozen==1:
        for param in all_param:
            if param[0].split('.')[1] in frozen_List:
                param[1].requires_grad = False
                frozen_count = frozen_count + 1
                
    all_param = [k[1] for k in all_param]    
    print('frozen layers:'+ str(frozen_count))
    print('unfrozen layers:'+ str(len(all_param)-frozen_count))
    
    optimizer_all = optim.Adam(filter(lambda p: p.requires_grad, all_param), lr=opt.lr, amsgrad=True)

    # load data
    dataloader_dict = get_loader(opt)
    poseaug_dict = get_poseaug_model(opt)   
    
    
    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            loss = pretrain_epoch(opt, actions, dataloader_dict['train'], model, optimizer_all, epoch, poseaug_dict)
                
        if opt.test == 1:
            maep1, mae3p1, liftp1, loss_test = pretrain_val_epoch(opt, actions, dataloader_dict['test'], model, poseaug_dict)
            data_threshold = maep1

            if opt.train:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
                                                   model['MAE'], 'pretrain',opt)
                opt.previous_best_threshold = data_threshold
                    
            if opt.train == 0:
                print('maep1: %.2f, mae3p1: %.2f, liftp1: %.2f' % (maep1, mae3p1, liftp1))
                break
            
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, loss_test: %.4f, maep1: %.2f, mae3dp1: %.2f, liftp1: %.2f' % (
                epoch, lr, loss, loss_test, maep1, mae3p1, liftp1))
                print('e: %d, lr: %.7f, loss: %.4f, loss_test: %.4f, maep1: %.2f, mae3dp1: %.2f, liftp1: %.2f' % (epoch, lr, loss, loss_test, maep1,  mae3p1, liftp1))
            

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask)

    output_2D_flip[:, 0] *= -1

    output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

    output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask)

    output_2D = (output_2D_non_flip + output_2D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_2D






