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
from model.MPM import Model
from model.component.refine import refine

from data.dataloader_getter import get_loader
from trainutils.finetune_epoch import train_epoch, train_val_epoch
from common.camera import get_uvd2xyz


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
            

    actions = define_actions(opt.actions,'h36m')
    
    print(torch.cuda.is_available())
    model = {}
    model['trans'] =   nn.DataParallel(Model(opt)).cuda()
    model['refine'] = nn.DataParallel(refine(opt)).cuda()


    if opt.MAEreload == 1:
        model_dict = model['trans'].state_dict()

        MAE_path = opt.previous_dir

        pre_dict = torch.load(MAE_path)

        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        logging.info('load+ ' + str(len(state_dict.keys())))
        print('load+ ' + str(len(state_dict.keys())))
        model_dict.update(state_dict)
        model['trans'].load_state_dict(model_dict)


    model_dict = model['trans'].state_dict()
    if opt.reload == 1:

        no_refine_path = opt.previous_dir

        pre_dict = torch.load(no_refine_path)
        # print(pre_dict.keys())
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload == 1:

        refine_path = opt.previous_refine_name

        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    # define optimizer
    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    # load data
    dataloader_dict = get_loader(opt)

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            loss, mpjpe = train_epoch(opt, actions, dataloader_dict['train'], model, optimizer_all, epoch)
                
        if opt.test == 1:
            p1, p2 = train_val_epoch(opt, actions, dataloader_dict['test'], model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')

                if opt.refine:
                    opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                            data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f, p2: %.2f' % (p1, p2))
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, mpjpe, p1, p2))
                print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, mpjpe, p1, p2))
        
        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay








