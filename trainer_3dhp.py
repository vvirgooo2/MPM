import os
import glob
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
from common.load_data_3dhp_mae import Fusion
from trainutils.finetune_3dhp_epoch import train_epoch, val_epoch
opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


if __name__ =='__main__':
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
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + '3dhp' + '.npz'

    actions = define_actions(opt.actions,'any')

    if opt.train:
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    model = {}
    model['trans'] = nn.DataParallel(Model(opt)).cuda()

    model_params = 0
    for parameter in model['trans'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)


    if opt.MAEreload == 1:
        model_dict = model['trans'].state_dict()

        MAE_path = opt.previous_dir

        pre_dict = torch.load(MAE_path)

        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        model['trans'].load_state_dict(model_dict)


    model_dict = model['trans'].state_dict()
    if opt.reload == 1:

        no_refine_path = opt.previous_dir

        pre_dict = torch.load(no_refine_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)


    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            loss, mpjpe = train_epoch(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
            p1, pck ,auc = val_epoch(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f, pck: %.2f auc: %.2f' % (p1, pck, auc),'%')
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f, pck: %.2f, auc: %.2f' % (epoch, lr, loss, mpjpe, p1, pck, auc))
                print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f, pck: %.2f, auc: %.2f' % (epoch, lr, loss, mpjpe, p1, pck, auc))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay
