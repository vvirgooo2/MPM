import torch
from common.utils import *
from Augpart.aug_infernce import inference_gan
from Augpart.gan_preparation import change_poseaug_model
import random
from tqdm import tqdm
def pretrain_epoch(opt, actions, train_loader, model, optimizer, epoch, poseaug_dict):
    return step('train', opt, actions, train_loader, model, optimizer, epoch, poseaug_dict)

def pretrain_val_epoch(opt, actions, val_loader, model, poseaug_dict):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model, poseaug_dict)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, poseaug_dict=None):
    model_MAE = model['MAE']

    if split == 'train':
        model_MAE.train()
    else:
        model_MAE.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    action = '*'
    
    action_error_sum_MAE = define_error_list(actions)
    action_error_sum_MAE3 = define_error_list(actions)
    action_error_sum_Lift = define_error_list(actions)

    # error_joints_2D = {'data':np.zeros((16)),'count':0}
    # error_joints_3D = {'data':np.zeros((16)),'count':0}

    total_iter = len(dataLoader)
    print()
    for i, data in enumerate(tqdm(dataLoader, 0)):
        print('\r' + 'Epoch '+ str(epoch)+' :' + 'iters: ' + str(i) + '/' + str(total_iter),end='')
        input_2D, input_3D, cam = data
        # input_2D, input_3D - tensor 
        if random.random() < 0.5 and split=='train':
            poseaug_dict['model_G'] = change_poseaug_model(poseaug_dict['model_G'])
            input_2D_new, input_3D_new = inference_gan(input_3D, poseaug_dict, cam)
            # input_2D_new  = torch.concat([input_2D, input_2D_new], axis = 0)
        else:
            input_2D_new, input_3D_new = input_2D, input_3D

        [input_2D, input_3D] = get_varialbe(split,[input_2D_new, input_3D_new])
        
        # convert to relative
        if len(input_3D.shape) == 4:
            input_3D= input_3D - input_3D[:, :, :1, :]
            input_3D[:, :, 0] = 0   
        else:
            raise ValueError("Invalid inputs 3D shape"+str(input_3D.shape))
    
        N = input_2D.size(0) + input_3D.size(0)

        input_2D = input_2D.type(torch.cuda.FloatTensor)
        input_3D = input_3D.type(torch.cuda.FloatTensor)
        
        spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D = MaskGen(opt)
        
        # match2D recover
        output_2D, output_3D_masklift = model_MAE(input_2D, spatial_mask2D, tmask2D)
        output_3D = model_MAE(input_3D, spatial_mask3D, tmask3D)
        
        loss2d   = mpjpe_cal(output_2D, input_2D)
        loss3d   = mpjpe_cal(output_3D, input_3D)
        losslift = mpjpe_cal(output_3D_masklift, input_3D)
        if opt.third_path==1:
            print('third')
            loss = loss2d + loss3d + losslift
        else:
            loss = loss2d + loss3d 
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
        
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            # matched 2D recover & lift error
            action_error_sum_MAE,   detail = test_calculation(output_2D, input_2D, action, action_error_sum_MAE, opt.dataset,0,MAE=opt.MAE)
            action_error_sum_Lift,  detail = test_calculation(output_3D, output_3D_masklift, action, action_error_sum_Lift, opt.dataset,0,MAE=opt.MAE)
            # matched 3D recover & delift error                                          
            action_error_sum_MAE3,  detail = test_calculation(output_3D, input_3D, action, action_error_sum_MAE3, opt.dataset,0,MAE=opt.MAE)

    print()
    
    if split == 'train':
        return loss_all['loss'].avg

    elif split == 'test':
        MAEp1, _ = print_error(opt.dataset, action_error_sum_MAE, opt.train)
        MAE3dp1, _ = print_error(opt.dataset, action_error_sum_MAE3, opt.train)
        MAEliftp1, _ = print_error(opt.dataset, action_error_sum_Lift, opt.train)
        return MAEp1, MAE3dp1, MAEliftp1, loss_all['loss'].avg

    
def MaskGen(opt):
    f = opt.frames
    random_num = opt.spatial_mask_num
    
    # Gen 2D mask
    spatial_mask2D = np.zeros((f, 16), dtype=bool)  #spatial mask  
    for k in range(f):
        rand_idx = np.random.choice(range(0,16), random_num, replace=False) 
        spatial_mask2D[k, rand_idx] = True

    mask_num = int(f*opt.temporal_mask_rate)
    mask2D = np.hstack([
        np.zeros(f - mask_num),
        np.ones(mask_num),
    ]).flatten()

    np.random.seed()
    np.random.shuffle(mask2D)
    tmask2D = mask2D
    mask2D = torch.from_numpy(mask2D).to(torch.bool).cuda()

    #Gen 3D mask
    spatial_mask3D = np.zeros((f, 16), dtype=bool)  #spatial mask
    for k in range(f):
        rand_idx = np.random.choice(range(0,16), random_num, replace=False) 
        spatial_mask3D[k, rand_idx] = True

    mask_num = int(f*opt.temporal_mask_rate)
    mask3D = np.hstack([
        np.zeros(f - mask_num),
        np.ones(mask_num),
    ]).flatten()
    np.random.shuffle(mask3D)
    tmask3D = mask3D
    mask3D = torch.from_numpy(mask3D).to(torch.bool).cuda()
    
    return spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D