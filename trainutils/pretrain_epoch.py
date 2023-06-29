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

    total_iter = len(dataLoader)
    print()
    
    if opt.shuffle==1:
        for i, data in enumerate(tqdm(dataLoader, 0)):
            print('\r' + 'NormalEpoch '+ str(epoch)+' :' + 'iters: ' + str(i) + '/' + str(total_iter),end='')
            input_2D, input_3D, cam = data
            # input_2D, input_3D - tensor 
            iter_aug = 0
            if opt.poseaug==1 and random.random() < 0.5:
                iter_aug = 1
                poseaug_dict['model_G'] = change_poseaug_model(poseaug_dict['model_G'])
                input_2D_aug, input_3D_aug = inference_gan(input_3D.clone(), poseaug_dict, cam)
                [input_2D_aug, input_3D_aug] = get_varialbe(split,[input_2D_aug, input_3D_aug])
                # convert to relative
                input_3D_aug= input_3D_aug - input_3D_aug[:, :, :1, :]
                input_3D_aug [:, :, 0] = 0
                input_2D_aug = input_2D_aug.type(torch.cuda.FloatTensor)
                input_3D_aug = input_3D_aug.type(torch.cuda.FloatTensor)

            [input_2D, input_3D] = get_varialbe(split,[input_2D, input_3D])
            # convert to relative
            if opt.poseaug==1:    
                input_3D= input_3D - input_3D[:, :, :1, :]
            input_3D[:, :, 0] = 0  
            input_2D = input_2D.type(torch.cuda.FloatTensor)
            input_3D = input_3D.type(torch.cuda.FloatTensor)
            
            N = input_2D.size(0)
            spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D = MaskGen(opt)
            
            if iter_aug==1:
                output_2D = model_MAE(input_2D, spatial_mask2D, tmask2D, False)
                output_3D = model_MAE(input_3D, spatial_mask3D, tmask3D, False)        
                output_2D_aug, output_3DLift_aug = model_MAE(input_2D_aug, spatial_mask2D, tmask2D, True)
                output_3D_aug = model_MAE(input_3D_aug, spatial_mask2D, tmask2D, False)
                loss2d   = mpjpe_cal(output_2D, input_2D) + mpjpe_cal(output_2D_aug, input_2D_aug)
                loss3d   = mpjpe_cal(output_3D, input_3D) + mpjpe_cal(output_3D_aug, input_3D_aug)
                losslift = mpjpe_cal(output_3DLift_aug, input_3D_aug)
                loss = 5* loss2d + 5* loss3d +  losslift

            else:
                output_2D = model_MAE(input_2D, spatial_mask2D, tmask2D, False)
                output_3D = model_MAE(input_3D, spatial_mask3D, tmask3D, False)                  
                loss2d   = mpjpe_cal(output_2D, input_2D)
                loss3d   = mpjpe_cal(output_3D, input_3D)
                loss = 5* loss2d + 2* loss3d 
                
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif split == 'test':
                action_error_sum_MAE,   detail = test_calculation(output_2D, input_2D, action, action_error_sum_MAE, opt.dataset,0,MAE=opt.MAE)                                       
                action_error_sum_MAE3,  detail = test_calculation(output_3D, input_3D, action, action_error_sum_MAE3, opt.dataset,0,MAE=opt.MAE)
                if iter_aug==1:
                    action_error_sum_Lift,  detail = test_calculation(input_3D, output_3D_masklift, action, action_error_sum_Lift, opt.dataset,0,MAE=opt.MAE)
        print()
        
        if split == 'train':
            return loss_all['loss'].avg

        elif split == 'test':
            MAEp1, _ = print_error(opt.dataset, action_error_sum_MAE, opt.train)
            MAE3dp1, _ = print_error(opt.dataset, action_error_sum_MAE3, opt.train)
            MAEliftp1, _ = print_error(opt.dataset, action_error_sum_Lift, opt.train)
            return MAEp1, MAE3dp1, MAEliftp1, loss_all['loss'].avg
    
    elif opt.onlylift==1:
        pass
    
    elif opt.comp3d==1:
        for i, data in enumerate(tqdm(dataLoader, 0)):
            print('\r' + 'CompEpoch '+ str(epoch)+' :' + 'iters: ' + str(i) + '/' + str(total_iter),end='')
            input_2D, input_3D, cam = data
            # input_2D, input_3D - tensor 

            input_2D_new, input_3D_new = input_2D, input_3D

            [input_2D, input_3D] = get_varialbe(split,[input_2D_new, input_3D_new])
            
            # convert to relative
            if len(input_3D.shape) == 4:
                input_3D= input_3D - input_3D[:, :, :1, :]
                input_3D[:, :, 0] = 0   
            else:
                raise ValueError("Invalid inputs 3D shape"+str(input_3D.shape))
        
            N = input_2D.size(0)

            input_2D = input_2D.type(torch.cuda.FloatTensor)
            input_3D = input_3D.type(torch.cuda.FloatTensor)
            spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D = TubeMaskGen(opt, N)
            # match2D recover
            output_3D = model_MAE(input_3D, spatial_mask3D, tmask3D)
            loss3d = mpjpe_cal(output_3D, input_3D)
            comp = input_3D.permute(0,2,1,3).contiguous()
            comp[spatial_mask3D] = output_3D.permute(0,2,1,3)[spatial_mask3D]
            comp = comp.permute(0,2,1,3)
            loss = loss3d
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif split == 'test':
                action_error_sum_MAE3,  detail = test_calculation(comp, input_3D, action, action_error_sum_MAE3, opt.dataset,0,MAE=opt.MAE)
        print()
        
        if split == 'train':
            return loss_all['loss'].avg

        elif split == 'test':
            MAE3dp1, _ = print_error(opt.dataset, action_error_sum_MAE3, opt.train)
            return 0, MAE3dp1, 0, loss_all['loss'].avg
        pass

    elif opt.comp2dlift==1:
        for i, data in enumerate(tqdm(dataLoader, 0)):
            print('\r' + 'CompEpoch '+ str(epoch)+' :' + 'iters: ' + str(i) + '/' + str(total_iter),end='')
            input_2D, input_3D, cam = data
            # input_2D, input_3D - tensor 
            input_2D_new, input_3D_new = input_2D, input_3D
            [input_2D, input_3D] = get_varialbe(split,[input_2D_new, input_3D_new])
            
            # convert to relative
            if len(input_3D.shape) == 4:
                input_3D= input_3D - input_3D[:, :, :1, :]
                input_3D[:, :, 0] = 0   
            else:
                raise ValueError("Invalid inputs 3D shape"+str(input_3D.shape))
        
            N = input_2D.size(0)

            input_2D = input_2D.type(torch.cuda.FloatTensor)
            input_3D = input_3D.type(torch.cuda.FloatTensor)
            spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D = TubeMaskGen(opt, N)
            
            # match2D recover
            output_2D, output_3D_masklift = model_MAE(input_2D, spatial_mask2D, tmask2D)
            loss2d = mpjpe_cal(input_2D, output_2D)
            losslift = mpjpe_cal(output_3D_masklift, input_3D)
            loss = 2* losslift + loss2d
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif split == 'test':
                # matched 2D recover & lift error
                action_error_sum_MAE,   detail = test_calculation(output_2D, input_2D, action, action_error_sum_MAE, opt.dataset,0,MAE=opt.MAE)
                action_error_sum_Lift,  detail = test_calculation(input_3D, output_3D_masklift, action, action_error_sum_Lift, opt.dataset,0,MAE=opt.MAE)
        print()
        
        if split == 'train':
            return loss_all['loss'].avg

        elif split == 'test':
            MAEp1, _ = print_error(opt.dataset, action_error_sum_MAE, opt.train)
            MAEliftp1, _ = print_error(opt.dataset, action_error_sum_Lift, opt.train)
            return MAEp1, 0, MAEliftp1, loss_all['loss'].avg
    
def MaskGen(opt):
    f = opt.frames
    random_num = opt.spatial_mask_num
    
    # Gen 2D mask
    spatial_mask2D = np.zeros((f, opt.n_joints), dtype=bool)  #spatial mask  
    for k in range(f):
        rand_idx = np.random.choice(range(0,opt.n_joints), random_num, replace=False) 
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
    spatial_mask3D = np.zeros((f, opt.n_joints), dtype=bool)  #spatial mask
    for k in range(f):
        rand_idx = np.random.choice(range(0,opt.n_joints), random_num, replace=False) 
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

def TubeMaskGen(opt, b):
    f = opt.frames
    random_num = opt.spatial_mask_num
    mask_manner = np.array([
        [1,2,3], # left leg
        [13,14,15], # left arm
        [10,11,12], # right arm
        [4,5,6]  # right leg
    ],dtype=np.int32)
    mask_sample = [0,1,2,3]
    
    mask_manner2d = np.array([
        [1,2,3,4,5,6], # two leg
        [10,11,12,13,14,15], # two arm
        [1,2,3,13,14,15], # left leg + left arm
        [4,5,6,10,11,12]  # right leg + right arm
    ],dtype=np.int32)
    mask_sample2d = [0,1,2,3]
    
    
    # Gen 2D mask
    spatial_mask2D = np.zeros((b, opt.n_joints), dtype=bool)  #spatial mask  
    for k in range(b):
        d = np.random.choice(mask_sample2d, 1)[0]
        if(d<4):
            spatial_mask2D[k,mask_manner2d[d]] = True
        else:
            num = d-3
            rand_idx = np.random.choice(range(0,16), num, replace=False) 
            spatial_mask2D[k,rand_idx] = True
            

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
    spatial_mask3D = np.zeros((b, opt.n_joints), dtype=bool)  #spatial mask
    for k in range(b):
        d = np.random.choice(mask_sample, 1)[0]
        spatial_mask3D[k,mask_manner[d]] = True

    mask_num = int(f*opt.temporal_mask_rate)
    mask3D = np.hstack([
        np.zeros(f - mask_num),
        np.ones(mask_num),
    ]).flatten()
    np.random.shuffle(mask3D)
    tmask3D = mask3D

    spatial_mask3D = torch.from_numpy(spatial_mask3D).to(torch.bool).cuda()
    spatial_mask2D = torch.from_numpy(spatial_mask2D).to(torch.bool).cuda()
    return spatial_mask2D, spatial_mask3D, mask2D, mask3D, tmask2D, tmask3D