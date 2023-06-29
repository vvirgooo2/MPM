import torch
from common.utils import *
from tqdm import tqdm
import random 

def train_epoch(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val_epoch(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']

    if split == 'train':
        model_trans.train()
    else:
        model_trans.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()
    pck_sum = AccumLoss()
    auc_sum = AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    action_error_sum_MAE = define_error_list(actions)

    joints_left =  opt.joints_left
    joints_right = opt.joints_right

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0)):
        if split == "train":
            batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                       [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)
        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels)
        out_target[:, :, 0] = 0
        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.test_augmentation and split =='test':
            input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, model_trans, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
            output_3D, output_3D_VTE = model_trans(input_2D)

        output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels)
        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels)

        output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1),opt.n_joints, opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.n_joints, opt.out_channels)
        output_3D_single = output_3D

        if split == 'train':
            pred_out = output_3D_VTE

        elif split == 'test':
            pred_out = output_3D_single
            
        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)
        loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(output_3D_single, out_target_single)

        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_out[:,:,0,:] = 0
            joint_error = mpjpe_cal(pred_out, out_target).item()
            error_sum.update(joint_error*N, N)

        elif split == 'test':
           
            pred_out[:, :, 0, :] = 0
            action_error_sum = test_calculation(pred_out, out_target, ["one"], action_error_sum, opt.dataset, 'S1')
            joint_error_test = mpjpe_cal(pred_out, out_target).item()
            out = pred_out
            pck_sum.update(compute_PCK(pred_out, out_target),n=1)
            auc_sum.update(compute_AUC(pred_out, out_target),n=1)

            error_sum_test.update(joint_error_test * N, N)

    if split == 'train':
        if opt.MAE:
            return loss_all['loss'].avg*1000
        else:
            return loss_all['loss'].avg, error_sum.avg*1000
    
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, pck_sum.avg,auc_sum.avg

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

def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    output_3D_flip_VTE[:, :, :, joints_left + joints_right] = output_3D_flip_VTE[:, :, :, joints_right + joints_left]
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE
