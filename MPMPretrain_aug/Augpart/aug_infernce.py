from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from common.utils import AccumLoss, set_grad


from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.camera import project_to_2d
from Augpart.gan_dataloader import PoseDataSet
from Augpart.aug_viz import plot_poseaug
from tqdm import tqdm
import numpy as np
import gc

def inference_gan(input_3D, poseaug_dict, cam):
    device = torch.device("cuda")
    # extract necessary module for training.
    model_G = poseaug_dict['model_G']
    model_G.eval()

    set_grad([model_G], False)

    inputs_3d, cam_param = input_3D.to(device), cam.to(device)

    # poseaug: BA BL RT
    g_rlt = model_G(inputs_3d)
        
    # extract the generator result
    outputs_3d_rt = g_rlt['pose_rt']
    outputs_2d_rt = project_to_2d(outputs_3d_rt, cam_param)  # fake 2d data

       
    # here add a check so that outputs_2d_rt that out of box will be remove.
    valid_rt_idx = torch.sum(outputs_2d_rt > 1, dim=(2, 3)) < 1
    valid_rt_idx = torch.all(valid_rt_idx, dim=1, keepdim=True)
    valid_rt_idx = valid_rt_idx.flatten()
    
    output3d = outputs_3d_rt.detach()[valid_rt_idx].cpu()
    output2d = outputs_2d_rt.detach()[valid_rt_idx].cpu()
    outputcam = cam_param.detach()[valid_rt_idx].cpu()
    
    
    
    return output2d, output3d