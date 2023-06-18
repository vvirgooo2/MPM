import torch
import torch.nn as nn

from Augpart.Generator import PoseGenerator
from Augpart.gan_utils import get_scheduler
import random
from collections import OrderedDict
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        
        
def get_poseaug_model(args):
    """
    return PoseAug augmentor and discriminator
    and corresponding optimizer and scheduler
    """
    # Create model: G and D
    print("==> Creating model...")
    device = torch.device("cuda")
    num_joints = 16

    # generator for PoseAug
    model_G = nn.DataParallel(PoseGenerator(args, num_joints * 3)).cuda()
    model_G.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G.parameters()) / 1000000.0))
    randchk = random.choice([
        '/home/syh/Zhangzy/MPMLP/AugPSTMO/Augpart/chk/ckpt_generater11.pth.tar',
        '/home/syh/Zhangzy/MPMLP/AugPSTMO/Augpart/chk/ckpt_generater21.pth.tar',
        '/home/syh/Zhangzy/MPMLP/AugPSTMO/Augpart/chk/ckpt_generater31.pth.tar',
        '/home/syh/Zhangzy/MPMLP/AugPSTMO/Augpart/chk/ckpt_generater37.pth.tar'
    ])
    # # discriminator for 3D
    # model_d3d = Pos3dDiscriminator(num_joints).to(device)
    # model_d3d.apply(init_weights)
    # print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d.parameters()) / 1000000.0))

    # # discriminator for 2D
    # model_d2d = Pos2dDiscriminator(num_joints).to(device)
    # model_d2d.apply(init_weights)
    # print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d.parameters()) / 1000000.0))

    # offline using

    # # prepare optimizer
    # g_optimizer = torch.optim.Adam(model_G.parameters(), lr=args.lr_g)
    # d3d_optimizer = torch.optim.Adam(model_d3d.parameters(), lr=args.lr_d)
    # d2d_optimizer = torch.optim.Adam(model_d2d.parameters(), lr=args.lr_d)

    # # prepare scheduler
    # g_lr_scheduler = get_scheduler(g_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    # d3d_lr_scheduler = get_scheduler(d3d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    # d2d_lr_scheduler = get_scheduler(d2d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)

    return {
        'model_G': model_G,
        # 'model_d3d': model_d3d,
        # 'model_d2d': model_d2d,
        # 'optimizer_G': g_optimizer,
        # 'optimizer_d3d': d3d_optimizer,
        # 'optimizer_d2d': d2d_optimizer,
        # 'scheduler_G': g_lr_scheduler,
        # 'scheduler_d3d': d3d_lr_scheduler,
        # 'scheduler_d2d': d2d_lr_scheduler,
    }

def change_poseaug_model(model_G):
    randchk = random.choice([
        'Augpart/chk/ckpt_generater21.pth.tar',
        'Augpart/chk/ckpt_generater31.pth.tar',
        'Augpart/chk/ckpt_generater37.pth.tar'
    ])
    

    state_dict = torch.load(randchk)['model_G']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.'+k # module字段在最前面，从第7个字符开始就可以去掉module
        new_state_dict[name] = v #新字典的key值对应的value一一对应

    model_G.load_state_dict(new_state_dict)
    # print('load ' + randchk)
    return model_G
