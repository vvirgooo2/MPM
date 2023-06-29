import torch
import torch.nn as nn
import numpy as np
from model.component.Encoder import FCBlock, Simplemlp
from model.component.Transformer_mask import Transformer as Transformer
from model.component.Transformer_mask import Transformer_dec
from model.component.strided_transformer_encoder import Transformer as Transformer_reduce

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MPMmask(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        
        # args
        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        self.tube = args.onlylift + args.comp3d + args.comp2dlift
        self.onlylift = args.onlylift
        self.comp2dlift = args.comp2dlift
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.n_joints
        self.length = length
        
        # encoder light mlp
        self.encoder2d =  Simplemlp(channel_in=2*self.num_joints_in, 
                                    linear_size=channel*2)
        
        self.encoder3d =  Simplemlp(channel_in=3*self.num_joints_in, 
                                    linear_size=channel*2)
        
        # shared layer
        self.shared_layers_spatial = FCBlock(channel_out=channel, linear_size=2* channel,
                                        block_num=1)
        
        self.shared_layers_temporal = Transformer(layers, channel, d_hid, length=length)
        
        channel_dec = channel
        self.encoder_LN = LayerNorm(channel)
        # self.encoder_to_decoder = nn.Linear(channel, channel_dec, bias=False)
        # token and embedding
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, length, channel_dec))
        self.temporal_mask_token = nn.Parameter(torch.randn(1, 1, channel_dec))
        self.spatial_mask_token3d = nn.Parameter(torch.randn(1, 1, 3))
        self.spatial_mask_token2d = nn.Parameter(torch.randn(1, 1, 2))
        
        self.decoder2d = Transformer_dec(2,channel_dec,d_hid,length=length)
        self.decoder3d = Transformer_dec(2,channel_dec,d_hid,length=length)
        
        self.fcn_dec2d = nn.Sequential(
            nn.BatchNorm1d(channel_dec, momentum=0.1),
            nn.Conv1d(channel_dec, 2*self.num_joints_out, kernel_size=1)
        )        
        
        self.fcn_dec3d = nn.Sequential(
            nn.BatchNorm1d(channel_dec, momentum=0.1),
            nn.Conv1d(channel_dec, 3*self.num_joints_out, kernel_size=1)
        )
        
    def forward(self,x,smask=None,tmask=None):
        # input size
        # 2d (batchsize, frame, joints_num, 2) 
        # 3d (batchsize, frame, joints_num, 3)
        if(x.shape[0]==0):
            return x
            
        if tmask.all()!=None:
            tmask= torch.from_numpy(tmask).to(torch.bool).cuda()
        smask = smask.detach().cpu().numpy()
        
        b, f, n,d = x.shape 
        x = x.clone()
        
        if self.tube > 0:
            n_mask = np.sum(smask==True)
            # tube mask
            if d==2:
                x = x.permute(1,0,2,3).contiguous()
                x[:,smask] = self.spatial_mask_token2d.expand(f,n_mask,2)
                x = x.permute(1,0,2,3).contiguous()
                
            elif d==3:
                x = x.permute(1,0,2,3).contiguous()
                x[:,smask] = self.spatial_mask_token3d.expand(f,n_mask,3)
                x = x.permute(1,0,2,3).contiguous()
        else:
            # joint mask
            if d==2 :
                x[:,smask] = self.spatial_mask_token2d.expand(b,self.spatial_mask_num*f,2)
            elif d==3 :
                x[:,smask] = self.spatial_mask_token3d.expand(b,self.spatial_mask_num*f,3)

        x = x.view(b, f, -1).permute(0,2,1).contiguous()

        if d==2:
            x = self.encoder2d(x)
        else:
            x = self.encoder3d(x)
            
        x = self.shared_layers_spatial(x)
        
        x = x.permute(0,2,1).contiguous()
        feas = self.shared_layers_temporal(x,mask_MAE=tmask)
        feas = self.encoder_LN(feas)
        # feas = self.encoder_to_decoder(feas)
        
        
        B,N,C = feas.shape
        expand_pos_embed = self.dec_pos_embedding.expand(B, -1, -1).clone()
        pos_emd_vis = expand_pos_embed[:, ~tmask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[:, tmask].reshape(B, -1, C)
        
        mask_idx = torch.nonzero(tmask).squeeze().cuda()
        unmask_idx = torch.nonzero(~tmask).squeeze().cuda()
        rearrange_idx = torch.concat((unmask_idx, mask_idx), dim=0)
        
        x_full = torch.cat([feas + pos_emd_vis, self.temporal_mask_token + pos_emd_mask], dim=1) 
        x_full[:,rearrange_idx] = x_full.clone()

        if d==2:
            x_out2 = self.decoder2d(x_full)
            x_out2 = x_out2.permute(0, 2, 1).contiguous()
            x_out2 = self.fcn_dec2d(x_out2)
            x_out2 = x_out2.permute(0, 2, 1).contiguous()
            x_out_2d = x_out2.view(b, f, self.num_joints_in, 2)

            if self.comp2dlift==1:
                x_out3 = self.decoder3d(x_full)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out3 = self.fcn_dec3d(x_out3)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out_3d = x_out3.view(b, f, self.num_joints_in, 3)  
                return x_out_2d, x_out_3d
        
            else:
                return x_out_2d, torch.zeros(b, f, n, 3).cuda()
        
        if d==3:
            x_out3 = self.decoder3d(x_full)
            x_out3 = x_out3.permute(0, 2, 1).contiguous()
            x_out3 = self.fcn_dec3d(x_out3)
            x_out3 = x_out3.permute(0, 2, 1).contiguous()
            x_out_3d = x_out3.view(b, f, self.num_joints_in, 3)
            return x_out_3d

class MPMmask2(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        
        # args
        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        self.tube = args.onlylift + args.comp3d + args.comp2dlift
        self.onlylift = args.onlylift
        self.comp2dlift = args.comp2dlift
        # self.third_path = args.third_path
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.n_joints
        self.length = length
        
        # encoder light mlp
        self.encoder2d =  Simplemlp(channel_in=2*self.num_joints_in, 
                                    linear_size=channel*2)
        
        self.encoder3d =  Simplemlp(channel_in=3*self.num_joints_in, 
                                    linear_size=channel*2)
        
        # shared layer
        self.shared_layers_spatial = FCBlock(channel_out=channel, linear_size=2* channel,
                                        block_num=1)
        
        self.shared_layers_temporal = Transformer(layers, channel, d_hid, length=length)
        
        channel_dec = channel
        self.encoder_LN = LayerNorm(channel)
        # self.encoder_to_decoder = nn.Linear(channel, channel_dec, bias=False)
        # token and embedding
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, length, channel_dec))
        self.temporal_mask_token = nn.Parameter(torch.randn(1, 1, channel_dec))
        self.spatial_mask_token3d = nn.Parameter(torch.randn(1, 1, 3))
        self.spatial_mask_token2d = nn.Parameter(torch.randn(1, 1, 2))
        
        self.decoder2d = Transformer_dec(1,channel_dec,d_hid,length=length)
        self.decoder3d = Transformer_dec(1,channel_dec,d_hid,length=length)
        
        self.fcn_dec2d = nn.Sequential(
            nn.BatchNorm1d(channel_dec, momentum=0.1),
            nn.Conv1d(channel_dec, 2*self.num_joints_out, kernel_size=1)
        )        
        
        self.fcn_dec3d = nn.Sequential(
            nn.BatchNorm1d(channel_dec, momentum=0.1),
            nn.Conv1d(channel_dec, 3*self.num_joints_out, kernel_size=1)
        )
        
    def forward(self,x, smask=None, tmask=None, liftflag=False):
        # input size
        # 2d (batchsize, frame, joints_num, 2) 
        # 3d (batchsize, frame, joints_num, 3)
        if(x.shape[0]==0):
            return x
            
        if tmask.all()!=None:
            tmask= torch.from_numpy(tmask).to(torch.bool).cuda()

        b, f, n,d = x.shape 
        x = x.clone()
        
        if self.tube > 0:
            n_mask = np.sum(smask==True)
            # tube mask
            if d==2:
                x = x.permute(1,0,2,3).contiguous()
                x[:,smask] = self.spatial_mask_token2d.expand(f,n_mask,2)
                x = x.permute(1,0,2,3).contiguous()
                
            elif d==3:
                x = x.permute(1,0,2,3).contiguous()
                x[:,smask] = self.spatial_mask_token3d.expand(f,n_mask,3)
                x = x.permute(1,0,2,3).contiguous()
        else:
            # joint mask
            if d==2 :
                x[:,smask] = self.spatial_mask_token2d.expand(b,self.spatial_mask_num*f,2)
            elif d==3 :
                x[:,smask] = self.spatial_mask_token3d.expand(b,self.spatial_mask_num*f,3)

        x = x.view(b, f, -1).permute(0,2,1).contiguous()

        if d==2:
            x = self.encoder2d(x)
        else:
            x = self.encoder3d(x)
            
        x = self.shared_layers_spatial(x)
        
        x = x.permute(0,2,1).contiguous()
        feas = self.shared_layers_temporal(x,mask_MAE=tmask)
        feas = self.encoder_LN(feas)
        # feas = self.encoder_to_decoder(feas)
        
        
        B,N,C = feas.shape
        expand_pos_embed = self.dec_pos_embedding.expand(B, -1, -1).clone()
        pos_emd_vis = expand_pos_embed[:, ~tmask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[:, tmask].reshape(B, -1, C)
        
        mask_idx = torch.nonzero(tmask).squeeze().cuda()
        unmask_idx = torch.nonzero(~tmask).squeeze().cuda()
        rearrange_idx = torch.concat((unmask_idx, mask_idx), dim=0)
        
        x_full = torch.cat([feas + pos_emd_vis, self.temporal_mask_token + pos_emd_mask], dim=1) 
        x_full[:,rearrange_idx] = x_full.clone()

        if d==2:
            x_out2 = self.decoder2d(x_full)
            x_out2 = x_out2.permute(0, 2, 1).contiguous()
            x_out2 = self.fcn_dec2d(x_out2)
            x_out2 = x_out2.permute(0, 2, 1).contiguous()
            x_out_2d = x_out2.view(b, f, self.num_joints_in, 2)

            if liftflag == True:
                x_out3 = self.decoder3d(x_full)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out3 = self.fcn_dec3d(x_out3)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out_3d = x_out3.view(b, f, self.num_joints_in, 3)  
                return x_out_2d, x_out_3d
        
            else:
                return x_out_2d
        
        if d==3:
            x_out3 = self.decoder3d(x_full)
            x_out3 = x_out3.permute(0, 2, 1).contiguous()
            x_out3 = self.fcn_dec3d(x_out3)
            x_out3 = x_out3.permute(0, 2, 1).contiguous()
            x_out_3d = x_out3.view(b, f, self.num_joints_in, 3)
            return x_out_3d