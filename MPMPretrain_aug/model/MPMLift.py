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
        self.third_path = args.third_path
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
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
        
        channel_dec = channel//2
        self.encoder_LN = LayerNorm(channel)
        self.encoder_to_decoder = nn.Linear(channel, channel_dec, bias=False)
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
        
    def forward(self,x,smask=None,tmask=None):
        # input size
        # 2d (batchsize, frame, joints_num, 2) 
        # 3d (batchsize, frame, joints_num, 3)
        if(x.shape[0]==0):
            return x
            
        if tmask.all()!=None:
            tmask= torch.from_numpy(tmask).to(torch.bool).cuda()
        # if tmask == None:
        #     tmask = np.hstack([np.zeros(243)])
        #     tmask= torch.from_numpy(tmask).to(torch.bool).cuda()
            
        b, f, n,d = x.shape 
        x = x.clone()
        
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
        feas = self.encoder_to_decoder(feas)
        
        
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
        
            if self.third_path ==1:
                x_out3 = self.decoder3d(x_full)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out3 = self.fcn_dec3d(x_out3)
                x_out3 = x_out3.permute(0, 2, 1).contiguous()
                x_out_3d = x_out3.view(b, f, self.num_joints_in, 3)  
                print('third')
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
   
    
class MPM(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        
        # args
        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        self.length = length
        
        # Encoder MLP 
        self.encoder2d =  FCBlock(channel_in=2*self.num_joints_in, 
                                    channel_out=channel, linear_size=2*channel,
                                    block_num=1)
        
        # Attension n layer transformer
        self.Att = Transformer(layers, channel, d_hid, length=length)
        
        channel_dec = channel//2
        self.encoder_LN = LayerNorm(channel)
        self.encoder_to_decoder = nn.Linear(channel, channel_dec, bias=False)
        
        # Decoder n-1 layer transformer
        self.decoder3d = Transformer_dec(layers-1, channel_dec, d_hid//2, length=length)
        
        # token and embedding
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, length, channel_dec))
        
        self.agg =  FCBlock(channel_in=length, 
                                    channel_out=1, linear_size=channel,
                                    block_num=1)
        
        self.fcn_dec3d = nn.Sequential(
            nn.BatchNorm1d(channel_dec, momentum=0.1),
            nn.Conv1d(channel_dec, 3*self.num_joints_out, kernel_size=1)
        )
        
        
        
    def forward(self,x):
        # input size
        # 2d (batchsize, frame, joints_num, 2) 
        # 3d (batchsize, frame, joints_num, 3)
        
        b,f,n,d = x.shape 
        x = x.clone()
        
        x = x.view(b, f, -1).permute(0,2,1).contiguous()
        
        x = self.encoder2d(x)

        x = x.permute(0,2,1).contiguous()

        x = self.Att(x)
        x = self.encoder_LN(x)
        x = self.encoder_to_decoder(x)

        b,f,c = x.shape
        expand_pos_embed = self.dec_pos_embedding.expand(b, f, -1).clone()
        x = x + expand_pos_embed.reshape(b,f,c)
        

        x = self.decoder3d(x)
        x = x.permute(0,2,1).contiguous() 
        x = self.fcn_dec3d(x)
        x = x.permute(0,2,1).contiguous() 
        x_vec = x
        x_vec = x_vec.permute(0,2,1).contiguous() 
        x_vec = x_vec.view(b, f,self.num_joints_in, 3)
        
        x = self.agg(x)
        x = x.permute(0,2,1).contiguous() 
        x = x.view(b, 1, self.num_joints_in, 3)
        # print(x.shape," ",x_vec.shape)
        return x,x_vec
    
class MPMAgg(torch.nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        
        # args
        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        self.length = length
        stride_num = args.stride_num
        # Encoder MLP 
        self.encoder2d =  FCBlock(channel_in=2*self.num_joints_in, 
                                    channel_out=channel, linear_size=2*channel,
                                    block_num=1)
        
        # Attension n layer transformer
        self.Att = Transformer(layers, channel, d_hid, length=length)
        
        self.Transformer_reduce = Transformer_reduce(len(stride_num), channel, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )
        
       
        
    def forward(self,x):
        # input size
        # 2d (batchsize, frame, joints_num, 2) 
        # 3d (batchsize, frame, joints_num, 3)
        
        b,f,n,d = x.shape 
        x = x.clone()
        
        x = x.view(b, f, -1).permute(0,2,1).contiguous()
        
        x = self.encoder2d(x)

        x = x.permute(0,2,1).contiguous()

        x = self.Att(x)

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 
        x_VTE = x_VTE.permute(0,2,1).contiguous()
        x_VTE = x_VTE.view(b, f, self.num_joints_in, 3)

        x = self.Transformer_reduce(x) 
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 
        x = x.permute(0, 2, 1).contiguous() 
        x = x.view(b, 1, self.num_joints_in, 3)

        # print(x.shape+" "+ x_VTE.shape)
        return x, x_VTE

