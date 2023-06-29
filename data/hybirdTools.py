
import torch.utils.data as data
import torch
import numpy as np
from common.load_data_hm36_tds import Fusion as Fusion_hm36
from common.h36m_dataset import Human36mDataset
from common.load_data_3dhp_mae import Fusion as Fusion_mpi
from common.load_data_amass import Fusion as Fusion_AMASS

class HybirdSet(data.Dataset):
    def __init__(self,opt,list,train,hm36dataset=None):
        root_path = opt.root_path
        self.myset = []
        self.opt = opt

        if '3dhp' in list:
            if train:
                train_data = Fusion_mpi(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
                self.myset.append(train_data)
            else:
                test_data = Fusion_mpi(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
                self.myset.append(test_data)
            print('mpi_inf_3dhp Loaded')
            
        if 'h36m' in list:
            dataset = hm36dataset
            if train:
                train_data = Fusion_hm36(opt=opt, train=True, dataset=dataset, root_path=root_path, MAE=opt.MAE, tds=opt.t_downsample)
                self.myset.append(train_data)
            else:
                test_data = Fusion_hm36(opt=opt, train=False,dataset=dataset, root_path =root_path, MAE=opt.MAE, tds=opt.t_downsample)
                self.myset.append(test_data)
            print('Human3.6M Loaded')
        
                
        if 'amass' in list:
            if train:
                train_data = Fusion_AMASS(opt=opt, train=True)
                self.myset.append(train_data)
        

        
        self.lenlist = [len(item) for item in self.myset]
        self.length = sum(self.lenlist)
        return 

    def __len__(self):
        if self.opt.debug==1:
            return 512
        return self.length
    
    def __getitem__(self, index):
        temp = 0
        for i in range(0,len(self.lenlist)):
            if(index<temp+self.lenlist[i]):
                return self.myset[i][index-temp]
            temp += self.lenlist[i]

        raise IndexError("HybirdDataset Index out of range")

