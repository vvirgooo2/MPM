
import torch
from common.load_data_hm36_tds import Fusion as Fusion_hm36
from common.load_data_3dhp_mae import Fusion as Fusion_mpi
from common.h36m_dataset import Human36mDataset
from data.hybirdTools import HybirdSet
from common.utils import define_actions
def get_loader(opt):
    root_path = opt.root_path
    if opt.MAE:
        dataset_path = root_path + 'data_3d_' + 'h36m' + '.npz'
        dataset = None
        if 'h36m' in opt.dataset:
            dataset = Human36mDataset(dataset_path,opt)
        if opt.train:
            train_data = HybirdSet(opt,opt.dataset,True,hm36dataset=dataset)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize*2,shuffle=True, num_workers=int(opt.workers), pin_memory=True,collate_fn=mycollate)
        if opt.test:
            test_data = HybirdSet(opt,opt.dataset,False,hm36dataset=dataset)
            test_dataloader = torch.utils.data.DataLoader(test_data , batch_size=opt.batchSize*2,shuffle=False, num_workers=int(opt.workers), pin_memory=True,collate_fn=mycollate)
    else:
        if 'h36m' in opt.dataset:
            dataset_path = root_path + 'data_3d_' + 'h36m' + '.npz'
            dataset = Human36mDataset(dataset_path,opt)
            actions = define_actions(opt.actions, 'h36m')
            if opt.train:
                train_data = Fusion_hm36(opt=opt, train=True, dataset=dataset, root_path=root_path, MAE=opt.MAE, tds=opt.t_downsample)
                train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                        shuffle=True, num_workers=int(opt.workers), pin_memory=True)
            if opt.test:
                test_data = Fusion_hm36(opt=opt, train=False,dataset=dataset, root_path =root_path, MAE=opt.MAE, tds=opt.t_downsample)
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                        shuffle=False, num_workers=int(opt.workers), pin_memory=True)

        elif '3dhp' in opt.dataset:
            root_path = opt.root_path
            dataset_path = root_path + 'data_3d_' + '3dhp' + '.npz'

            actions = define_actions(opt.actions,'3dhp')

            if opt.train:
                #train_data = Fusion(opt=opt, train=True, root_path=root_path)
                train_data = Fusion_mpi(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
                train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                            shuffle=True, num_workers=int(opt.workers), pin_memory=True)
            if opt.test:
                #test_data = Fusion(opt=opt, train=False,root_path =root_path)
                test_data = Fusion_mpi(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                            shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    dataloader_dict = {
        'train': train_dataloader,
        'test' : test_dataloader
    }  
    return dataloader_dict

def mycollate(batch):
        x = [ torch.as_tensor(batch[k][0]) for k in range(0,len(batch)) if batch[k][0].shape[0]!=0]
        y = [ torch.as_tensor(batch[k][1]) for k in range(0,len(batch)) if batch[k][1].shape[0]!=0]
        z = [ torch.as_tensor(batch[k][2]) for k in range(0,len(batch)) if batch[k][2].shape[0]!=0]

        x = torch.ones((0,0,16,3)) if len(x)==0 else torch.stack(x,0)
        y = torch.ones((0,0,16,3)) if len(y)==0 else torch.stack(y,0)
        z = torch.ones((0,9,)) if len(z)==0 else torch.stack(z,0)
        return(x, y, z)