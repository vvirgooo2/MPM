import torch.utils.data as data
import numpy as np
from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_tds import ChunkedGenerator

class ChunkedGenerator:
    def __init__(self, opt, data, tds=1, MAE=True):
        self.opt = opt
        self.data = data['poses'].item()
        self.frames = opt.frames
        self.pairs = self.prepare_pairs(self.data)
        print('amass length: ', len(self.pairs))
        
    def prepare_pairs(self,data):
        pairlist = []
        for seq in data.keys():
            length = data[seq]["length"]
            keys = np.tile(np.array(seq),(length - 1,1))
            lower = np.arange(0, length - (self.frames)-1, 1)
            upper = np.arange(self.frames, length-1, 1)
            pairlist += list(zip(keys, lower, upper))
        return pairlist
    
    def get_sequence(self, seq, start, end):
        seq = seq[0]
        data = self.data
        seq2d = data[seq]['pose2d'][start:end, :, :]
        seq3d = data[seq]['pose3d'][start:end, : ,:]
        return seq2d, seq3d
    
            
class Fusion(data.Dataset):
    def __init__(self, opt, root_path='/mnt/sdb/Zhenyu/PSTMO/dataset/amass.npz', train=True, MAE=True, tds=1):
        self.opt =  opt
        self.data_type = 'h36m' 
        self.train = train
        self.root_path = root_path
        self.MAE = MAE
        self.tds = tds
        self.pad = opt.pad
        self.frames = opt.frames
        assert(train == True and MAE == True), "AMASS only support pretraining"
        self.data = np.load(root_path,allow_pickle=True)
        self.generator = ChunkedGenerator(opt, self.data, tds, self.MAE)
        
        self.ignore2d = True
        self.ignore3d = False
        self.rateof2d = 1
        self.rateof3d = 1
        
    def __len__(self):
        return 2*len(self.generator.pairs)
    
    
    def __getitem__(self, index):
        if self.MAE:
            flag3d = index%2
            index = (index - index%2) // 2

        seq , start, end = self.generator.pairs[index]
        matched_2D, matched_3D = self.generator.get_sequence(seq, start, end)

        return  matched_2D, np.zeros((0,16,3))
        
        # if flag3d==1:
        #     return  np.zeros((0,16,2)), matched_3D