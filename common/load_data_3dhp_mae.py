
import sqlite3
import torch.utils.data as data
import numpy as np
import sys
from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_3dhp import ChunkedGenerator

class Fusion(data.Dataset):
    def __init__(self, opt, root_path, train=True, MAE=False):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.opt = opt
        
        self.only3d = False
        self.only2d = False
        self.rateof2d = 1
        self.rateof3d = 1

        self.MAE=MAE
        if self.train:
            self.poses_train, self.poses_train_2d = self.prepare_data(opt.root_path, train=True)
            # self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
            #                                                                        subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_train,
                                              self.poses_train_2d, None, chunk_length=self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, MAE=MAE, train = True)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(opt.root_path, train=False)
            # self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
            #                                                                     subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_test,
                                              self.poses_test_2d, self.valid_frame,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, MAE=MAE, train = False)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        valid_frame={}

        self.kps_left, self.kps_right = [4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]
        self.joints_left, self.joints_right = [4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]

        if train == True:
            data = np.load(path+"data_train_3dhp.npz",allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']

                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    out_poses_2d[(subject_name, seq_name, cam)]=data_2d
                
            assert self.opt.n_joints==16,"n_joints must be 16 in mpi_inf_3dhp"  
            if self.opt.n_joints==16:
                trans = [14,8,9,10,11,12,13,15,16,0,5,6,7,2,3,4]
            # only use in 16 keypoints, in 17 keypoints mode, better use h36m+poseaug
            # else:
            #     trans = 
            
                    
            for seq in out_poses_3d.keys():
                out_poses_3d[seq][:,0] = (out_poses_3d[seq][:,16]+out_poses_3d[seq][:,0])/2
                out_poses_3d[seq]=out_poses_3d[seq][:,trans]
                out_poses_3d[seq] = out_poses_3d[seq] / 1000
                    
            for seq in out_poses_2d.keys():
                out_poses_2d[seq][:,0]=(out_poses_2d[seq][:,16]+out_poses_2d[seq][:,0])/2
                out_poses_2d[seq]=out_poses_2d[seq][:,trans]
                out_poses_2d[seq] = out_poses_2d[seq] / 1000


            return out_poses_3d, out_poses_2d
        
        else:
            data = np.load(path + "data_test_3dhp.npz", allow_pickle=True)['data'].item()
            for seq in data.keys():

                anim = data[seq]

                valid_frame[seq] = anim["valid"]

                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                out_poses_2d[seq] = data_2d

            assert self.opt.n_joints==16,"n_joints must be 16 in mpi_inf_3dhp"
            trans = [14,8,9,10,11,12,13,15,16,0,5,6,7,2,3,4]
                    
            for seq in out_poses_3d.keys():
                out_poses_3d[seq][:,0] = (out_poses_3d[seq][:,16]+out_poses_3d[seq][:,0])/2
                out_poses_3d[seq]=out_poses_3d[seq][:,trans]
                out_poses_3d[seq] = out_poses_3d[seq] / 1000
                    
            for seq in out_poses_2d.keys():
                out_poses_2d[seq][:,0]=(out_poses_2d[seq][:,16]+out_poses_2d[seq][:,0])/2
                out_poses_2d[seq]=out_poses_2d[seq][:,trans]
                out_poses_2d[seq] = out_poses_2d[seq] / 1000
            
            return out_poses_3d, out_poses_2d, valid_frame

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): 
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        if self.opt.debug==1:
            return 1024
        if self.opt.MAE:
            return 2*len(self.generator.pairs)
        else:
            return len(self.generator.pairs)

    def __getitem__(self, index):

        assert self.opt.shuffle == 1, "3dhp don't support completion task"
        assert self.opt.n_joints == 16, "3dhp only support 16 joints"
        
        if self.MAE:
            flag3d = -1
            if self.MAE and not self.only2d and not self.only3d:
                flag3d = index%2
                index = (index - index%2) // 2
            
            seq_name, start, end, flip, reverse = self.generator.pairs[index]
            cam, input_2D, input_3D, action, subject, cam_ind = self.generator.get_batch(seq_name, start, end, flip,reverse)
            
            # 预训练数据
            if flag3d==0:        
                return input_2D, np.zeros((0,16,3)), np.zeros(0,)
            
            if flag3d==1:                        
                return np.zeros((0,16,2)), input_3D, np.zeros(0,)
        
        else:
            seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
            cam, gt_3D, input_2D, seq, subject, cam_ind = self.generator.get_batch2(seq_name, start_3d, end_3d, flip, reverse)

            if self.train == False and self.test_aug:
                _, _, input_2D_aug, _, _,_ = self.generator.get_batch2(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
            bb_box = np.array([0, 0, 1, 1])
            input_2D_update = input_2D

            scale = np.float64(1.0)

            if self.train == True:
                return cam, gt_3D, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, gt_3D, input_2D_update, seq, scale, bb_box
            
            
            
    




