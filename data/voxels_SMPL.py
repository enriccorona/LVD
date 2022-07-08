import os.path
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import os
import torch
import scipy.io

import os.path
from data.dataset import DatasetBase
import random
import numpy as np
import pickle
import os
from skimage.transform import resize
import torch
from utils import util
import torchvision
from PIL import Image
import kaolin
import glob
import trimesh
import cv2

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset'
        self.scans_dir = '/home/ecorona/eccv22/reconstruction/data/ipnet_data/generate_voxels/voxels/'
        self.scans_dir_augmented = '/home/ecorona/eccv22/reconstruction/data/ipnet_data/generate_voxels/voxels_augmented/'

        if self._mode == 'train':
            self.data = [self.scans_dir + '/axyz/']*106 + \
                    [self.scans_dir + '/th_good_1/']*145 + \
                    [self.scans_dir + '/th_good_3/']*170 + \
                    [self.scans_dir + '/treddy/']*42

            self.scan_indices = np.concatenate((
                np.arange(106), np.arange(145), np.arange(170), np.arange(42)
                ))

            #inds_removing = [73, 56, 84, 126 + 106] # Bad fits
            #self.data = np.delete(self.data, inds_removing)
            #self.scan_indices = np.delete(self.scan_indices, inds_removing)

        else:
            # Renderpeople entirely for validation:
            self.data = [self.scans_dir + '/renderpeople/']*304
            self.scan_indices = np.arange(304)
                
            #inds_removing = [7,131,48,55,243]
            #self.data = np.delete(self.data, inds_removing)
            #self.scan_indices = np.delete(self.scan_indices, inds_removing)

        # read dataset
        self._dataset_size = len(self.scan_indices)
        self.n_uniform = 400
        self.n_fine_sampled = 1800
        self.angle_every = 60 # Augmentation by rotating scans every 60 degrees
        self.fine_std = 0.05

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        scan_index = self.scan_indices[index]
        if self._mode == 'train':
            motion_id = np.random.randint(0, 6)
        else:
            motion_id = 0

        angle_y = np.random.randint(0, int(360/self.angle_every))*self.angle_every
        if motion_id == 0:
            name_voxels = self.data[index] + '/voxels_%03d_%d.npy'%(scan_index, angle_y)
            name_gt = self.data[index]+ '/vertices_gt_%03d_%d.npy'%(scan_index, angle_y)
        else:
            name_voxels = str.replace(self.data[index], '/voxels/', '/voxels_augmented/') + '/voxels_%03d_%d_%d.npy'%(scan_index, motion_id, angle_y)
            name_gt = str.replace(self.data[index],'/voxels/', '/voxels_augmented/')+ '/vertices_gt_%03d_%d_%d.npy'%(scan_index, motion_id, angle_y)

        voxels = np.load(name_voxels)
        smpl_vertices = np.load(name_gt)[:6890]

        b_min = np.array([-0.8, -0.8, -0.8])
        b_max = np.array([0.8, 0.8, 0.8])
        rand_uniform = np.random.uniform(b_min, b_max, (self.n_uniform, 3))

        smpl_inds = np.arange(6890)
        np.random.shuffle(smpl_inds)
        smpl_inds = smpl_inds[:self.n_fine_sampled]
        noise_smpl = np.random.normal(0, self.fine_std, (self.n_fine_sampled, 3))
        rand_smpl = smpl_vertices[smpl_inds] + noise_smpl

        point_pos = np.concatenate((rand_uniform, rand_smpl))

        # pack data
        sample = {
                  'input_voxels': voxels[None],
                  'input_points': point_pos,
                  'smpl_vertices': smpl_vertices,
                  }

        return sample

    def __len__(self):
        return self._dataset_size
