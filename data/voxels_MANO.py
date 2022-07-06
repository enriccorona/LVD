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
import pickle as pkl
from PIL import Image
import kaolin
import glob
import trimesh
import cv2

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset'
        self.scans_dir = '/home/ecorona/cvpr22/reconstruction/data/mano/voxels/train/'
        self.scans_dir_test = '/home/ecorona/cvpr22/reconstruction/data/mano/voxels/test/'

        self.train_inds = np.load('train_inds_mano.npy')
        self.test_inds = np.load('test_inds_mano.npy')

        # read dataset
        if self._mode == 'train':
            self._dataset_size = len(self.train_inds)*6 # x6 because we are rotating each hand every 60 degrees to augment the data
        else:
            self._dataset_size = len(self.test_inds)*6

        self.n_uniform = 400
        self.n_fine_sampled = 600
        self.angle_every = 60
        self.fine_std = 0.05

        from manopth.manolayer import ManoLayer
        self.MANO = ManoLayer(ncomps=45, mano_root='/home/ecorona/libraries/manopth/mano/models/', use_pca=False, side='left')
        self.mano_faces = self.MANO.th_faces.cpu().data.numpy()

        self.inds_hand = self.MANO.th_J_regressor.argmax(0)

    def __getitem__(self, index):
        #index = 0
        assert (index < self._dataset_size)
        angle_id = index % 6
        dir_ = self.scans_dir 
        if self._mode == 'train':
            scan_id = self.train_inds[index//6]
        else:
            scan_id = self.test_inds[index//6]

        name_voxels = dir_ + '/voxels_%03d_%d.npy'%(scan_id, angle_id*60)
        name_gt = dir_+ '/vertices_gt_%03d_%d.npy'%(scan_id, angle_id*60)
        name_vertices_scan = dir_+ '/vertices_scan_%03d_%d.npy'%(scan_id, angle_id*60)
        name_faces_scan = dir_+ '/faces_%03d.npy'%(scan_id)

        vertices_scan = np.load(name_vertices_scan)
        faces_scan = np.load(name_faces_scan)
        scan = trimesh.Trimesh(vertices_scan, faces_scan)

        voxels = np.load(name_voxels)
        mano_vertices = np.load(name_gt)

        b_min = np.array([-1.2, -1.2, -1.2])
        b_max = np.array([1.2, 1.2, 1.2])
        rand_uniform = np.random.uniform(b_min, b_max, (self.n_uniform, 3))
        
        mano_inds = np.arange(778)
        np.random.shuffle(mano_inds)
        mano_inds = mano_inds[:self.n_fine_sampled]
        noise_mano = np.random.normal(0, self.fine_std, (self.n_fine_sampled, 3))
        rand_mano = mano_vertices[mano_inds] + noise_mano
        point_pos = np.concatenate((rand_uniform, rand_mano))

        # Retrieve occupancy:
        mano_mesh = trimesh.Trimesh(mano_vertices, self.mano_faces)
        mano_mesh = mano_mesh.subdivide()
        dist = ((point_pos[None] - np.array(mano_mesh.vertices[:,None]))**2).sum(-1)
        occupancy_body = dist.min(0)
        threshold = 0.001
        occupancy_body = occupancy_body < threshold

        # Retrieve closest part:
        dist = ((point_pos[None] - mano_vertices[:,None])**2).sum(-1)
        closest = dist.argmin(0)
        category_hand = self.inds_hand[closest]

        # pack data
        sample = {
                  'input_voxels': voxels[None],
                  'input_points': point_pos,
                  'mano_vertices': mano_vertices,
                  'occupancy_body': occupancy_body,
                  'category_body': category_hand,
                  }
        return sample

    def __len__(self):
        return self._dataset_size
