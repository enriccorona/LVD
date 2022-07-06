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
import imgaug
import kaolin
import glob
import trimesh
import cv2

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset'
        self.data_dir = 'data_demo/'
        self.scans_dir = 'data_demo/scan/'

        # read dataset
        self.get_data()
        self._dataset_size = len(self.renders)*360 # We have 360 viewpoints of the scan

        self.n_uniform = 200
        self.n_fine_sampled = 200
        self.fine_std = 0.8
        self.b_min = np.array([-130, -130, -130])
        self.b_max = np.array([130, 130, 130])

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        angle_y = index
        render_name = self.renders[index//360]
        render_name = str.replace(render_name, 'render_0', 'render_%d'%angle_y)
        mask_name = str.replace(str.replace(render_name, 'RENDER', 'MASK'), '00.png', '0.png')
        name_scan = str.split(str.split(render_name, '/')[-1], '_scaled')[0]

        img = cv2.imread(render_name)
        img = cv2.resize(img, (256, 256))
        mask = cv2.imread(mask_name, 0)
        mask = cv2.resize(mask, (256, 256))/255.0

        if self._mode == 'train':
            if np.random.rand() < 0.3:
                img = imgaug.augmenters.MultiplyAndAddToBrightness(mul=np.random.uniform(0.5, 1.5), add=np.random.uniform(-30, 30))(image=img)
            if np.random.rand() < 0.3:
                img = imgaug.augmenters.MultiplyHueAndSaturation(np.random.uniform(0.75, 1.25))(image=img)
            if np.random.rand() < 0.3:
                img = imgaug.augmenters.GammaContrast(np.random.uniform(0.7, 1.3))(image=img)

        # Normalize input image:
        img = img/255.0
        img = img - [0.485, 0.456, 0.406]
        img = img / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)
        img[:, mask==0] = 0
        mask = mask[...,None]

        img = np.concatenate((img, mask.transpose(2, 0, 1)))

        smpl_verts_path = self.scans_dir + name_scan + '_j_00_smplfit_vertices.npy'
        smpl_vertices = np.load(smpl_verts_path)[:6890]

        R = np.load(self.data_dir + '/TRANSFORMATION/%s_j_00_R_%d_0_00.npy'%(name_scan + '_scaled.obj', angle_y))
        smpl_trans = np.load(self.data_dir + '/TRANSFORMATION/%s_j_00_T_%d_0_00.npy'%(name_scan + '_scaled.obj', angle_y))
        smpl_scale = np.load(self.data_dir + '/TRANSFORMATION/%s_j_00_S_%d_0_00.npy'%(name_scan + '_scaled.obj', angle_y))
        smpl_vertices = np.array(np.matmul(R, ((smpl_vertices - smpl_trans)*smpl_scale).T).T)

        rand_uniform = np.random.uniform(self.b_min, self.b_max, (self.n_uniform, 3))

        smpl_inds = np.arange(6890)
        np.random.shuffle(smpl_inds)
        smpl_inds = smpl_inds[:self.n_fine_sampled]
        #noise_smpl = np.random.normal(0, 0, (self.n_fine_sampled, 3))
        noise_smpl = np.random.normal(0, self.fine_std, (self.n_fine_sampled, 3))
        rand_smpl = smpl_vertices[smpl_inds] + noise_smpl

        point_pos = np.concatenate((rand_uniform, rand_smpl))

        # pack data
        sample = {
                  'input_image': img,
                  'input_points': point_pos,
                  'smpl_vertices': smpl_vertices,
                  }

        return sample

    def __len__(self):
        return self._dataset_size

    def get_data(self):
        # GET LIST OF DATA:
        self.renders = glob.glob(self.data_dir + '/RENDER/*render_0_0_00.png')
        self.renders.sort()

