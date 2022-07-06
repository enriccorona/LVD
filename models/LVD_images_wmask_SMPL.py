import torch
from collections import OrderedDict
from utils import  util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
from utils.sdf import create_grid
from torch import nn
import tqdm
from skimage import measure
import pyrender

import trimesh
from utils.SMPL import SMPL

def gaussian(x, mu=0, sig=1):
    return np.exp(-1*np.power(x - mu, 2.).sum(-1) / (2 * np.power(sig, 2.)))

def gaussian_tensor(x, mu=0, sig=1):
    return torch.exp(-1*((x - mu)**2).sum(-1) / (2 * sig**2))
    #return torch.exp(-1*torch.pow(x - mu, 2.).sum(-1) / (2 * torch.pow(sig, 2.)))

class OptimizationSMPL(nn.Module):
    def __init__(self):
        super(OptimizationSMPL, self).__init__()

        self.pose = nn.Parameter(torch.zeros(1, 72).cuda())
        self.beta = nn.Parameter((torch.zeros(1, 300).cuda()-0.5/100))
        self.trans = nn.Parameter(torch.zeros(1, 3).cuda())

    def forward(self):
        return self.pose, self.beta, self.trans


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'model1'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # init
        self._init_losses()
        self.SMPL = SMPL('utils/neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).cuda()

        self.pred_mesh = None
        # Sigma different per each axis
        self.gaussian_sigma = 30
        #self.gaussian_sigma = 0.5
        #self.gaussian_sigma = 5
        #self.gaussian_sigma = 0.5

    def _init_create_networks(self):
        # generator network
        self._img_encoder = self._create_img_encoder()
        self._img_encoder.init_weights()
        if torch.cuda.is_available():
            self._img_encoder.cuda()

    def _create_img_encoder(self):
        return NetworksFactory.get_by_name('LVD_images', input_channels=4, pred_dimensions=6890*3)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer_img_encoder = torch.optim.Adam(self._img_encoder.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])

        self.mesh = None

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):
        # init losses G
        self._loss_L2 = torch.FloatTensor([0]).cuda()
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self._sigmoid = nn.Sigmoid()

    def set_input(self, input):
        self._input_image = input['input_image'].float().cuda()
        self._input_points = input['input_points'].float().cuda()
        self._target_smpl = input['smpl_vertices'].float().cuda()

        return 

    def set_train(self):
        self._img_encoder.train()
        self._is_train = True

    def set_eval(self):
        self._img_encoder.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, interpolate=0, resolution=128):
        if not self._is_train:
            # Reconstruct first 
            _B = self._input_image.shape[0]
            _numpoints = self._input_points.shape[1]
            with torch.no_grad():
                self._img_encoder(self._input_image)
                pred = self._img_encoder.query(self._input_points)
                dist = self._input_points.unsqueeze(1) - self._target_smpl.unsqueeze(2)
                clamp = 30
                dist = torch.clamp(dist, -1*clamp, clamp)

                pred = pred.reshape(_B, 6890, 3, _numpoints).permute(0, 1, 3, 2)

                #gt = gt * 10
                self._loss_L2 = torch.abs(dist - pred) # Actually L1 for now
                self._loss_L2 = self._loss_L2.mean()

        if True:
            with torch.no_grad():
                input_points = torch.zeros(1, 6890, 3).cuda()
                _B = 1
                self._img_encoder(self._input_image[:1])
                iters = 60
                inds = np.arange(6890)
                for it in range(iters):
                    pred_dist = self._img_encoder.query(input_points)
                    pred_dist = pred_dist.reshape(_B, 6890, 3, -1).permute(0, 1, 3, 2)
                    input_points = - pred_dist[:, inds, inds] + input_points
                    #print(torch.abs(pred_dist[:,inds, inds]).mean())
                self.pred_mesh = input_points[0].cpu().data.numpy()
                

        else:
            with torch.no_grad():
                _B = 1
                resolution = 64
                #resolution = 32
                #resolution = 256
                b_min = np.array([-130, -130, -130])
                b_max = np.array([130, 130, 130])
                coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
                coords_tensor = torch.FloatTensor(coords).cuda().reshape(1, 3, -1).permute(0, 2, 1)
                norm_div = resolution**3
                sum_pred_mesh = torch.zeros(6890, 3).cuda()
                self._img_encoder(self._input_image[:1])
                step = 10000
                iterations = coords_tensor.shape[1]//step+1
                for i in range(iterations):
                    batch_coords = coords_tensor[:, i*step:(i+1)*step]
                    pred_dist = self._img_encoder.query(batch_coords)
                    pred_dist = pred_dist.reshape(_B, 6890, 3, -1).permute(0, 1, 3, 2)
                    pred_pos = - pred_dist + batch_coords.unsqueeze(1)
                    sum_pred_mesh += pred_pos.sum(2)[0]
                self.pred_mesh = (sum_pred_mesh/norm_div).cpu().data.numpy()

        return

    def optimize_parameters(self):
        if self._is_train:
            self._optimizer_img_encoder.zero_grad()
            loss_G = self._forward_G()
            loss_G.backward()
            self._optimizer_img_encoder.step()

    def _forward_G(self):
        self._img_encoder(self._input_image)
        pred = self._img_encoder.query(self._input_points)
        _B = self._input_image.shape[0]
        _numpoints = self._input_points.shape[1]

        pred = self._img_encoder.query(self._input_points)
        with torch.no_grad():
            dist = self._input_points.unsqueeze(1) - self._target_smpl.unsqueeze(2)
            clamp = 30
            dist = torch.clamp(dist, -1*clamp, clamp)

        pred = pred.reshape(_B, 6890, 3, _numpoints).permute(0, 1, 3, 2)

        #gt = gt * 10
        self._loss_L2 = torch.abs(dist - pred) # Actually L1 for now
        self._loss_L2 = self._loss_L2.mean()

        # TODO ADD Deformation!
        return self._loss_L2

    def get_current_errors(self):
        loss_dict = OrderedDict([
                                 ('Distance loss', self._loss_L2.cpu().data.numpy()),
                                 ])
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()
        visuals['input_image'] = (self._input_image[0, :3].cpu().data.numpy().transpose(1,2,0)* [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        visuals['input_mask'] = self._input_image[0, 3].cpu().data.numpy()*255

        if type(self.pred_mesh) == type(None):
            return visuals

        trim_mesh = trimesh.Trimesh(self.pred_mesh/100, self.SMPL.faces, process=False)
        trim_mesh.visual.vertex_colors[:, :3] = [160, 160, 255]
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(trim_mesh, smooth=False)

        # Overwrite pyrender's mesh normals:
        scene.add(pyrender_mesh)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        scene.add(light, pose=np.eye(4))

        dist = 1.5
        angle = 20
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, 0],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
        #camera_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0.8],[0, 0, 0, 1]])
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, znear=0.5, zfar=5)
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(512, 512)
        color, depth = renderer.render(scene)
        visuals['smpl_predicted'] = color

        trim_mesh = trimesh.Trimesh(self._target_smpl[0].cpu().data.numpy()/100, self.SMPL.faces, process=False)
        trim_mesh.visual.vertex_colors[:, :3] = [160, 160, 255]
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(trim_mesh, smooth=False)

        # Overwrite pyrender's mesh normals:
        scene.add(pyrender_mesh)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        scene.add(light, pose=np.eye(4))

        dist = 1.5
        angle = 20
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, 0],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
        #camera_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0.8],[0, 0, 0, 1]])
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, znear=0.5, zfar=5)
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(512, 512)
        color, depth = renderer.render(scene)
        visuals['smpl_groundtruth'] = color

        self.pred_mesh = None
        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._img_encoder, 'img_encoder', label)

        # save optimizers
        self._save_optimizer(self._optimizer_img_encoder, 'img_encoder', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._img_encoder, 'img_encoder', load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer_img_encoder, 'img_encoder', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_img_encoder.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))
