import torch
from collections import OrderedDict
from utils import  util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
from torch import nn
import tqdm
from skimage import measure
import pyrender
import trimesh

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
        from manopth.manolayer import ManoLayer
        self.MANO = ManoLayer(ncomps=45, mano_root='/home/ecorona/libraries/manopth/mano/models/', use_pca=False)

        self.pred_mesh = None
        # Sigma different per each axis

        # NOTE: WE COULD ALSO TRY SHRINKING STD PROGRESSIVELY DURING TRAINING !!!

    def _init_create_networks(self):
        # generator network
        self._img_encoder = self._create_img_encoder()
        self._img_encoder.init_weights()
        if torch.cuda.is_available():
            self._img_encoder.cuda()

    def _create_img_encoder(self):
        return NetworksFactory.get_by_name('LVD_voxels', 2048, 778*3, input_dim=7, b_min = np.array([-1.2, -1.2, -1.2]), b_max = np.array([1.2, 1.2, 1.2]))
        #return NetworksFactory.get_by_name('img_encoder', 6 +12+48+49+48+48+24)

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

        self._sigmoid = nn.Sigmoid()

    def set_input(self, input):
        self._input_voxels = input['input_voxels'].float().cuda()
        self._input_points = input['input_points'].float().cuda()
        self._target_mano = input['mano_vertices'].float().cuda()

        self._input_voxels = torch.cat((torch.clamp(self._input_voxels, 0, 0.001)*100,
                                        torch.clamp(self._input_voxels, 0, 0.002)*50,
                                        torch.clamp(self._input_voxels, 0, 0.01)*20,
                                        torch.clamp(self._input_voxels, 0, 0.05)*20,
                                        torch.clamp(self._input_voxels, 0, 0.1)*15,
                                        torch.clamp(self._input_voxels, 0, 0.5)*10,
                                        self._input_voxels
                                        ), 1)

        #self._input_voxels = torch.cat((torch.clamp(self._input_voxels, 0, 0.01)*100,
        #                                torch.clamp(self._input_voxels, 0, 0.02)*50,
        #                                torch.clamp(self._input_voxels, 0, 0.05)*20,
        #                                torch.clamp(self._input_voxels, 0, 0.1)*20,
        #                                torch.clamp(self._input_voxels, 0, 0.15)*15,
        #                                torch.clamp(self._input_voxels, 0, 0.2)*10,
        #                                self._input_voxels
        #                                ), 1)

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
            with torch.no_grad():
                _B = self._input_voxels.shape[0]
                _numpoints = self._input_points.shape[1]

                self._img_encoder(self._input_voxels)
                pred = self._img_encoder.query(self._input_points)
                pred = pred.reshape(_B, 778, 3, _numpoints).permute(0, 1, 3, 2)
                dist = self._input_points.unsqueeze(1) - self._target_mano.unsqueeze(2)
                #clamp = 0.6
                #clamp = 0.1
                clamp = 0.4
                dist = torch.clamp(dist, -1*clamp, clamp)

                self._loss_L2 = torch.abs(dist - pred) # Actually L1 for now
                self._loss_L2 = self._loss_L2.mean()

        if True:
            with torch.no_grad():
                input_points = torch.zeros(1, 778, 3).cuda()
                _B = 1
                self._img_encoder(self._input_voxels[:1])
                iters = 10
                inds = np.arange(778)
                for it in range(iters):
                    pred_dist = self._img_encoder.query(input_points)
                    pred_dist = pred_dist.reshape(_B, 778, 3, -1).permute(0, 1, 3, 2)
                    input_points = - pred_dist[:, inds, inds] + input_points
                self.pred_mesh = input_points[0].cpu().data.numpy()

        return

    def optimize_parameters(self):
        if self._is_train:
            self._optimizer_img_encoder.zero_grad()
            loss_G = self._forward_G()
            loss_G.backward()
            self._optimizer_img_encoder.step()

    def _forward_G(self):
        self._img_encoder(self._input_voxels)
        _B = self._input_voxels.shape[0]
        _numpoints = self._input_points.shape[1]

        pred = self._img_encoder.query(self._input_points)
        with torch.no_grad():
            dist = self._input_points.unsqueeze(1) - self._target_mano.unsqueeze(2)
            #clamp = 0.6
            #clamp = 0.1
            #clamp = 0.3
            clamp = 0.4
            dist = torch.clamp(dist, -1*clamp, clamp)

        pred = pred.reshape(_B, 778, 3, _numpoints).permute(0, 1, 3, 2)

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

        trim_mesh = trimesh.Trimesh(self._target_mano[0].cpu().data.numpy(), self.MANO.th_faces.cpu().data.numpy(), process=False)
        trim_mesh.visual.vertex_colors[:, :3] = [160, 160, 255]
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(trim_mesh, smooth=False)

        # Overwrite pyrender's mesh normals:
        scene.add(pyrender_mesh)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        scene.add(light, pose=np.eye(4))

        dist = 2.5
        angle = 20
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, 0],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
        #camera_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0.8],[0, 0, 0, 1]])
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, znear=0.5, zfar=5)
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(512, 512)
        color, depth = renderer.render(scene)
        visuals['mano_target'] = color


        if type(self.pred_mesh) == type(None):
            return visuals

        trim_mesh = trimesh.Trimesh(self.pred_mesh, self.MANO.th_faces.cpu().data.numpy()[:,::-1], process=False)
        trim_mesh.visual.vertex_colors[:, :3] = [160, 160, 255]
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(trim_mesh, smooth=False)

        # Overwrite pyrender's mesh normals:
        scene.add(pyrender_mesh)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        scene.add(light, pose=np.eye(4))

        dist = 2.5
        angle = 20
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, 0],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
        #camera_pose = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0.8],[0, 0, 0, 1]])
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, znear=0.5, zfar=5)
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(512, 512)
        color, depth = renderer.render(scene)
        visuals['mano_pred'] = color

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
