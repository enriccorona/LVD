import torch
from collections import OrderedDict
from utils import  util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
from utils.sdf import create_grid, eval_grid_octree, eval_grid
from torch import nn
import tqdm
from skimage import measure
import pyrender

import trimesh
from utils.SMPL import SMPL

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

        # NOTE: WE COULD ALSO TRY SHRINKING STD PROGRESSIVELY DURING TRAINING !!!

    def _init_create_networks(self):
        # generator network
        self._img_encoder = self._create_img_encoder()
        self._img_encoder.init_weights()
        if torch.cuda.is_available():
            self._img_encoder.cuda()

    def _create_img_encoder(self):
        #return NetworksFactory.get_by_name('voxel_encoder_and_predictor3', 2048, 16)
#        return NetworksFactory.get_by_name('voxel_encoder_and_predictor3', 2048, 2+14, input_dim=7, b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]))
        return NetworksFactory.get_by_name('voxel_encoder_and_predictor3', 2048, 6890*3, input_dim=7, b_min = np.array([-0.8, -0.8, -0.8]), b_max = np.array([0.8, 0.8, 0.8]))
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
        self.bce_loss = nn.BCELoss()
        self.CE = nn.CrossEntropyLoss()

    def set_input(self, input):
        self._input_voxels = input['input_voxels'].float().cuda()
        self._input_points = input['input_points'].float().cuda()
        self._target_smpl = input['smpl_vertices'].float().cuda()
        self._target_cloth_occupancy = input['occupancy_clothes'].float().cuda().unsqueeze(1)
        self._target_body_occupancy = input['occupancy_body'].float().cuda().unsqueeze(1)
        self._target_body_category = input['category_body'].long().cuda().unsqueeze(1)

#        multiplier = 100
        #multiplier = 1000
        #clamp = 0.1
#        #clamp = 0.001
#        clamp = 0.04
#        self._input_voxels = torch.clamp(self._input_voxels, 0, clamp)*multiplier
        self._input_voxels = torch.cat((torch.clamp(self._input_voxels, 0, 0.01)*100,
                        torch.clamp(self._input_voxels, 0, 0.02)*50,
                        torch.clamp(self._input_voxels, 0, 0.05)*20,
                        torch.clamp(self._input_voxels, 0, 0.1)*20,
                        torch.clamp(self._input_voxels, 0, 0.15)*15,
                        torch.clamp(self._input_voxels, 0, 0.2)*10,
                        self._input_voxels
                        ), 1)


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
                self._img_encoder(self._input_voxels)

                pred = self._img_encoder.query(self._input_points)
                self._loss_L2 = self.bce_loss(self._sigmoid(pred[:,:1]), self._target_body_occupancy) + \
                            self.bce_loss(self._sigmoid(pred[:,1:2]), self._target_cloth_occupancy)

                body_class_pred = pred[:,2:].unsqueeze(1).permute(0, 1, 3, 2)
                self.body_pred_loss = self.CE(body_class_pred[self._target_body_occupancy==1], self._target_body_category[self._target_body_occupancy==1])

        if True:
            with torch.no_grad():
                resolution = 128
                b_min = np.array([-0.8, -0.8, -0.8])
                b_max = np.array([0.8, 0.8, 0.8])
                coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

                self._img_encoder(self._input_voxels[:1])
                # TODO: Just modify eval_grid code to have a common process for this which only runs network once per point batch:
                def eval_func(points):
                    points = torch.FloatTensor(points.T).cuda().unsqueeze(0)
                    pred = self._img_encoder.query(points)
                    return pred.cpu().data.numpy()[0][0]
                sdf_body = eval_grid(coords, eval_func, num_samples=10000)
                def eval_func(points):
                    points = torch.FloatTensor(points.T).cuda().unsqueeze(0)
                    pred = self._img_encoder.query(points)
                    return pred.cpu().data.numpy()[0][1]
                sdf_clothes = eval_grid(coords, eval_func, num_samples=10000)

                try:
                    body_verts, body_faces, pred_normals, values = measure.marching_cubes(sdf_body, 0.0)
                    cloth_verts, cloth_faces, pred_normals, values = measure.marching_cubes(sdf_clothes, 0.0)
                    body_verts = body_verts / resolution * 200 + 100
                    cloth_verts = cloth_verts / resolution * 200 + 100
                    self.pred_mesh = trimesh.Trimesh(body_verts, body_faces[:,::-1])
                    self.pred_mesh_cloth = trimesh.Trimesh(cloth_verts, cloth_faces[:,::-1])
                    #self.pred_mesh_cloth = self.pred_mesh
                except:
                    try:
                        body_verts, body_faces, pred_normals, values = measure.marching_cubes_lewiner(sdf_body, 0.0)
                        cloth_verts, cloth_faces, pred_normals, values = measure.marching_cubes_lewiner(sdf_clothes, 0.0)
                        body_verts = body_verts / resolution * 200 - 100
                        cloth_verts = cloth_verts / resolution * 200 - 100
                        self.pred_mesh = trimesh.Trimesh(cloth_verts, cloth_faces[:,::-1])
                        self.pred_mesh = trimesh.Trimesh(body_verts, body_faces[:,::-1])
                        self.pred_mesh_cloth = trimesh.Trimesh(cloth_verts, cloth_faces[:,::-1])
                        #self.pred_mesh_cloth = self.pred_mesh
                    except:
                        #from IPython import embed
                        #embed()
                        pass

        return

    def optimize_parameters(self):
        if self._is_train:
            self._optimizer_img_encoder.zero_grad()
            loss_G = self._forward_G()
            loss_G.backward()
            self._optimizer_img_encoder.step()

    def _forward_G(self):
        self._img_encoder(self._input_voxels)

        pred = self._img_encoder.query(self._input_points)
        self._loss_L2 = self.bce_loss(self._sigmoid(pred[:,:1]), self._target_body_occupancy) + \
                    self.bce_loss(self._sigmoid(pred[:,1:2]), self._target_cloth_occupancy)

        body_class_pred = pred[:,2:].unsqueeze(1).permute(0, 1, 3, 2)
        self.body_pred_loss = self.CE(body_class_pred[self._target_body_occupancy==1], self._target_body_category[self._target_body_occupancy==1])
        #self.body_pred_loss = self.CE(pred[:,2:], self._target_body_category)

        self._loss_L2 = self._loss_L2 + self.body_pred_loss/5

        # TODO ADD Deformation!
        return self._loss_L2

    def get_current_errors(self):
        loss_dict = OrderedDict([
                                 ('Distance loss', self._loss_L2.cpu().data.numpy()),
                                 ('Body Cat. loss', self.body_pred_loss.cpu().data.numpy()),
                                 ])
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        if type(self.pred_mesh) == type(None):
            return visuals
        trim_mesh = trimesh.Trimesh(self._target_smpl[0].cpu().data.numpy(), self.SMPL.faces, process=False)
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
        visuals['smpl_optimized'] = color


        trim_mesh = self.pred_mesh_cloth
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
        visuals['smpl_target'] = color

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
