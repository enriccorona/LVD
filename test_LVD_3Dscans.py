import os
#os.environ["PYOPENGL_PLATFORM"] = "osmesa" # Here we will use xvfb with frankmocap's rendering code, so don't need osmesa
import glob
import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os
import cv2
import tqdm
import numpy as np
from utils.sdf import create_grid
import torch
import trimesh
from utils.prior import SMPLifyAnglePrior, MaxMixturePrior
from utils import plots as plot_utils
from skimage import measure

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()

        self._tb_visualizer = TBVisualizer(self._opt)
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        
        # SMPL prior to fit pose and shape vectors:
        self.prior = MaxMixturePrior(prior_folder='utils/prior/', num_gaussians=8) #.get_gmm_prior()
        self.prior = self.prior.cuda()

        self._display_visualizer_test()

    def voxelize_scan(self, scan):
        resolution = 128 # Voxel resolution
        b_min = np.array([-0.8, -0.8, -0.8]) 
        b_max = np.array([0.8, 0.8, 0.8])
        step = 5000

        vertices = scan.vertices
        bounding_box = (vertices.max(0) - vertices.min(0))[1]

        vertices = vertices / bounding_box * 1.5
        trans = (vertices.max(0) + vertices.min(0))/2
        vertices = vertices - trans

        factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

        print("Voxelizing input scan:")
        # NOTE: It was easier and faster to just get distance to vertices, instead of voxels carrying inside/outside information,
        # which will only be possible for closed watertight meshes.
        with torch.no_grad():
            v = torch.FloatTensor(vertices).cuda()
            coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
            points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
            points_npy = coords.reshape(3, -1).T
            iters = len(points)//step + 1

            all_distances = []
            for it in range(iters):
                it_v = points[it*step:(it+1)*step]
                it_v_npy = points_npy[it*step:(it+1)*step]
                distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
                #contain = scan.contains(it_v_npy)
                distance = distance.min(0)[0].cpu().data.numpy()
                all_distances.append(distance)
            #contains = scan.contains(points_npy)
            signed_distance = np.concatenate(all_distances)

        voxels = signed_distance.reshape(resolution, resolution, resolution)
        return voxels


    def _display_visualizer_test(self):
        # set model to eval
        self._model.set_eval()

        path_in = 'test_data/3dscan/'
        scans = glob.glob(path_in + '*obj')

        #for index in tqdm.tqdm(range(304)):
        for index in tqdm.tqdm(range(len(scans))):
            scan = trimesh.load(scans[index], process=False)
            voxels = self.voxelize_scan(scan)

            voxels = torch.FloatTensor(voxels)[None, None].cuda()

            # Encode efficiently to input to network:
            voxels = torch.cat((torch.clamp(voxels, 0, 0.01)*100,
                                torch.clamp(voxels, 0, 0.02)*50,
                                torch.clamp(voxels, 0, 0.05)*20,
                                torch.clamp(voxels, 0, 0.1)*20,
                                torch.clamp(voxels, 0, 0.15)*15,
                                torch.clamp(voxels, 0, 0.2)*10,
                                voxels
                                ), 1)

            print("Forward pass:")
            self._model.set_eval()
            with torch.no_grad():
                input_points = torch.zeros(1, 6890, 3).cuda()
                _B = 1
                self._model._img_encoder(voxels)
                iters = 10
                inds = np.arange(6890)
                for it in range(iters):
                    pred_dist = self._model._img_encoder.query(input_points)
                    pred_dist = pred_dist.reshape(_B, 6890, 3, -1).permute(0, 1, 3, 2)
                    input_points = - pred_dist[:, inds, inds] + input_points
                pred_mesh = input_points[0].cpu().data.numpy()

            print("Fitting SMPL on the prediction:")
            parameters_smpl = OptimizationSMPL().cuda()
            lr = 1e-1
            optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
            iterations = 500
            ind_verts = np.arange(6890)
            pred_mesh_torch = torch.FloatTensor(pred_mesh).cuda()
            factor_beta_reg = 0.01
            for i in tqdm.tqdm(range(iterations)):
                pose, beta, trans, scale = parameters_smpl.forward()
                vertices_smpl = (self._model.SMPL.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
                distances = torch.abs(pred_mesh_torch - vertices_smpl)
                loss = distances.mean()

                prior_loss = self.prior.forward(pose[:, 3:], beta)
                beta_loss = (beta**2).mean()
                loss = loss + prior_loss*0.00000001 + beta_loss*factor_beta_reg

                optimizer_smpl.zero_grad()
                loss.backward()
                optimizer_smpl.step()

                for param_group in optimizer_smpl.param_groups:
                    param_group['lr'] = lr*(iterations-i)/iterations

            with torch.no_grad():
                pose, beta, trans, scale = parameters_smpl.forward()
                vertices_smpl = (self._model.SMPL.forward(theta=pose, beta=beta, get_skin=True)[0][0] + trans)*scale
                fit_mesh = vertices_smpl.cpu().data.numpy()

            m = trimesh.Trimesh(pred_mesh, np.array(self._model.SMPL.faces), process=False)
            m_fit = trimesh.Trimesh(fit_mesh, np.array(self._model.SMPL.faces), process=False)

            m.export('results/sample_%d_smpl_pred.obj'%index);
            m_fit.export('results/sample_%d_smpl_fit.obj'%index);
            scan.export('results/sample_%d_input_scan.obj'%index);



class OptimizationSMPL(torch.nn.Module):
    def __init__(self):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 300).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        self.scale = torch.nn.Parameter(torch.ones(1).cuda()*1)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale

if __name__ == "__main__":
    Test()
