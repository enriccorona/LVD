#os.environ["PYOPENGL_PLATFORM"] = "osmesa" # Here we will use xvfb with frankmocap's rendering code, so don't need osmesa
import os
import glob
from options.test_options import TestOptions
from models.models import ModelsFactory
from collections import OrderedDict
import tqdm
import numpy as np
from utils.sdf import create_grid
import torch
import trimesh
from kaolin.metrics.pointcloud import sided_distance

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._display_visualizer_test()



    def voxelize_scan(self, scan):
        resolution = 128 # Voxel resolution
        b_min = np.array([-1.2, -1.2, -1.2])
        b_max = np.array([1.2, 1.2, 1.2])
        step = 3000

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
        return voxels, vertices




    def _display_visualizer_test(self):
        # set model to eval
        self._model.set_eval()
        from manopth.manolayer import ManoLayer
        mano_root = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'manopth/mano/models')
        self.MANO = ManoLayer(ncomps=12, mano_root=mano_root, use_pca=True, side='left')
        self.MANO.cuda()
        self.mano_faces = self.MANO.th_faces.cpu().data.numpy()

        path_in = 'test_data/hands/'
        scans = glob.glob(path_in + '*obj')

        self._model.set_eval()
        for index in tqdm.tqdm(range(len(scans))):
            name = str.split(str.split(scans[index], '/')[-1], '.')[0]

            scan = trimesh.load(scans[index])
            voxels, normalized_vertices = self.voxelize_scan(scan)
            voxels = torch.FloatTensor(voxels)[None, None].cuda()
            scan.vertices = normalized_vertices
            vertices_scan_torch = torch.FloatTensor(normalized_vertices).cuda()

            # Encode efficiently to input to network:
            voxels = torch.cat((torch.clamp(voxels, 0, 0.001)*100,
                                        torch.clamp(voxels, 0, 0.002)*50,
                                        torch.clamp(voxels, 0, 0.01)*20,
                                        torch.clamp(voxels, 0, 0.05)*20,
                                        torch.clamp(voxels, 0, 0.1)*15,
                                        torch.clamp(voxels, 0, 0.5)*10,
                                        voxels
                                        ), 1)

            with torch.no_grad():
                input_points = torch.zeros(1, 778, 3).cuda()
                _B = 1
                self._model._img_encoder(voxels)
                iters = 10
                inds = np.arange(778)
                for it in range(iters):
                    pred_dist = self._model._img_encoder.query(input_points)
                    pred_dist = pred_dist.reshape(_B, 778, 3, -1).permute(0, 1, 3, 2)
                    input_points = - pred_dist[:, inds, inds] + input_points

            #m_pred = trimesh.Trimesh(input_points[0].cpu().data.numpy(), self.mano_faces, process=False)

            lr = 1e-1
            parameters_smpl = OptimizationMANO().cuda()
            optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
            iterations = 300 #2000 #10000

            factor_beta_reg = 0.01
            lambda_pose = 0.01
            lambda_tot = 0.02

            # Fit MANO to prediction to get pose and shape parameters:
            for i in tqdm.tqdm(range(iterations)):
                pose, beta, trans, scale = parameters_smpl.forward()
                vertices_smpl, pred_joints = self.MANO.forward(pose, beta)
                vertices_smpl /= 100
                vertices_smpl = (vertices_smpl + trans)*scale

                loss = ((vertices_smpl - input_points)**2).sum(-1).mean()

                prior_loss = (pose**2).mean() #self.prior.forward(pose[:, 3:], beta)
                beta_loss = (beta**2).mean()
                loss = loss + prior_loss*0.00000001 + beta_loss*factor_beta_reg

                optimizer_smpl.zero_grad()
                loss.backward()
                optimizer_smpl.step()

                for param_group in optimizer_smpl.param_groups:
                    param_group['lr'] = lr*(iterations-i)/iterations


            # Fit MANO to input pointcloud to finally refine pose and shape parameters:
            optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
            for i in tqdm.tqdm(range(iterations)):
                pose, beta, trans, scale = parameters_smpl.forward()
                vertices_smpl, pred_joints = self.MANO.forward(pose, beta)
                vertices_smpl /= 100
                vertices_smpl = (vertices_smpl + trans)*scale

                # d1 = torch.sqrt(directed_distance(vertices_scan_torch, vertices_smpl[0], False)).mean()
                # d2 = torch.sqrt(directed_distance(vertices_smpl[0], vertices_scan_torch, False)).mean()

                loss = 2 * sided_distance(torch.unsqueeze(vertices_scan_torch,0), torch.unsqueeze(vertices_smpl[0],0))[0].mean()

                prior_loss = (pose**2).mean() #self.prior.forward(pose[:, 3:], beta)
                beta_loss = (beta**2).mean()
                loss = loss + prior_loss*0.00000001 + beta_loss*factor_beta_reg

                optimizer_smpl.zero_grad()
                loss.backward()
                optimizer_smpl.step()

                for param_group in optimizer_smpl.param_groups:
                    param_group['lr'] = lr*(iterations-i)/iterations

            with torch.no_grad():
                pose, beta, trans, scale = parameters_smpl.forward()
                vertices_smpl, pred_joints = self.MANO.forward(pose, beta)
                vertices_smpl /= 100
                vertices_smpl = (vertices_smpl + trans)*scale
                fit_mesh = vertices_smpl.cpu().data.numpy()[0]

            m_fit = trimesh.Trimesh(fit_mesh, np.array(self.mano_faces), process=False)
            m_fit.export('results/sample_mano_%s_fit.obj'%name);
            scan.export('results/sample_mano_%s_input_scan.obj'%name);


class OptimizationMANO(torch.nn.Module):
    def __init__(self):
        super(OptimizationMANO, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 15).cuda())
        #self.pose = torch.nn.Parameter(torch.zeros(1, 48).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 10).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        self.scale = torch.nn.Parameter(torch.ones(1).cuda()*1)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale

class OptimizationOffsets(torch.nn.Module):
    def __init__(self):
        super(OptimizationOffsets, self).__init__()
        self.offsets = torch.nn.Parameter(torch.zeros(6890, 3).cuda())

    def forward(self):
        return self.offsets


if __name__ == "__main__":
    Test()
