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
from utils.sdf import create_grid, eval_grid_octree, eval_grid
import torch
import trimesh
import pyrender
from utils import plots as plot_utils
from skimage import measure
from utils import image_fitting
try:
    import mediapipe as mp
except:
    raise("Please install mediapipe to predict rough segmentation masks")

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)

        if 'mask' in self._opt.name:
            self.with_mask = True
        else:
            self.with_mask = False

        self.fit_SMPL_parameters = True
        if self.fit_SMPL_parameters:
            from utils.prior import SMPLifyAnglePrior, MaxMixturePrior
            self.prior = MaxMixturePrior(prior_folder='utils/prior/', num_gaussians=8) #.get_gmm_prior()
            self.prior = self.prior.cuda()

        # Initialize mediapipe
        self.pose_mp = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5)

        # Run forward pass on all images
        self._display_visualizer_test()

    def crop_image(self, img, mask):
        (x,y,w,h) = cv2.boundingRect(np.uint8(mask))
        mask = mask[y:y+h,x:x+w]
        img = img[y:y+h,x:x+w]

        # Prepare new image, with correct size:
        margin = 1.1
        im_size = 256
        clean_im_size = im_size/margin
        size = int((max(w, h)*margin)//2)
        new_x = size - w//2
        new_y = size - h//2
        new_img = np.zeros((size*2, size*2, 3))
        new_mask = np.zeros((size*2, size*2))
        new_img[new_y:new_y + h, new_x:new_x+w] = img
        new_mask[new_y:new_y + h, new_x:new_x+w] = mask

        # Resizing cropped and centered image to desired size:
        img = cv2.resize(new_img, (im_size,im_size))
        mask = cv2.resize(new_mask, (im_size,im_size))

        return img, mask

    def _display_visualizer_test(self):

        # set model to eval
        self._model.set_eval()

        path_in = 'test_data/images/'
        names = glob.glob(path_in + '*img.png')
        names.sort()

        renderer = image_fitting.meshRenderer()

        # Forward pass for all images:
        for index in tqdm.tqdm(range(len(names))):
            print("Loading input image and mask")
            name = str.split(str.split(names[index], '/')[-1], '.')[0]

            # Load input image and mask:
            img = cv2.imread(names[index])[:,:,::-1]

            # Predict mask. We're actually using the pose estimation method here. 
            # TODO: Just use the correct segmentation method
            results_mediapipe = self.pose_mp.process(img)
            if not results_mediapipe.pose_landmarks:
                continue # No person in input image
            mask = (results_mediapipe.segmentation_mask>0.5)*255
            img, mask = self.crop_image(img, mask)

            # Normalize values as in ImageNet:
            img = img / 255
            imgtensor = img - [0.485, 0.456, 0.406]
            imgtensor = imgtensor / [0.229, 0.224, 0.225]
            imgtensor = imgtensor.transpose(2, 0, 1)
            # Mask background out:
            imgtensor[:, mask[:,:]==0] = 0
            if self.with_mask:
                imgtensor = torch.FloatTensor(np.concatenate((imgtensor, mask[None]))).cuda().unsqueeze(0)
            else:
                imgtensor = torch.FloatTensor(imgtensor).cuda().unsqueeze(0)

            print("Running forward pass")
            # LVD's forward pass:
            self._model.set_eval()
            with torch.no_grad():
                input_points = torch.zeros(1, 6890, 3).cuda()
                self._model._img_encoder(imgtensor)
                iters = 5
                inds = np.arange(6890)
                for it in range(iters):
                    pred_dist = self._model._img_encoder.query_test(input_points)[None]
                    input_points = - pred_dist + input_points

                pred_mesh = input_points[0].cpu().data.numpy()

            # Fit SMPL using SMPL layer to obtain pose and shape parameters. This should not be necessary:
            if self.fit_SMPL_parameters:
                print("Fitting SMPL")
                parameters_smpl = OptimizationSMPL().cuda()
                lr = 1e-1
                optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
                iterations = 500
                ind_verts = np.arange(6890)
                pred_mesh_torch = torch.FloatTensor(pred_mesh).cuda()

                factor_beta_reg = 0.02
                for i in range(iterations):
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
                    pred_mesh = vertices_smpl.cpu().data.numpy()

            # Render image of the SMPL reconstruction:
            resolution = 1024

            background_img = np.uint8(np.ones((resolution, resolution, 3))*255)
            m = trimesh.Trimesh(pred_mesh, np.array(self._model.SMPL.faces), process=False)

            m.vertices[:, 1] *= -1
            m.vertices[:, 2] *= -1

            m.vertices = (m.vertices/0.2 + 512)/1024*resolution
            m.visual.vertex_colors[:,:3] = np.uint8([246,212,185])
            image = image_fitting.render_image_projection_multiperson_wrenderer(background_img, [m], [m.vertex_normals], [m.visual.vertex_colors], [1], [[0, 0]], [[0, 0]], [1], mode='rgb', renderer=renderer, done_projection=True)
            img = cv2.resize(np.uint8(img*255), (resolution, resolution))
            full_image = np.concatenate((img, image), 1)[:,:,::-1]
            cv2.imwrite("results/prediction_%s.png"%name, full_image)

class OptimizationSMPL(torch.nn.Module):
    def __init__(self):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 300).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        self.scale = torch.nn.Parameter(torch.ones(1).cuda()*90)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale

if __name__ == "__main__":
    Test()
