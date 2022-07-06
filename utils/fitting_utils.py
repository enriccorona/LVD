
import torch
import pickle as pkl
import torchgeometry
from HierarchicalProbabilistic3DHuman.predict.predict_hrnet import predict_hrnet
from HierarchicalProbabilistic3DHuman.utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from HierarchicalProbabilistic3DHuman.utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from HierarchicalProbabilistic3DHuman.models.canny_edge_detector import CannyEdgeDetector
from HierarchicalProbabilistic3DHuman.models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from HierarchicalProbabilistic3DHuman.models.pose2D_hrnet import PoseHighResolutionNet
from HierarchicalProbabilistic3DHuman.utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from smplx.lbs import batch_rodrigues

def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
    rotation_matrix (Tensor): rotation matrix.

    Returns:
    Tensor: Rodrigues vector transformation.

    Shape:
    - Input: :math:`(N, 3, 4)`
    - Output: :math:`(N, 3)`
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return torchgeometry.quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
    rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
    Tensor: the rotation in quaternion

    Shape:
    - Input: :math:`(N, 3, 4)`
    - Output: :math:`(N, 4)`
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                        type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
                        "Input size must be a three dimensional tensor. Got {}".format(
                        rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
                        "Input size must be a N x 3 x 4  tensor. Got {}".format(
                            rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
    t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
    rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
    rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
    t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
    rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
    rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
    rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
    rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2.float() * mask_d0_d1.float()
    mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
    mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
    mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q







class SMPL_Sengupta:
    def __init__(self):
        # Sengupta model for first estimation of SMPL:
        self.edge_detect_model = CannyEdgeDetector(non_max_suppression=True,
                                                   gaussian_filter_std=1.0,
                                                   gaussian_filter_size=5,
                                                   threshold=0.0).cuda()
        with open('/home/ecorona/libraries/HierarchicalProbabilistic3DHuman/pose_shape_cfg.pkl', 'rb') as f:
            self.pose_shape_cfg = pkl.load(f)
        with open('/home/ecorona/libraries/HierarchicalProbabilistic3DHuman/pose2D_hrnet_cfg.pkl', 'rb') as f:
            pose2D_hrnet_cfg = pkl.load(f)
        with open('/home/ecorona/libraries/HierarchicalProbabilistic3DHuman/hrnet_cfg.pkl', 'rb') as f:
            self.hrnet_cfg = pkl.load(f)
        self.pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                                            config=self.pose_shape_cfg).cuda()
        checkpoint = torch.load('/home/ecorona/libraries/HierarchicalProbabilistic3DHuman/model_files/poseMF_shapeGaussian_net_weights.tar')
        self.hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).cuda()
        hrnet_checkpoint = torch.load('/home/ecorona/libraries/HierarchicalProbabilistic3DHuman/model_files/pose_hrnet_w48_384x288.pth')
        self.hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
        self.pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])

        self.pose_shape_dist_model.eval()
        self.hrnet_model.eval()

    def forward(self, image):
        '''
        SMPL initialization from Sengupta et al (ICCV'21) that predicts pose and shape
        from an input image, in a forward pass.
        '''
        hrnet_output = predict_hrnet(hrnet_model=self.hrnet_model,
                                hrnet_config=self.hrnet_cfg,
                                object_detect_model=None,
                                image=torch.FloatTensor(image).permute(2,0,1).cuda()/255,
                                object_detect_threshold=0.95,
                                bbox_scale_factor=1.2)
        hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                            hrnet_output['cropped_image'].shape[2]]],
                                            dtype=torch.float32).cuda()* 0.5
        hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                            dtype=torch.float32,
                                            ).cuda()
        cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(self.hrnet_cfg.MODEL.IMAGE_SIZE[0], self.hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                      output_wh=(self.pose_shape_cfg.DATA.PROXY_REP_SIZE, self.pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                      num_to_crop=1,
                                                      device=hrnet_input_height.device,
                                                      joints2D=hrnet_output['joints2D'][None, :, :],
                                                      rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                      bbox_centres=hrnet_input_centre,
                                                      bbox_heights=hrnet_input_height,
                                                      bbox_widths=hrnet_input_height,
                                                      orig_scale_factor=1.0)
        edge_detector_output = self.edge_detect_model(cropped_for_proxy['rgb'])
        proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if self.pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
        proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                         img_wh=self.pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                         std=self.pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
        hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > 0.75 #joints2Dvisib_threshold
        hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
        proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
        proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)
        # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
        pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
        pred_shape_dist, pred_glob, pred_cam_wp = self.pose_shape_dist_model(proxy_rep_input)
        pred_glob_rotmats = rot6d_to_rotmat(pred_glob)
        pred_glob_rotmats = torch.matmul(pred_glob_rotmats, torch.FloatTensor([[1,0,0],[0,-1,0],[0,0,-1]]).cuda())
        global_rot = rotation_matrix_to_angle_axis(torch.cat((pred_glob_rotmats, torch.zeros(1, 3, 1).cuda() * 1.0), -1).float())
        pose = rotation_matrix_to_angle_axis(torch.cat((pred_pose_rotmats_mode, torch.zeros(1, 23, 3, 1).cuda() * 1.0), -1).float()[0]).reshape(1,-1)
        shape = pred_shape_dist.loc
        return global_rot, pose, shape, pred_cam_wp # pred_cam is (scale, trans x, trans y)
