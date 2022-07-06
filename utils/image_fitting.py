import numpy as np
import trimesh
import torch
import cv2
from .sdf import create_grid, eval_grid_octree, eval_grid
from . import projection as projection_utils
import tqdm
from torch import nn

# Define optimization class
# TODO: Add initialization option and way to retrieve initialization to check the difference
class OptimizationCloth(nn.Module):
    def __init__(self, n_style_params=12, n_shape_params=12, initialize_style=None):
        super(OptimizationCloth, self).__init__()
        #self.style = nn.Parameter(torch.FloatTensor([[ 0.00298888,  0.02868305, -0.07099994,  0.0119554 , -0.15135743, 0.00622638, -0.21091736,  0.00664981, -0.03769903,  0.10176827, -0.23113497,  0.07247545]]).cuda())

        if type(initialize_style) is type(None):
            self.style = nn.Parameter((torch.rand(1, n_style_params).cuda() - 0.5)/10)
        else:
            self.style = nn.Parameter(torch.FloatTensor(initialize_style).unsqueeze(0).cuda())
        self.shape = nn.Parameter((torch.rand(1, n_shape_params).cuda() - 0.5)/10)

    def forward(self):
        return self.style, self.shape

class OptimizationSMPL(nn.Module):
    def __init__(self, pose, beta, trans):
        super(OptimizationSMPL, self).__init__()
        self.pose_factor = 1.0
        self.beta_factor = 0.3
        self.trans_factor = 0.1

        self.pose = nn.Parameter(torch.FloatTensor(pose).cuda()/self.pose_factor)
        self.beta = nn.Parameter(torch.FloatTensor(beta).cuda()/self.beta_factor)
        self.trans = nn.Parameter(torch.FloatTensor(trans).cuda()/self.trans_factor)

    def forward(self):
        return self.pose*self.pose_factor, self.beta*self.beta_factor, self.trans*self.trans_factor

def to_single_mesh(posed_meshes, colors, all_camscales, all_camtrans, all_toplefts, allscaleratios):
    # TODO: Add colors to trimesh mesh

    from IPython import embed
    embed()

    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3))

    meanscaleimg = np.mean(allscaleratios)
    for index in range(len(posed_meshes)):
        factor_depth_vs_scale = 1.0
        factor_relative_scale = meanscaleimg/allscaleratios[index]
        depth_trans = factor_depth_vs_scale/all_camscales[index]*factor_relative_scale

        trans_2d = all_toplefts[index]/224 + 112/allscaleratios[index]/224
        trans_2d = trans_2d + all_camtrans[index][::-1]
        #trans_2d = all_camtrans[index][::-1] + all_toplefts[index]/112
        #trans_2d = all_camtrans[index][::-1] + (all_toplefts[index] + 224/2/allscaleratios[index])/112
        trans_2d = trans_2d/all_camscales[index]

        translated_vertices = posed_meshes[index].vertices.copy()
        translated_vertices[:, 0] += trans_2d[1] #[1]
        translated_vertices[:, 1] += trans_2d[0] #[0]
        translated_vertices[:, 2] += depth_trans

        faces = np.concatenate((faces, posed_meshes[index].faces + len(vertices)))
        vertices = np.concatenate((vertices, translated_vertices))

    # TODO: Add colors to trimesh mesh
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh


def render_image_projection_multiperson_wrenderer(input_image, posed_meshes, posed_normals, colors, camScale, camTrans, topleft, scale_ratio, mode='rgb', view='cam', renderer=None, done_projection=False):
    #renderer = meshRenderer()
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)

    renderer.setWindowSize(input_image.shape[1], input_image.shape[0])
    renderer.setBackgroundTexture(input_image)
    renderer.setViewportSize(input_image.shape[1], input_image.shape[0])

    # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
    renderer.clear_mesh()
    for mesh_index in range(len(colors)): #TODO: DO THIS FOR EVERYONE IN THE IMAGE:
        if done_projection:
            vertices_2d = posed_meshes[mesh_index].vertices.copy()
            vertices_2d[:,0] -= input_image.shape[1]*0.5
            vertices_2d[:,1] -= input_image.shape[0]*0.5
        else:
            vertices_2d = project_points(posed_meshes[mesh_index].vertices, camScale[mesh_index], camTrans[mesh_index], topleft[mesh_index], scale_ratio[mesh_index], input_image)
            vertices_2d[:,0] -= input_image.shape[1]*0.5
            vertices_2d[:,1] -= input_image.shape[0]*0.5
        if mode == 'normals':
            color = posed_normals[mesh_index]*0.5 + 0.5
            fake_normals = np.zeros_like(posed_normals[mesh_index])
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1

            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
            # TODO: CLEANER RENDER WOULD BE ALL NORMALS THE SAME (POINTING TOWARDS CAM) AND JUST CHANGING VERTEX COLOR ACCORDINGLY?
            #renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces,posed_normals[mesh_index])
        elif mode == 'rgb':
            if len(colors[mesh_index]) > 10:
                renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, normals=posed_normals[mesh_index], color=np.array(colors[mesh_index])[:, :3]/255.0)
            else:
                renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index], np.array(colors[mesh_index])[:3]/255.0)
        elif mode == 'depth':
            fake_normals = np.zeros_like(posed_meshes[mesh_index].vertices)
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1
            color = colors[mesh_index]
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
        else:
            print(mode)
            raise('unknown projection mode')

    if view=='cam':
        renderer.showBackground(True)
    else:
        renderer.showBackground(False)
    renderer.setWorldCenterBySceneCenter()
    renderer.setCameraViewMode(view)

    renderer.display()
    renderImg = renderer.get_screen_color_ibgr()
    return renderImg




























def render_image_projection(input_image, posed_meshes, posed_normals, colors, camScale, camTrans, topleft, scale_ratio, mode='rgb', view='cam'):
    renderer = meshRenderer()
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)

    renderer.setWindowSize(input_image.shape[1], input_image.shape[0])
    renderer.setBackgroundTexture(input_image)
    renderer.setViewportSize(input_image.shape[1], input_image.shape[0])

    # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
    renderer.clear_mesh()
    #from IPython import embed
    #embed()
    for mesh_index in range(len(colors)): #TODO: DO THIS FOR EVERYONE IN THE IMAGE:
        vertices_2d = project_points(posed_meshes[mesh_index].vertices, camScale, camTrans, topleft, scale_ratio, input_image)
        vertices_2d[:,0] -= input_image.shape[1]*0.5
        vertices_2d[:,1] -= input_image.shape[0]*0.5
        if mode == 'normals':
            color = posed_normals[mesh_index]*0.5 + 0.5
            fake_normals = np.zeros_like(posed_normals[mesh_index])
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1

            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
            # TODO: CLEANER RENDER WOULD BE ALL NORMALS THE SAME (POINTING TOWARDS CAM) AND JUST CHANGING VERTEX COLOR ACCORDINGLY?
            #renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces,posed_normals[mesh_index])
        elif mode == 'rgb':
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index], np.array(colors[mesh_index])[:3]/255.0)
        elif mode == 'depth':
            fake_normals = np.zeros_like(posed_meshes[mesh_index].vertices)
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1
            color = colors[mesh_index]
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
        else:
            print(mode)
            raise('unknown projection mode')

        #renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index])#, np.array(colors[mesh_index])[:3]/255.0)

    if view=='cam':
        renderer.showBackground(True)
    else:
        renderer.showBackground(False)
    renderer.setWorldCenterBySceneCenter()
    renderer.setCameraViewMode(view)

    renderer.display()
    renderImg = renderer.get_screen_color_ibgr()
    return renderImg


def render_image_projection_multiperson(input_image, posed_meshes, posed_normals, colors, camScale, camTrans, topleft, scale_ratio, mode='rgb', view='cam'):
    renderer = meshRenderer()
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)

    renderer.setWindowSize(input_image.shape[1], input_image.shape[0])
    renderer.setBackgroundTexture(input_image)
    renderer.setViewportSize(input_image.shape[1], input_image.shape[0])

    # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
    renderer.clear_mesh()
    #from IPython import embed
    #embed()
    for mesh_index in range(len(colors)): #TODO: DO THIS FOR EVERYONE IN THE IMAGE:
        vertices_2d = project_points(posed_meshes[mesh_index].vertices, camScale[mesh_index], camTrans[mesh_index], topleft[mesh_index], scale_ratio[mesh_index], input_image)
        vertices_2d[:,0] -= input_image.shape[1]*0.5
        vertices_2d[:,1] -= input_image.shape[0]*0.5
        if mode == 'normals':
            color = posed_normals[mesh_index]*0.5 + 0.5
            fake_normals = np.zeros_like(posed_normals[mesh_index])
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1

            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
            # TODO: CLEANER RENDER WOULD BE ALL NORMALS THE SAME (POINTING TOWARDS CAM) AND JUST CHANGING VERTEX COLOR ACCORDINGLY?
            #renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces,posed_normals[mesh_index])
        elif mode == 'rgb':
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index], np.array(colors[mesh_index])[:3]/255.0)
        elif mode == 'depth':
            fake_normals = np.zeros_like(posed_meshes[mesh_index].vertices)
            if view == 'cam':
                fake_normals[:, 2] = -1
            elif view == 'side':
                fake_normals[:, 0] = -1
            color = colors[mesh_index]
            renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, fake_normals, color=color)
        else:
            print(mode)
            raise('unknown projection mode')

        #renderer.add_mesh(vertices_2d, posed_meshes[mesh_index].faces, posed_normals[mesh_index])#, np.array(colors[mesh_index])[:3]/255.0)

    if view=='cam':
        renderer.showBackground(True)
    else:
        renderer.showBackground(False)
    renderer.setWorldCenterBySceneCenter()
    renderer.setCameraViewMode(view)

    renderer.display()
    renderImg = renderer.get_screen_color_ibgr()
    return renderImg

def project_points(vertices_3d, camScale, camTrans, topleft, scale_ratio, input_image):
    # 1. SMPL -> 2D bbox
    vertices_2d = projection_utils.convert_smpl_to_bbox(vertices_3d, camScale, camTrans)

    # 2. 2D bbox -> original 2D image
    vertices_2d = projection_utils.convert_bbox_to_oriIm(
        vertices_2d, scale_ratio, topleft, input_image.shape[1], input_image.shape[0])

    return vertices_2d

def project_points_tensor(vertices_3d, camScale, camTrans, topleft, scale_ratio, input_image):
    # TODO: Use camTrans tensor as well to propagate gradient through it
    # 1. SMPL -> 2D bbox
    vertices_2d = projection_utils.convert_smpl_to_bbox_tensor(vertices_3d, camScale, camTrans)

    # 2. 2D bbox -> original 2D image
    vertices_2d = projection_utils.convert_bbox_to_oriIm_tensor(
        vertices_2d, scale_ratio, topleft, input_image.shape[1], input_image.shape[0])
    return vertices_2d

def project_points_tensor2(vertices_3d, camScale, camTrans, topleft, scale_ratio, input_image):
    # TODO: Use camTrans tensor as well to propagate gradient through it
    # 1. SMPL -> 2D bbox
    vertices_2d = projection_utils.convert_smpl_to_bbox_tensor2(vertices_3d, camScale, camTrans)

    # 2. 2D bbox -> original 2D image
    vertices_2d = projection_utils.convert_bbox_to_oriIm_tensor2(
        vertices_2d, scale_ratio, topleft, input_image.shape[1], input_image.shape[0])
    return vertices_2d

def get_distance_to_closest_mask(vertices_2d, positive_indices, step=100):
    indices_x = torch.FloatTensor(positive_indices[0]).cuda()
    indices_y = torch.FloatTensor(positive_indices[1]).cuda()
    iters = len(vertices_2d)//step
    if len(vertices_2d)%step != 0:
        iters += 1
    dists = torch.zeros(len(vertices_2d)).cuda()
    for i in range(iters):
        in_ = vertices_2d[i*step:(i+1)*step]
        dist_x = (in_[None, :, 1] - indices_x[:, None])**2
        dist_y = (in_[None, :, 0] - indices_y[:, None])**2
        #dist_x = (in_[None, :, 0] - indices_x[:, None])**2
        #dist_y = (in_[None, :, 1] - indices_y[:, None])**2
        dist = torch.sqrt((dist_x + dist_y).min(0)[0])
        dists[i*step:(i+1)*step] = dist

    return dists

def remove_outside_vertices(verts_2d, input_image):
    valid = verts_2d[:, 0] > 0
    valid &= verts_2d[:, 1] > 0
    valid &= verts_2d[:, 0] < input_image.shape[1]
    valid &= verts_2d[:, 1] < input_image.shape[0]

    verts_2d = verts_2d[valid]
    return verts_2d, valid

# TODO: FOLLOWING FUNCT. IS REALLY SLOW. 
# TODO HAVE TO FIND RENDERER THAT TAKES DEPTH IMAGE MUCH FASTER FOR WEAK PROJECTION:
def paint_depth_image(vertices_2d_smpl, vertices_3d_smpl, input_image):
    depth_image_smpl = np.zeros(input_image.shape[:2])
    condys = []
    for y in range(input_image.shape[1]):
        cond2 = vertices_2d_smpl[:, 0] == y
        condys.append(cond2)

    for x in tqdm.tqdm(range(input_image.shape[0])):
        cond1 = vertices_2d_smpl[:, 1] == x
        if not cond1.max():
            continue
        for y in range(input_image.shape[1]):
            #cond2 = vertices_2d_smpl[:, 0] == y
            cond2 = condys[y]
            if not cond2.max():
                continue
            indices = np.where(np.logical_and(cond1, cond2))[0]
            if len(indices) == 0:
                continue
            depth = np.min(vertices_3d_smpl[indices, 2])
            depth_image_smpl[x, y] = depth
    return depth_image_smpl

def render_depth_image(mesh_smpl, camScale, camTrans, topleft, scale_ratio, input_image):
    fake_colors = mesh_smpl.vertices[:, 2:3].repeat(3, 1)
    min_depth = fake_colors.min()
    max_depth = fake_colors.max()
    fake_colors = (fake_colors - min_depth)/(max_depth-min_depth)
    depth_image_smpl = render_image_projection(input_image*0, [mesh_smpl], [], [fake_colors], camScale, camTrans, topleft, scale_ratio, mode='depth')[:, :, 0]/255.0
    mask = depth_image_smpl == 0
    depth_image_smpl = depth_image_smpl*(max_depth - min_depth) + min_depth
    depth_image_smpl[mask] = 0
    return depth_image_smpl

def unpose_and_deform_cloth(model_trimesh, pose_from, pose_to, beta, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    iters = len(model_trimesh.vertices)//step
    if len(model_trimesh.vertices)%step != 0:
        iters += 1
    for i in range(iters):
        in_verts = torch.FloatTensor(model_trimesh.vertices[i*step:(i+1)*step])
        verts = SMPL_Layer.unpose_and_deform_cloth(in_verts, pose_from, pose_to, beta.cpu(), J, v)
        model_trimesh.vertices[step*i:step*(i+1)] = verts.cpu().data.numpy()
    return model_trimesh

def unpose_and_deform_cloth_tensor(vertices_tensor, pose_from, pose_to, beta, J, v, SMPL_Layer, step=1000):
    #SMPL_Layer = SMPL_Layer.cpu()
    iters = len(vertices_tensor)//step
    if len(vertices_tensor)%step != 0:
        iters += 1
    posed_verts = []
    for i in range(iters):
        in_verts = vertices_tensor[i*step:(i+1)*step]#.cpu()
        verts = SMPL_Layer.unpose_and_deform_cloth(in_verts, pose_from, pose_to, beta, J, v)
        #verts = SMPL_Layer.unpose_and_deform_cloth(in_verts, pose_from.cpu(), pose_to.cpu(), beta.cpu(), J.cpu(), v.cpu())
        posed_verts.append(verts)
        #vertices_tensor[step*i:step*(i+1)] = verts
    return torch.cat(posed_verts)

def unpose_and_deform_cloth_w_normals(model_trimesh, pose_from, pose_to, beta, J, v, SMPL_Layer, normals, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        in_verts = torch.FloatTensor(model_trimesh.vertices[i*step:(i+1)*step])
        in_normals = torch.FloatTensor(normals[i*step:(i+1)*step])
        verts, out_normals = SMPL_Layer.unpose_and_deform_cloth_w_normals(in_verts, in_normals, pose_from, pose_to, beta.cpu(), J, v)
        model_trimesh.vertices[step*i:step*(i+1)] = verts.cpu().data.numpy()
        normals[step*i:step*(i+1)] = out_normals.cpu().data.numpy()

    normals = normals/np.linalg.norm(normals, axis=1)[:, None]
    return model_trimesh, normals

def unpose_and_deform_cloth_w_normals2(model_trimesh, pose_from, pose_to, beta, J, v, SMPL_Layer, normals, smooth_normals, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        in_verts = torch.FloatTensor(model_trimesh.vertices[i*step:(i+1)*step])
        in_normals = torch.FloatTensor(normals[i*step:(i+1)*step])
        in_smooth_normals = torch.FloatTensor(smooth_normals[i*step:(i+1)*step])

        verts, out_normals = SMPL_Layer.unpose_and_deform_cloth_w_normals2(in_verts, in_normals, pose_from, pose_to, beta.cpu(), J, v, in_smooth_normals)
        model_trimesh.vertices[step*i:step*(i+1)] = verts.cpu().data.numpy()

        normals[step*i:step*(i+1)] = out_normals.cpu().data.numpy()

    normals = normals/np.linalg.norm(normals, axis=1)[:, None]
    return model_trimesh, normals

def batch_posing(model_trimesh, pose, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0))
        # Try consistent too:
        #vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0), neighbors=3)
        model_trimesh.vertices[step*i:step*(i+1)] = deformed_v.cpu().data.numpy()[0]
    SMPL_Layer = SMPL_Layer.cuda()

    return model_trimesh

def batch_posing_w_normals(model_trimesh, normals, pose, J, v, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        in_verts = torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0)
        in_norms = torch.FloatTensor(normals[step*i:step*(i+1)]).unsqueeze(0)
        vertices_smpl, deformed_v, out_normals= SMPL_Layer.deform_clothed_smpl_w_normals(pose, J, v, in_verts, in_norms)
        # Try consistent too:
        #vertices_smpl, deformed_v = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0), neighbors=3)
        model_trimesh.vertices[step*i:step*(i+1)] = deformed_v.cpu().data.numpy()[0]
        normals[step*i:step*(i+1)] = out_normals.cpu().data.numpy()
    SMPL_Layer = SMPL_Layer.cuda()
    normals = normals/np.linalg.norm(normals, axis=1)[:, None]

    return model_trimesh, normals

def batch_posing_consistent(model_trimesh, pose, J, v, normals, SMPL_Layer, step=10000):
    SMPL_Layer = SMPL_Layer.cpu()
    for i in range(len(model_trimesh.vertices)//step + 1):
        _, deformed_v = SMPL_Layer.deform_clothed_smpl_consistent(pose, J, v, torch.FloatTensor(model_trimesh.vertices[step*i:step*(i+1)]).unsqueeze(0), normals[step*i:step*(i+1)])
        model_trimesh.vertices[step*i:step*(i+1)] = deformed_v.cpu().data.numpy()[0]
    SMPL_Layer = SMPL_Layer.cuda()

    return model_trimesh

def join_trimesh_models(full_model, vertices, faces, normals=None, color=None):
    newverts = np.concatenate((full_model.vertices, vertices)) 
    past_normals = full_model.vertex_normals
    past_colors = full_model.visual.vertex_colors
    past_n_verts = len(full_model.vertices)
    newfaces = np.concatenate((full_model.faces, faces+past_n_verts ))
    
    full_model = trimesh.Trimesh(newverts, newfaces, process=False)
    all_normals = full_model.vertex_normals.copy()
    all_normals[:past_n_verts] = past_normals
    if type(normals) != type(None):
        all_normals[past_n_verts:] = normals
    full_model.vertex_normals = all_normals
    if type(color) != type(None):
        new_color = np.array(color).reshape(1, 4).repeat(len(vertices), 0)
        full_model.visual.vertex_colors[past_n_verts:] = new_color

    return full_model



# RENDERER CLASS:

from OpenGL.GLUT import *
from OpenGL.GLU import *
from .shaders.framework import *

from .glRenderer import glRenderer

# from renderer.render_utils import ComputeNormal
_glut_window = None

class meshRenderer(glRenderer):

    def __init__(self, width=1600, height=1200, name='GL Renderer',
                #  program_files=['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs'],
                #  program_files=['renderer/shaders/normal140.fs', 'renderer/shaders/normal140.vs'],
                # program_files=['renderer/shaders/geo140.fs', 'renderer/shaders/geo140.vs'],
                render_mode ="normal",  #color, geo, normal
                color_size=1, ms_rate=1):

        self.render_mode = render_mode
        self.program_files ={}
        self.program_files['color'] = ['/home/ecorona/cvpr22/reconstruction/model/utils/shaders//simple140.fs', '//home/ecorona/cvpr22/reconstruction/model/utils/shaders/simple140.vs']
        self.program_files['normal'] = ['//home/ecorona/cvpr22/reconstruction/model/utils/shaders/normal140.fs', '//home/ecorona/cvpr22/reconstruction/model/utils/shaders/normal140.vs']
        self.program_files['geo'] = ['//home/ecorona/cvpr22/reconstruction/model/utils/shaders/colorgeo140.fs', '//home/ecorona/cvpr22/reconstruction/model/utils/shaders/colorgeo140.vs']

        glRenderer.__init__(self, width, height, name, self.program_files[render_mode], color_size, ms_rate)

    def setRenderMode(self, render_mode):
        """
        Set render mode among ['color', 'normal', 'geo']
        """
        if self.render_mode == render_mode:
            return

        self.render_mode = render_mode
        self.initShaderProgram(self.program_files[render_mode])


    def drawMesh(self):
        if self.vertex_dim is None:
            return
        # self.draw_init()

        glColor3f(1,1,0)
        glUseProgram(self.program)

        mvMat = glGetFloatv(GL_MODELVIEW_MATRIX)
        pMat = glGetFloatv(GL_PROJECTION_MATRIX)
        # mvpMat = pMat*mvMat

        self.model_view_matrix = mvMat
        self.projection_matrix = pMat

        # glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        # glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix)
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix)

        # Handle vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_DOUBLE, GL_FALSE, 0, None)

        if True:#self.meshindex_data:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)           #Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        # glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)
        glDrawElements(GL_TRIANGLES, len(self.meshindex_data), GL_UNSIGNED_INT, None)       #For index array (mesh face data)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

