from __future__ import print_function
import numpy as np
import socket
from utils import util

pc_name = socket.gethostname()
if pc_name == 'visen3':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.io
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import cv2

colors = {
    # colorbline/print/copy safe:
    'light_red': [0.85882353, 0.74117647, 0.65098039],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
    'light_blue': [.7, .7, .9],
}


def quiver_plot_normals(sdf, save_name='quiver_plot.png'):
    from IPython import embed
    embed()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    resolution = len(sdf)
    xi = np.int32(np.round(np.linspace(0, 1, resolution)*(resolution-1)))
    yi = np.int32(np.round(np.linspace(0, 1, resolution)*(resolution-1)))
    zi = np.int32(np.round(np.linspace(0, 1, resolution)*(resolution-1)))
    xi, yi, zi = np.meshgrid(xi, yi, zi)

    inds = (sdf != 0).all(-1)

    ax.quiver(xi[inds], yi[inds], zi[inds], sdf[inds, 0], sdf[inds, 1], sdf[inds, 2], length=1.2, normalize=True)
    plt.savefig(save_name)
    plt.close()

def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)

def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _create_depth_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    rn = DepthRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn

def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def render_hand_on_img(hand_verts, hand_faces, img, cam_intrinsics):
    # THIS IS BASED ON https://github.com/3d-hand-shape/hand-graph-cnn
    h, w, _ = img.shape

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight


    #cam = [cam_intrinsics[0][0], cam_intrinsics[0][2], cam_intrinsics[1][2]]
    use_cam = ProjectPoints(
            f=[cam_intrinsics[0][0], cam_intrinsics[1][1]], #cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=[cam_intrinsics[0][2], cam_intrinsics[1][2]])

    rn = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)

    rn.background_image = img/255.0 # in 0-1 range
    imtmp = simple_renderer(rn, hand_verts, hand_faces, color=colors['light_red'])

    img_w_hand = (imtmp * 255).astype('uint8')
    return img_w_hand

def render_hand_objects_on_img(hand_verts, hand_faces, obj_verts, obj_faces, img, cam_intrinsics):
    # THIS IS BASED ON https://github.com/3d-hand-shape/hand-graph-cnn
    h, w, _ = img.shape

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    #cam = [cam_intrinsics[0][0], cam_intrinsics[0][2], cam_intrinsics[1][2]]
    use_cam = ProjectPoints(
            f=[cam_intrinsics[0][0], cam_intrinsics[1][1]], #cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=[cam_intrinsics[0][2], cam_intrinsics[1][2]])

    rn_mask_objs = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    rn_mask_hand = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    mask_hand = simple_renderer(rn_mask_objs, hand_verts, hand_faces, color=colors['light_red'])
    mask_objs = simple_renderer(rn_mask_hand, obj_verts, obj_faces, color=colors['light_blue'])

    mask = mask_hand > mask_objs
    mask &= mask_hand != 25

    rn = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    rn_objs = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)

    rn.background_image = img/255.0 # in 0-1 range
    imtmp = simple_renderer(rn, hand_verts, hand_faces, color=colors['light_red'])

    rn_objs.background_image = img/255.0 # in 0-1 range
    imtmp_objs = simple_renderer(rn_objs, obj_verts, obj_faces, color=colors['light_blue'])

    # TODO: RENDER OBJECTS WITH ANOTHER COLOR AND APPLY MASK TO SELECT WHAT IS VISIBLE !!!

    img_w_hand = (imtmp * 255).astype('uint8')
    img_w_objs = (imtmp_objs * 255).astype('uint8')
    img_w_hand[mask] = img_w_objs[mask]


    return img_w_hand

def render_occluded_hand_on_img(hand_verts, hand_faces, obj_verts, obj_faces, img, cam_intrinsics):
    # THIS IS BASED ON https://github.com/3d-hand-shape/hand-graph-cnn
    h, w, _ = img.shape

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    #cam = [cam_intrinsics[0][0], cam_intrinsics[0][2], cam_intrinsics[1][2]]
    use_cam = ProjectPoints(
            f=[cam_intrinsics[0][0], cam_intrinsics[1][1]], #cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=[cam_intrinsics[0][2], cam_intrinsics[1][2]])

    rn_mask_objs = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    rn_mask_hand = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    mask_hand = simple_renderer(rn_mask_objs, hand_verts, hand_faces, color=colors['light_red'])
    mask_objs = simple_renderer(rn_mask_hand, obj_verts, obj_faces, color=colors['light_blue'])

    mask = mask_hand > mask_objs
    mask &= mask_hand != 25

    rn = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)

    rn.background_image = img/255.0 # in 0-1 range
    imtmp = simple_renderer(rn, hand_verts, hand_faces, color=colors['light_red'])

    img_w_hand = (imtmp * 255).astype('uint8')
    img_w_hand[mask] = img[mask]
    return img_w_hand



def video_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces, plane_parameters, path):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    allverts = np.zeros((0,3))
    allfaces = np.zeros((0,3))
    colors = []
    for i in range(len(list_obj_verts)):
        allfaces = np.concatenate((allfaces, list_obj_faces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_verts[i]))
        colors = np.concatenate((colors, ['r']*len(list_obj_faces[i])))
    for i in range(len(list_obj_handverts)):
        allfaces = np.concatenate((allfaces, list_obj_handfaces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_handverts[i]))
        colors = np.concatenate((colors, ['b']*len(list_obj_handfaces[i])))
    allfaces = np.int32(allfaces)
    print(np.max(allfaces))
    print(np.shape(allverts))
    add_group_meshs(ax, allverts, allfaces, alpha=1, c=colors)

    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    frames = []
    for i in range(50):
        ax.view_init(elev=0., azim=i*-2)

        # Change azimuth and get data
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        frames.append(data)

    save_video(np.array(frames), path)
    return

def plot_scene_w_grasps(list_obj_verts, list_obj_faces, list_obj_handverts, list_obj_handfaces, plane_parameters):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
#    for i in range(len(list_obj_verts)):
#        add_mesh(ax, list_obj_verts[i], list_obj_faces[i], alpha=0.4, c='r')
#
#    for i in range(len(list_obj_handverts)):
#        add_mesh(ax, list_obj_handverts[i], list_obj_handfaces[i], alpha=0.4, c='b')


    allverts = np.zeros((0,3))
    allfaces = np.zeros((0,3))
    colors = []
    for i in range(len(list_obj_verts)):
        allfaces = np.concatenate((allfaces, list_obj_faces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_verts[i]))
        colors = np.concatenate((colors, ['r']*len(list_obj_faces[i])))
    for i in range(len(list_obj_handverts)):
        allfaces = np.concatenate((allfaces, list_obj_handfaces[i]+len(allverts)))
        allverts = np.concatenate((allverts, list_obj_handverts[i]))
        colors = np.concatenate((colors, ['b']*len(list_obj_handfaces[i])))
    allfaces = np.int32(allfaces)
    print(np.max(allfaces))
    print(np.shape(allverts))
    add_group_meshs(ax, allverts, allfaces, alpha=1, c=colors)

    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_scene(list_obj_verts, list_obj_faces, plane_parameters):
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111, projection='3d')
    ax.axis('on')
    for i in range(len(list_obj_verts)):
        add_mesh(ax, list_obj_verts[i], list_obj_faces[i], alpha=0.4, c='r')

    #add_group_meshs(ax, np.concatenate((hand_gt[0], obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, np.concatenate(list_obj_verts, 0), flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    # Show plane too:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    step = 0.05
    border = 0.0 #step
    X, Y = np.meshgrid(np.arange(xlim[0]-border, xlim[1]+border, step),
               np.arange(ylim[0]-border, ylim[1]+border, step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
      for c in range(X.shape[1]):
        Z[r, c] = (-plane_parameters[0] * X[r, c] - plane_parameters[1] * Y[r, c] + plane_parameters[3])/plane_parameters[2]
    ax.plot_wireframe(X, Y, Z, color='r')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

def plot_points(verts, switch_axes=False, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    cam_equal_aspect_3d(ax, verts, flip_x=switch_axes, flip_y=switch_axes)
    ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], 'b*')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_pointsandmesh(hand_verts, hand_faces, verts, switch_axes=False, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    add_mesh(ax, hand_verts, hand_faces, alpha=0.4)
#    cam_equal_aspect_3d(ax, hand_verts)
    cam_equal_aspect_3d(ax, hand_verts, flip_x=switch_axes, flip_y=switch_axes)

    ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], 'b*')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_image(image):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('on')
    # Expecting a depth image of size NxN
    # Unnormalize
    image = image*0.16042266926713555

    maxim = image.max() + 0.1
    minim = image[image>0].min()-0.1
    image = image-minim
    image = image*255/(maxim-minim)
    image = image[:, :, np.newaxis].repeat(3, 2)
    #image = image*255.0/maxim
    image[image <= 0] = 255
    image[image > 255] = 255
    image = np.uint8(image)
    #image[image==0] = 255

    ax.imshow(image)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def plot_rgb(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins
    ax.imshow(img)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data



def plot_voxels(voxels):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, edgecolor="k")
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_hand_using_verts(verts, joints, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], 'b*')
    ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], 'r*')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_3d_verts(verts, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], 'b*')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_image_w_projectedjoints(image, joints_gt, joints_pred):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('on')
    ax.imshow(image)

    ax.plot(joints_gt[[0, 1, 2, 3, 4], 0], joints_gt[[0, 1, 2, 3, 4], 1], 'g')
    ax.plot(joints_gt[[0, 5, 6, 7, 8], 0], joints_gt[[0, 5, 6, 7, 8], 1], 'g')
    ax.plot(joints_gt[[0, 9, 10, 11, 12], 0], joints_gt[[0, 9, 10, 11, 12], 1], 'g')
    ax.plot(joints_gt[[0, 13, 14, 15, 16], 0], joints_gt[[0, 13, 14, 15, 16], 1], 'g')
    ax.plot(joints_gt[[0, 17, 18, 19, 20], 0], joints_gt[[0, 17, 18, 19, 20], 1], 'g')

    ax.plot(joints_pred[[0, 1, 2, 3, 4], 0], joints_pred[[0, 1, 2, 3, 4], 1], 'r')
    ax.plot(joints_pred[[0, 5, 6, 7, 8], 0], joints_pred[[0, 5, 6, 7, 8], 1], 'r')
    ax.plot(joints_pred[[0, 9, 10, 11, 12], 0], joints_pred[[0, 9, 10, 11, 12], 1], 'r')
    ax.plot(joints_pred[[0, 13, 14, 15, 16], 0], joints_pred[[0, 13, 14, 15, 16], 1], 'r')
    ax.plot(joints_pred[[0, 17, 18, 19, 20], 0], joints_pred[[0, 17, 18, 19, 20], 1], 'r')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def plot_video_refining_process_wo_objfaces(obj_verts, list_hand_verts, hand_faces):
    frames = []
    for i in range(len(list_hand_verts)):
        fig = plt.figure()
        fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')

        verts = list_hand_verts[i]

        add_mesh(ax, verts, hand_faces, c='b')
        ax.plot3D(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 'r*')
        cam_equal_aspect_3d(ax, verts, flip_x=True, flip_y=True)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(data)

    return np.array(frames)


def plot_hand_w_object_newstyle(obj_verts, obj_faces, hand_verts, hand_faces, flip=True, rot=None):
    w = 640
    h = 480

    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer, DepthRenderer
    from opendr.lighting import LambertianPointLight

    import pyquaternion
    if True:# and rot != None:
        #rot = pyquaternion.Quaternion.random().transformation_matrix[:3, :3]
        #rot = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        #hand_verts = np.matmul(hand_verts, rot)
        #obj_verts = np.matmul(obj_verts, rot)
        #rot = np.array([[1,0,0],[0,0,1],[0,1,0]])
        #hand_verts = np.matmul(hand_verts, rot)
        #obj_verts = np.matmul(obj_verts, rot)
        #rot = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        hand_verts = np.matmul(hand_verts, rot)
        obj_verts = np.matmul(obj_verts, rot)

    hand_verts = hand_verts - obj_verts.mean(0)
    obj_verts = obj_verts - obj_verts.mean(0)

    if flip:
        obj_verts *= -1
        hand_verts *= -1

    use_cam = ProjectPoints(
            f=[1000, 1000], #cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=[0,0,0.6], #np.zeros(3),
            #t=[0,0,0.4], #np.zeros(3),
            k=np.zeros(5),
            c=[w/2,h/2])

    rn_mask_objs = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    rn_mask_hand = _create_depth_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)

    mask_hand = simple_renderer(rn_mask_objs, hand_verts, hand_faces, color=colors['light_red'])
    mask_objs = simple_renderer(rn_mask_hand, obj_verts, obj_faces, color=colors['light_blue'])

    mask = mask_hand > mask_objs
    mask &= mask_hand != 25

    rn = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)
    rn_objs = _create_renderer(w=w, h=h, near=0.1, far=25, rt=use_cam.rt, t=use_cam.t, f=use_cam.f, c=use_cam.c)

    imtmp = simple_renderer(rn, hand_verts, hand_faces, color=colors['light_red'])

    imtmp_objs = simple_renderer(rn_objs, obj_verts, obj_faces, color=colors['light_blue'])

    # TODO: RENDER OBJECTS WITH ANOTHER COLOR AND APPLY MASK TO SELECT WHAT IS VISIBLE !!!

    img = (imtmp * 255).astype('uint8')
    img_w_objs = (imtmp_objs * 255).astype('uint8')
    img[mask] = img_w_objs[mask]

    return img


def plot_hand_w_object(obj_verts, obj_faces, hand_verts, hand_faces, flip=True):
    colors = ['r']*len(hand_faces) + ['b']*len(obj_faces)
    #colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    frames = []
    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    verts = hand_verts
    add_group_meshs(ax, np.concatenate((verts, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, obj_verts, flip_x=flip, flip_y=flip)
    #cam_equal_aspect_3d(ax, obj_verts, flip_x=True, flip_y=True)
    #cam_equal_aspect_3d(ax, verts, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_video_refining_process(obj_verts, obj_faces, list_hand_verts, hand_faces):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    frames = []
    for i in range(len(list_hand_verts)):
        fig = plt.figure()
        fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')

        verts = list_hand_verts[i]
        add_group_meshs(ax, np.concatenate((verts, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
        cam_equal_aspect_3d(ax, verts, flip_x=True, flip_y=True)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(data)

    return np.array(frames)


def plot_hand_classification(depth_image, hand_faces, hand_predicted, hand_gt, title=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131)
    ax.axis('on')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(132, projection='3d')
    ax.axis('on')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins
    add_mesh(ax, hand_gt[0], hand_faces, c='r')
    cam_equal_aspect_3d(ax, hand_gt[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(133, projection='3d')
    ax.axis('on')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins
    add_mesh(ax, hand_predicted[0], hand_faces)
    cam_equal_aspect_3d(ax, hand_predicted[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_hand_classification_and_model(obj_verts, obj_faces, hand_exact_pose, depth_image, hand_faces, hand_predicted, hand_gt, title=None):
    fig = plt.figure(figsize=(15, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(141)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(142, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_gt[0], hand_faces, c='r')
    cam_equal_aspect_3d(ax, hand_gt[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(143, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_predicted[0], hand_faces)
    cam_equal_aspect_3d(ax, hand_predicted[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(144, projection='3d')
    ax.axis('on')

    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)
    add_group_meshs(ax, np.concatenate((hand_exact_pose, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    #add_mesh(ax, hand_exact_pose, hand_faces, alpha=1.0)
    #add_mesh(ax, obj_verts, obj_faces, alpha=1.0, c='r')
#    cam_equal_aspect_3d(ax, hand_verts)
    cam_equal_aspect_3d(ax, np.concatenate((hand_exact_pose, obj_verts)), flip_y=True, flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_hand_classification_with_model_extrapolating_position(obj_verts, obj_faces, hand_exact_pose, depth_image, hand_faces, hand_predicted, hand_gt, title=None):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    fig = plt.figure(figsize=(15, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(141)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(142, projection='3d')
    ax.axis('on')
    hand_gt[0] = hand_gt[0] + hand_exact_pose.mean((0)) - hand_gt[0].mean((0))
    #add_mesh(ax, hand_gt[0], hand_faces, c='r')
    add_group_meshs(ax, np.concatenate((hand_gt[0], obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_gt[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    hand_predicted[0] = hand_predicted[0] + hand_exact_pose.mean((0)) - hand_predicted[0].mean((0))
    ax = fig.add_subplot(143, projection='3d')
    ax.axis('on')
    #add_mesh(ax, hand_predicted[0], hand_faces)
    add_group_meshs(ax, np.concatenate((hand_predicted[0], obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_predicted[0])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(144, projection='3d')
    ax.axis('on')

    add_group_meshs(ax, np.concatenate((hand_exact_pose, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, np.concatenate((hand_exact_pose, obj_verts)), flip_y=True, flip_z=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_only_optimized_hand(depth_image, obj_verts, obj_faces, hand_cluster, hand_pred, hand_ref, hand_faces, title=None):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(151)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(152, projection='3d')
    ax.axis('off')
    add_mesh(ax, hand_cluster, hand_faces, alpha=1, c='b')
    cam_equal_aspect_3d(ax, hand_cluster)

    ax = fig.add_subplot(153, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_pred, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_pred, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(154, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_ref, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_ref, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(155, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_ref, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_ref)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_optimized_hand(depth_image, obj_verts, obj_faces, hand_gt, hand_cluster, hand_pred, hand_ref, hand_faces, title=None):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(161)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(162, projection='3d')
    ax.axis('on')
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)
    add_group_meshs(ax, np.concatenate((hand_gt, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_gt)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(163, projection='3d')
    ax.axis('off')
    add_mesh(ax, hand_cluster, hand_faces, alpha=1, c='b')
    cam_equal_aspect_3d(ax, hand_cluster)

    ax = fig.add_subplot(164, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_pred, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_pred, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(165, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_ref, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_ref, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(166, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_ref, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_ref)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_optimized_hand_wo_objfaces(depth_image, obj_verts, hand_gt, hand_cluster, hand_pred, hand_ref, hand_faces, title=None):

    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(161)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(162, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_gt, hand_faces, c='b')
    ax.plot3D(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 'r*')
    cam_equal_aspect_3d(ax, hand_gt)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(163, projection='3d')
    ax.axis('off')
    add_mesh(ax, hand_cluster, hand_faces, alpha=1, c='b')
    cam_equal_aspect_3d(ax, hand_cluster)

    #print("Im showing refined hand in the last three plots now")
    ax = fig.add_subplot(164, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_pred, hand_faces, c='b')
    #add_mesh(ax, hand_ref, hand_faces, c='b')
    ax.plot3D(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 'r*')
    #cam_equal_aspect_3d(ax, hand_ref, flip_x=True, flip_y=True)
    cam_equal_aspect_3d(ax, hand_pred, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(165, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_ref, hand_faces, c='b')
    ax.plot3D(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 'r*')
    cam_equal_aspect_3d(ax, hand_ref, flip_x=True, flip_y=True)
    #cam_equal_aspect_3d(ax, hand_ref, flip_x=True, flip_y=False)#True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(166, projection='3d')
    ax.axis('on')
    add_mesh(ax, hand_ref, hand_faces, c='b')
    ax.plot3D(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 'r*')
    cam_equal_aspect_3d(ax, hand_ref)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data



def plot_everything(depth_image, obj_verts, obj_faces, hand_gt, hand_classification_task, hand_pred, hand_faces, title=None):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(151)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(152, projection='3d')
    ax.axis('on')
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)
    add_group_meshs(ax, np.concatenate((hand_gt, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_gt)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(153, projection='3d')
    ax.axis('off')
    add_mesh(ax, hand_classification_task, hand_faces, alpha=1, c='b')
    cam_equal_aspect_3d(ax, hand_classification_task)
    #ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(154, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_pred, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_pred, flip_x=True, flip_y=True)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(155, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_pred, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_pred)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_refined_pose(depth_image, obj_verts, obj_faces, hand_gt, hand_pred, hand_faces, title=None):
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)

    fig = plt.figure(figsize=(15, 4))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins
    ax = fig.add_subplot(131)
    ax.axis('off')
    ax.imshow(depth_image[0])

    ax = fig.add_subplot(132, projection='3d')
    ax.axis('on')
    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)
    add_group_meshs(ax, np.concatenate((hand_gt, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_gt)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    ax = fig.add_subplot(133, projection='3d')
    ax.axis('on')
    add_group_meshs(ax, np.concatenate((hand_pred, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    cam_equal_aspect_3d(ax, hand_pred)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def plot_all_3d(obj_verts, obj_faces, hand_verts, hand_faces, title=None, flip_x=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    colors = ['b']*len(hand_faces) + ['r']*len(obj_faces)
    add_group_meshs(ax, np.concatenate((hand_verts, obj_verts)), np.concatenate((hand_faces, obj_faces + 778)), alpha=1, c=colors)
    #add_mesh(ax, hand_verts, hand_faces, alpha=1.0)
    #add_mesh(ax, obj_verts, obj_faces, alpha=1.0, c='r')
#    cam_equal_aspect_3d(ax, hand_verts)
    cam_equal_aspect_3d(ax, np.concatenate((hand_verts, obj_verts)), flip_x=flip_x, flip_y=flip_x)

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_hand_w_joints(hand_verts, hand_faces, joints, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    add_mesh(ax, hand_verts, hand_faces, alpha=0.2, c='b')
    ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], 'r*')
    cam_equal_aspect_3d(ax, hand_verts)

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_hand(hand_verts, hand_faces, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    add_mesh(ax, hand_verts, hand_faces, alpha=0.4)
#    cam_equal_aspect_3d(ax, hand_verts)
    cam_equal_aspect_3d(ax, hand_verts)

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def plot_hand_w_normals(hand_verts, hand_faces, normals, inds_faces_normals, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1.0, 1)  # get rid of margins

    # TODO: PLOT NORMALS
    for i in range(len(inds_faces_normals)):
        p1, p2, p3 = hand_verts[hand_faces[inds_faces_normals[i]]]
        center = (p1 + p2 + p3)/3
        ax.plot([center[0], center[0] + normals[i, 0]], [center[1], center[1] + normals[i, 1]], [center[2], center[2] + normals[i, 2]], 'r-')

    add_mesh(ax, hand_verts, hand_faces)
#    cam_equal_aspect_3d(ax, hand_verts)
    cam_equal_aspect_3d(ax, hand_verts)

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

def add_group_meshs(ax, verts, faces, alpha=0.1, c='b'):
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = []
    for i in range(len(c)):
        if c[i] == 'b':
            face_color.append((141 / 255, 184 / 255, 226 / 255))
        elif c[i] == 'r':
            face_color.append((226 / 255, 184 / 255, 141 / 255))
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)


def cam_equal_aspect_3d(ax, verts, flip_x=False, flip_y=False, flip_z=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    if flip_y:
        ax.set_ylim(centers[1] + r, centers[1] - r)
    else:
        ax.set_ylim(centers[1] - r, centers[1] + r)

    if flip_z:
        ax.set_zlim(centers[2] + r, centers[2] - r)
    else:
        ax.set_zlim(centers[2] - r, centers[2] + r)

def save_video(images, path, freeze_first=50, freeze_last=50, framerate = 5):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(path+'.mp4',fourcc, framerate, (images[0].shape[1], images[0].shape[0]))
    #out = cv2.VideoWriter(path+'.mp4',fourcc, framerate, (images[0].shape[0], images[0].shape[1]))

    # Leave first frame for some time
    for i in range(freeze_first):
        #frame = cv2.flip(images[0],0)
        frame = images[0]
        out.write(frame)

    # Pass through video:
    for i in range(len(images)):
        #frame = cv2.flip(images[i],0)
        frame = images[i]
        out.write(frame)

    # Leave last frame for around a second
    for i in range(freeze_last):
        out.write(frame)

    out.release()
