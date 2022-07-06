from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torchvision
import math
import cv2
import torch

def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_str_data(data, path):
    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")



def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size 
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord

def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap 
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic


#def extract_coord_from_output(output, center=True):
#    '''
#    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
#    center: if True, add 0.5, default is true
#    return: shape (batch, jointNum, 3)
#    '''
#    assert(len(output.shape) >= 3)
#    vsize = output.shape[-3:]
#
#    output_rs = output.reshape(-1, np.prod(vsize))
#    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
#    max_index = np.array(max_index).T
#
#    xyz_output = max_index.reshape([*output.shape[:-3], 3])
#
#    # Note discrete coord can represents real range [coord, coord+1), see function scattering() 
#    # So, move coord to range center for better fittness
#    if center: xyz_output = xyz_output + 0.5
#
#    return xyz_output

def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize
    coord = (coord - refpoint) / (cubic_size/2)  # -> [-1, 1]

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2)  # move center to original volume 

    # resize around original volume center
    resize_scale = new_size / 100
    if new_size < 100:
        coord = coord * resize_scale + original_size/2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size/2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,0] -= original_size / 2
        original_coord[:,1] -= original_size / 2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += original_size / 2
        coord[:,1] += original_size / 2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode. 
    coord -= trans

    return coord

def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic


def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

    # use center of cell
    center_offset = 0.5

    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x+center_offset-xi)/std, 2)/2 + \
            np.power((d3output_y+center_offset-yi)/std, 2)/2 + \
            np.power((d3output_z+center_offset-zi)/std, 2)/2))

    return heatmap


#class V2VVoxelization(object):
#    def __init__(self, cubic_size, augmentation=True):
#        self.cubic_size = cubic_size
#        self.cropped_size, self.original_size = 88, 96
#        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
#        self.pool_factor = 2
#        self.std = 1.7
#        self.augmentation = augmentation
#
#        output_size = int(self.cropped_size / self.pool_factor)
#        # Note, range(size) and indexing = 'ij'
#        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')
#
#    def __call__(self, sample):
#        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']
#
#        ## Augmentations
#        # Resize
#        new_size = np.random.rand() * 40 + 80
#
#        # Rotation
#        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi
#
#        # Translation
#        trans = np.random.rand(3) * (self.original_size-self.cropped_size)
#
#        if not self.augmentation:
#            new_size = 100
#            angle = 0
#            trans = self.original_size/2 - self.cropped_size/2
#
#        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
#        heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
#
#        return input.reshape((1, *input.shape)), heatmap
#
#    def voxelize(self, points, refpoint):
#        new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
#        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
#        return input.reshape((1, *input.shape))
#
#    def generate_heatmap(self, keypoints, refpoint):
#        new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
#        heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
#        return heatmap
#
#    def evaluate(self, heatmaps, refpoints):
#        coords = extract_coord_from_output(heatmaps)
#        coords *= self.pool_factor
#        keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
#        return keypoints


