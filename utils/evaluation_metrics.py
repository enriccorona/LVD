import os
import argparse
import glob
import cv2
from utils import cv_utils
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

import time
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os

import pyhull.convex_hull as cvh
import cvxopt as cvx

from manopth import rodrigues_layer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros

import trimesh
import pyquaternion
import utils.plots as plot_utils

from matplotlib import pyplot as plt
import point_cloud_utils as pcu

def graspit_measure(forces, torques, normals):
    G = grasp_matrix(np.array(forces).transpose(), np.array(torques).transpose(), np.array(normals).transpose())
    measure = min_norm_vector_in_facet(G)[0]
    return measure

def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite

    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1] # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T.dot(facet)
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))         # combinations of vertices

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol['x'])
    min_norm = np.sqrt(sol['primal objective'])

    return abs(min_norm), v


def grasp_matrix(forces, torques, normals, soft_fingers=False,
                 finger_radius=0.005, params=None):
    if params is not None and 'finger_radius' in params.keys():
        finger_radius = params.finger_radius
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces
    if soft_fingers:
        num_normals = 2
        if normals.ndim > 1:
            num_normals = 2*normals.shape[1]
        num_cols = num_cols + num_normals

    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3,i] = forces[:,i]
        G[3:,i] = forces[:,i] # ZEROS
        #G[3:,i] = params.torque_scaling * torques[:,i]

    if soft_fingers:
        torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
        pos_normal_i = -num_normals
        neg_normal_i = -num_normals + num_normals / 2
        G[3:,pos_normal_i:neg_normal_i] = torsion
        G[3:,neg_normal_i:] = -torsion

    return G
