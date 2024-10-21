'''
Adapted from Occupancy Networks repository: https://github.com/autonomousvision/occupancy_networks/blob/master/scripts/sample_mesh.py
'''

import trimesh
import numpy as np
import os
from os.path import join
from tqdm import tqdm

from implicitmorph.mesh_utils.libmesh import check_mesh_contains
from implicitmorph.utils import cond_mkdir
from implicitmorph.mesh_utils import mesh_utils

POINT_SIZE = 100000             # Size of points.
POINTS_UNIFORM_RATIO = 0.5 # 1. # Ratio of points to sample uniformly in bounding box.
POINTS_PADDING = 0.1            # Additional padding applied to the uniformly sampled points on both sides (in total).
POINTS_SIGMA = 0.0001 #0.01     # Standard deviation of gaussian noise added to points samples on the surfaces.
POINTS_SIGMA_HARD_NEG1 = 0.001
POINTS_SIGMA_HARD_NEG2 = 0.01

# OCCNET
# given a mesh in .off format, they export
# pointcloud.npz: sampled points on surface of mesh; this file they use to calculate the chamfer distance -> centered and normalized pointcloud of object
# points.npz: calculate iou -> samples on surface and uniform in [-1,1] with gt occ -> check for each point if contained in mesh
# mesh.off: the mesh itself again in centered and normalized

# so this file is for bringing the GT pointcloud in the format OccNet used for evaluating the meshes
# we have the unnormalized neurons in pointcloud.npz files
# we also have mesh files but those are not watertight and this is what check_mesh_contains method expects!
# reconstruct watertight GT mesh from pointcloud

def sample_mesh(dataset, out_data_dir):
    # dict: segment_split:pointcloud N x 3
    pointcloud_dict = dataset.coords
    normals_dict = dataset.normals

    for segment_split in tqdm(pointcloud_dict):
        category_id = dataset.classes[segment_split]
        object_dir = join(out_data_dir, category_id, segment_split)
        cond_mkdir(object_dir)
        process_path(object_dir, pointcloud_dict[segment_split], normals_dict[segment_split])


def process_path(out_data_dir, coords, normals):
    # Export various modalities
    export_pointcloud(out_data_dir, coords, normals)
    export_mesh(out_data_dir, coords)
    export_points(out_data_dir)
    export_points_hard_neg(out_data_dir)


def export_pointcloud(out_data_dir, coords, normals):
    print('[INFO] Export pointcloud.')
    filename = join(out_data_dir, 'pointcloud.npz')
    if os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=coords, normals=normals)


def export_points(out_data_dir):
    print('[INFO] Export points.')
    # load watertight mesh that was constructed in export_mesh
    mesh_filename = join(out_data_dir, 'mesh.off')
    mesh = trimesh.load(mesh_filename, process=False)
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % mesh_filename)
        return

    filename = join(out_data_dir, 'points.npz')

    if os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_uniform = int(POINT_SIZE * POINTS_UNIFORM_RATIO)
    n_points_surface = POINT_SIZE - n_points_uniform

    boxsize = 2 + POINTS_PADDING
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += POINTS_SIGMA * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies_uniform = check_mesh_contains(mesh, points_uniform)
    # NOTE since on_surface=0 in our model, we need to negate the occupancies
    occupancies_uniform = ~occupancies_uniform
    occupancies = np.concatenate([occupancies_uniform, np.zeros((len(points_surface)))], axis=0)

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies)


def export_points_hard_neg(out_data_dir):
    print('[INFO] Export hard negative points.')
    # load watertight mesh that was constructed in export_mesh
    mesh_filename = join(out_data_dir, 'mesh.off')
    mesh = trimesh.load(mesh_filename, process=False)
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % mesh_filename)
        return

    filename = join(out_data_dir, 'points_hard_neg.npz')

    if os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_hard_neg = int(POINT_SIZE * POINTS_UNIFORM_RATIO)
    n_points_hard_neg2 = POINT_SIZE - n_points_hard_neg

    points_hard_neg1 = mesh.sample(n_points_hard_neg)
    points_hard_neg1 += POINTS_SIGMA_HARD_NEG1 * np.random.randn(n_points_hard_neg, 3)

    points_hard_neg2 = mesh.sample(n_points_hard_neg2)
    points_hard_neg2 += POINTS_SIGMA_HARD_NEG2 * np.random.randn(n_points_hard_neg2, 3)
    points = np.concatenate([points_hard_neg1, points_hard_neg2], axis=0)

    occupancies = check_mesh_contains(mesh, points)
    # NOTE since on_surface=0 in our model, we need to negate the occupancies
    occupancies = np.array(~occupancies, dtype=int)

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies)


def export_mesh(out_data_dir, coords):
    print('[INFO] Export mesh.')
    filename = join(out_data_dir, 'mesh.off')
    if os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    
    ply_filename = join(out_data_dir, 'mesh.ply')
    mesh_utils.save_points_as_ply(coords, ply_filename)
    obj_filename = join(out_data_dir, 'mesh.obj')
    mesh_utils.reconstruct_mesh_with_splashsurf(ply_filename, obj_filename)
    mesh = trimesh.load(obj_filename)
    mesh = mesh_utils.remove_inner_components(mesh)

    print('Writing mesh: %s' % filename)
    mesh.export(filename)

