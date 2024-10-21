'''
From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import os
from os.path import join
# import torch
# from torch import distributions as dist

from implicitmorph import utils

def create_pointcloud_bb(
    decoder, sdf_dataset, opt, filename, N=256, max_batch=32 ** 3, neuron_id=0, shape_code=None, path=None
):
    """ Create mesh of reconstruction in bounding box of pointcloud

    Parameters
    ----------
    decoder : model
        trained model
    dataset : dataset
    opt : argument parser
        config file
    filename : str
        name of output file
    N : int, optional
        determines the number of samples i.e. the reconstruction quality, by default 256
    max_batch : int, optional
        number of points in batch, by default 64**3
    neuron_id : int, optional
        index for which pointcloud in dataset the mesh is created, by default 0
    """
    if path is None:
        summaries_dir = join(opt.root_path, 'summaries')
    else:
        summaries_dir = path
    utils.cond_mkdir(summaries_dir)

    samples = utils.get_prediction_encdec(model=decoder, dataset=sdf_dataset, N=N, max_batch=max_batch, neuron_id=neuron_id, shape_code=shape_code) if opt.model in 'encoder-decoder' or opt.model == 'occnet' else utils.get_prediction(decoder, sdf_dataset)

    # if opt.model == 'occnet':
    #     p_r = dist.Bernoulli(logits=samples[:,3])
    #     samples[:,3] = p_r.probs
    # else:
    #     samples[:,3] = torch.sigmoid(samples[:,3])

    surface_points = samples[samples[:,3]<=opt.surface_level][:,:3]

    # utils.save_to_xyz(surface, None, summaries_dir, filename)
    num_verts = surface_points.shape[0]
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(surface_points[i, :])
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    ply_data = plyfile.PlyData([el_verts])
    logging.debug("saving mesh to %s" % (filename))
    ply_data.write(join(summaries_dir, f'{filename}.ply'))


def create_gt_pointcloud(sdf_dataset, opt, filename, neuron_id=0, path=None):
    """ Create ground truth pointcloud

    Parameters
    ----------
    dataset : dataset
    opt : argument parser
        config file
    filename : str
        name of output file
    neuron_id : int, optional
        index for which pointcloud in dataset the pointcloud is created, by default 0
    """
    if path is None:
        summaries_dir = join(opt.root_path, 'summaries')
    else:
        summaries_dir = path
    utils.cond_mkdir(summaries_dir)

    segment_split = sdf_dataset.local_files[sdf_dataset.local_ids[neuron_id]]
    print(f'[INFO] reconstructing {segment_split}.')
    coords, normals = sdf_dataset.coords[segment_split], sdf_dataset.normals[segment_split]

    utils.save_to_xyz(coords, normals, summaries_dir, filename)


def create_mesh_bb(
    decoder, sdf_dataset, opt, filename, N=256, max_batch=32 ** 3, offset=None, scale=None, neuron_id=0, neuron_id2=None, shape_code=None, path=None
):
    """ Create mesh of reconstruction in bounding box of pointcloud

    Parameters
    ----------
    decoder : model
        trained model
    dataset : dataset
    opt : argument parser
        config file
    filename : str
        name of output file
    N : int, optional
        determines the number of samples i.e. the reconstruction quality, by default 256
    max_batch : int, optional
        number of points in batch, by default 64**3
    offset : float, optional
        shifts points, by default None
    scale : float, optional
        scales points, by default None
    neuron_id : int, optional
        index for which pointcloud in dataset the mesh is created, by default 0
    """
    if path is None:
        summaries_dir = join(opt.root_path, 'summaries')
    else:
        summaries_dir = path
    utils.cond_mkdir(summaries_dir)
    ply_filename = os.path.join(summaries_dir, filename)

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    mins, maxs = utils.calc_mins_maxs(sdf_dataset, neuron_id, neuron_id2)
    size = maxs - mins
    voxel_size = size / (N - 1)

    samples = utils.get_prediction_encdec(model=decoder, dataset=sdf_dataset, N=N, max_batch=max_batch, neuron_id=neuron_id, neuron_id2=neuron_id2, shape_code=shape_code)[:,3] if opt.model == 'encoder-decoder' or opt.model == 'occnet' else utils.get_prediction(decoder=decoder, dataset=sdf_dataset, N=N, max_batch=max_batch, neuron_id=neuron_id)[:,3]

    # if opt.model == 'occnet':
    #     p_r = dist.Bernoulli(logits=samples)
    #     samples = p_r.probs
    # else:
    #     samples = torch.sigmoid(samples)

    sdf_values = samples.reshape(N, N, N)

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        list(mins),
        list(voxel_size),
        opt.surface_level,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    surface_level,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=surface_level, spacing=voxel_size, method='lewiner'
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    write_ply_file(mesh_points, faces, ply_filename_out)


def write_ply_file(mesh_points, faces, ply_filename_out):
    num_verts = mesh_points.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def refine_mesh(path, filename, percentage_threshold):
    import trimesh

    mesh = trimesh.load_mesh(join(path, f'{filename}.ply'))
    components = mesh.split(only_watertight=False)
    sizes = np.array([comp.vertices.shape[0] for comp in components])
    sorting = sizes.argsort()[::-1]
    sorted_components = components[sorting]
    threshold = mesh.vertices.shape[0] * percentage_threshold

    submesh_components = []
    i, n_vertices = 0, 0

    while n_vertices < threshold:
        comp = sorted_components[i]
        submesh_components.append(comp)
        n_vertices += comp.vertices.shape[0]
        i +=1
    submesh = trimesh.util.concatenate(submesh_components)
    submesh.export(join(path, f'refined_{filename}.ply'))