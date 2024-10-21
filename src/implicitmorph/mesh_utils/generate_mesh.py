from tqdm import tqdm
from os.path import join
import os
import trimesh
import numpy as np

from implicitmorph import sdf_meshing, utils


def generate(opt, decoder, dataset, data_dir, point_dir, point_hard_neg_dir, mesh_dir, pointcloud_dir):
    for i in tqdm(range(dataset.n_files)):
        category_id = dataset.local_classes[i]# if dataset.local_classes is not None else '0'
        generate_pointcloud(opt, decoder, dataset, pointcloud_dir, category_id, i)
        generate_mesh(opt, decoder, dataset, mesh_dir, category_id, i)
        convert_mesh_ply2off(dataset, mesh_dir, category_id, i)
        generate_points(opt, decoder, dataset, data_dir, point_dir, category_id, i)
        generate_points_hard_neg(opt, decoder, dataset, data_dir, point_hard_neg_dir, category_id, i)


def generate_pointcloud(opt, decoder, dataset, pointcloud_dir, category_id, idx):
    utils.cond_mkdir(join(pointcloud_dir, category_id))
    filename = join(pointcloud_dir, category_id, f'{dataset.local_files[idx]}.ply')
    if os.path.exists(filename):
        print('Reconstructed pointcloud already exist: %s' % filename)
        return
    sdf_meshing.create_pointcloud_bb(decoder, dataset, opt, filename=dataset.local_files[idx], N=opt.resolution, neuron_id=idx, path=join(pointcloud_dir, category_id))
    print('Writing pointcloud: %s' % filename)
    

def generate_mesh(opt, decoder, dataset, mesh_dir, category_id, idx):
    utils.cond_mkdir(join(mesh_dir, category_id))
    filename = join(mesh_dir, category_id, f'{dataset.local_files[idx]}.ply')
    if os.path.exists(filename):
        print('Reconstructed mesh already exist: %s' % filename)
        return
    sdf_meshing.create_mesh_bb(decoder, dataset, opt, filename=dataset.local_files[idx], N=opt.resolution, neuron_id=idx, path=join(mesh_dir, category_id))
    print('Writing mesh: %s' % filename)


def generate_points(opt, decoder, dataset, data_dir, point_dir, category_id, idx):
    utils.cond_mkdir(join(point_dir, category_id))
    filename = join(point_dir, category_id, f'{dataset.local_files[idx]}.npz')
    if os.path.exists(filename):
        print('Reconstructed points already exist: %s' % filename)
        return
    sample_filename = join(data_dir, category_id, dataset.local_files[idx], 'points.npz')
    samples = np.load(sample_filename)['points']
    points = utils.get_prediction_encdec(decoder, dataset, samples, neuron_id=idx) if opt.model == 'encoder-decoder' or opt.model == 'occnet' else utils.get_prediction(decoder, dataset, samples, neuron_id=idx)
    print('Writing points: %s' % filename)
    np.savez(filename, points=points[:,:3], occupancy=points[:,3])


def generate_points_hard_neg(opt, decoder, dataset, data_dir, point_dir, category_id, idx):
    utils.cond_mkdir(join(point_dir, category_id))
    filename = join(point_dir, category_id, f'{dataset.local_files[idx]}.npz')
    if os.path.exists(filename):
        print('Reconstructed hard negative points already exist: %s' % filename)
        return
    sample_filename = join(data_dir, category_id, dataset.local_files[idx], 'points_hard_neg.npz')
    samples = np.load(sample_filename)['points']
    points = utils.get_prediction_encdec(decoder, dataset, samples, neuron_id=idx) if opt.model == 'encoder-decoder' or opt.model == 'occnet' else utils.get_prediction(decoder, dataset, samples, neuron_id=idx)
    print('Writing hard negative points: %s' % filename)
    np.savez(filename, points=points[:,:3], occupancy=points[:,3])


def convert_mesh_ply2off(dataset, mesh_dir, category_id, idx):
    filename = join(mesh_dir, category_id, f'{dataset.local_files[idx]}.off')
    if os.path.exists(filename):
        print('Reconstructed mesh already exist: %s' % filename)
        return
    ply_filename = join(mesh_dir, category_id, f'{dataset.local_files[idx]}.ply')
    mesh = trimesh.load(ply_filename)
    mesh.export(filename)
    print('Writing mesh: %s' % filename)
