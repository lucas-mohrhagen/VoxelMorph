import os
from plyfile import PlyElement, PlyData
import numpy as np
import trimesh

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices


def save_points_as_ply(points, ply_filename):
    # subsample if more than 500k points
    MAX_N_POINTS = 500000
    if len(points) > MAX_N_POINTS:
        rand_inds = np.random.randint(0, len(points), size=MAX_N_POINTS)
        points = points[rand_inds]
    vertices = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = points[:, 0].astype('f4')
    vertices['y'] = points[:, 1].astype('f4')
    vertices['z'] = points[:, 2].astype('f4')

    el = PlyElement.describe(vertices, "vertex")
    ply_data = PlyData([el])
    ply_data.write(ply_filename)


def reconstruct_mesh_with_splashsurf(ply_filename, obj_filename):
     # https://github.com/w1th0utnam3/splashsurf#all-command-line-options
     os.system(
        'splashsurf reconstruct -i ./' + ply_filename + ' --cube-size 0.005 --particle-radius 0.3 --check-mesh=on --normals=on --smoothing-length 0.01 -o ./' + obj_filename)
     

def remove_inner_components(mesh, verbose=False):
    from implicitmorph.mesh_utils.libmesh import check_mesh_contains
    
    components_tmp = mesh.split()
    components_by_size = sorted(components_tmp, key=lambda x: x.vertices.shape[0], reverse=True)
    if verbose:
        print('Found', len(components_by_size), 'connected components')

    to_remove_mask = np.zeros(len(components_by_size), dtype=bool)

    components_vertices_list = [c.vertices for c in components_by_size]

    for i, component in enumerate(components_by_size):
        if to_remove_mask[i]:
            if verbose:
                print("skipped ", i)
            continue
        if verbose:
            print("checking ", i)
        # check if i+1 to n are inside i

        # first check bounding boxes

        # after that check with the intersections

        if len(components_vertices_list[i + 1:]) > 0:
            vertices = np.concatenate(components_vertices_list[i + 1:])
            tmp_inside_values = check_mesh_contains(component, vertices, hash_resolution=2048)
            tmp_mask = []

            idx = 0
            for j in range(i + 1, len(components_by_size)):
                tmp_mask.append(tmp_inside_values[idx:len(components_vertices_list[j]) + idx].any())
                idx += len(components_vertices_list[j])

            to_remove_mask[i + 1:] = np.logical_or(to_remove_mask[i + 1:], tmp_mask)

    outside_components = [c for c, mask in zip(components_by_size, to_remove_mask) if not mask]

    print('Remaining', len(outside_components), 'connected components')

    return trimesh.util.concatenate(outside_components)