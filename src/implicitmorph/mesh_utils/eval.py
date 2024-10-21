'''
From Occupancy Networks repository: https://github.com/autonomousvision/occupancy_networks
'''

import numpy as np
from implicitmorph.mesh_utils.libmesh import check_mesh_contains
from implicitmorph.metrics import calc_iou, calc_chamferL1, calc_chamferL2, distance_p2p, calc_normals_correctness


# Maximum values for bounding box [-1, 1]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}


class MeshEvaluator(object):
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, surface_level, n_points=100000):
        self.surface_level = surface_level
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt, points_hard_neg_iou, occ_hard_neg_tgt):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict['iou'] = float(calc_iou(~occ, occ_tgt, self.surface_level))
            occ_hard_neg = check_mesh_contains(mesh, points_hard_neg_iou)
            out_dict['iou_hard_neg'] = float(calc_iou(~occ_hard_neg, occ_hard_neg_tgt, self.surface_level))
        else:
            out_dict['iou'] = 0.
            out_dict['iou_hard_neg'] = 0.

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness = completeness.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy = accuracy.mean()
        accuracy_normals = accuracy_normals.mean()

        completeness2 = completeness**2
        completeness2 = completeness2.mean()
        accuracy2 = accuracy**2
        accuracy2 = accuracy2.mean()

        chamferL1, chamferL2 = calc_chamferL1(completeness, accuracy), calc_chamferL2(completeness, accuracy)
        normals_correctness = calc_normals_correctness(completeness_normals, accuracy_normals)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
        }

        return out_dict

    def eval_points(self, points, occ, occ_tgt):
        ''' Evaluates samples.

        Args:
            occ (numpy_array): occupancy values which should be evaluated for IoU points
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''

        if occ.shape[0] == 0:
            print('Empty points detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            return out_dict
        
        out_dict = self.eval_pointcloud(
            points[occ<=self.surface_level], points[occ_tgt<=self.surface_level])
        out_dict['iou'] = float(calc_iou(occ, occ_tgt, self.surface_level))

        return out_dict