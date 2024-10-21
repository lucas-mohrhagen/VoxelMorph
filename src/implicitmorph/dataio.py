import numpy as np
from os.path import join, exists
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import os
import pickle

SCALE = 1
MIN_NORM = 5e-2
MAX_NORM = 1e-1
N_ENCODER_POINTS = 100000

class BaseDataset(Dataset):
    """_summary_

    Parameters
    ----------
    Dataset : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    def __init__(self, on_surface_points, uniform_points, bb_points, perturbed_points, data_augmentation, rank=0, device='cuda:{}') -> None:
        super().__init__()

        self.norm = MIN_NORM
        self.data_augmentation = data_augmentation

        self.bb_points = bb_points
        self.perturbed_points = perturbed_points

        self.encoder_points = N_ENCODER_POINTS
        self.on_surface_points = on_surface_points
        self.off_surface_points = uniform_points
        self.total_samples = self.on_surface_points + self.perturbed_points + self.bb_points + self.off_surface_points

        print('on surface:', self.on_surface_points, 'uniform:', self.off_surface_points, 'bb:', self.bb_points, 'perturbed:', self.perturbed_points, 'total:', self.total_samples, 'encoder:', self.encoder_points)


    def preprocess_shape(self, neuron_segment_split, pointcloud_path, normalization_per_shape, centering):
        """ Load, center and normalize pointcloud

        Parameters
        ----------
        neuron_segment_split : str
            segment id of neuron
        pointcloud_path : str
            path where pointclouds are saved
        normalization_per_shape : bool
            normalize the pointcloud per shape or over the whole dataset
        centering : str
            information on what to center the pointcloud; options are 'mean' and 'soma' and 'soma_xz'

        Returns
        -------
        dict
            number of points in pointcloud, ndarray containing 3D points of pointcloud, ndarry containing surface normals of pointcloud
        """
        # multiply with 1000 since divided by this factor in preprocessing to downscale to float16 to fit into memory
        coords = np.load(join(pointcloud_path, 'coords', f'{neuron_segment_split}.npz'))['data']*1000
        n_points = len(coords)
        normals = np.load(join(pointcloud_path, 'normals', f'{neuron_segment_split}.npz'))['data']

        # Reshape point cloud such that it lies in bounding box of (-1, 1)
        coords = self.center_coords(coords, center_type=centering, neuron_id=neuron_segment_split)
        coords = self.norm_coords(coords, center_type=centering, per_shape=normalization_per_shape)

        return {'n_points':n_points, 'coords':coords, 'normals':normals}


    def load_shape(self, neuron_segment_split, pointcloud_path):
        """ Load pointcloud

        Parameters
        ----------
        neuron_segment_split : str
            segment id of neuron
        pointcloud_path : str
            path where pointclouds are saved

        Returns
        -------
        dict
            number of points in pointcloud, ndarray containing 3D points of pointcloud, ndarry containing surface normals of pointcloud
        """
        pointcloud = np.load(join(pointcloud_path, neuron_segment_split, 'pointcloud.npz'))
        coords = pointcloud['points']
        n_points = len(coords)
        normals = pointcloud['normals']

        return {'n_points':n_points, 'coords':coords, 'normals':normals}


    def center_coords(self, coords, center_type='soma', **kwargs):
        """ Center the coordinates of the pointcloud to the 1) mean, 2) soma or 3) soma while preserving depth information

        Parameters
        ----------
        coords : ndarray N x 3
            3D coordinates of pointcloud
        center_type : str, optional
            information on what to center the pointcloud, by default 'mean'; other options are 'soma' and 'soma_xz'

        Returns
        -------
        ndarray N x 3
            centered pointcloud
        """
        if center_type == 'mean':
            return coords - np.mean(coords, axis=0, keepdims=True)
        else:
            neuron_id = str(kwargs.get('neuron_id'))
            if center_type == 'soma':
                return coords - self.soma_dict[neuron_id]
            elif center_type == 'soma_xz':
                soma = self.soma_dict[neuron_id]
                soma_xz = np.array([soma[0], 0, soma[2]])
                return coords - soma_xz


    def norm_coords(self, coords, center_type='soma', per_shape=False):
        """ Normalize coordinates of pointcloud

        Parameters
        ----------
        coords : ndarray N x 3
            3D coordinates of pointcloud
        center_type : str, optional
            information on what to center the pointcloud, by default 'mean'; other options are 'soma' and 'soma_xz'
            influences the range of coordinates for normalization
        per_shape : bool, optional
            normalize the pointcloud per shape or over the whole dataset, by default True
        scale_only : bool, optional
            normalize the pointcloud by scaling or shifting, by default False

        Returns
        -------
        ndarray N x 3
            3D coordinates of normalized pointcloud
        """
        if per_shape:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = self.normalizing_constants[center_type]['max']
            coord_min = self.normalizing_constants[center_type]['min']

        # soma/mean are shifted to optimally use the full volume
        coords = (coords - coord_min) / (coord_max - coord_min)
        coords -= 0.5
        coords *= 2.
        return coords


    # https://github.com/thangvubk/SoftGroup/blob/11dcbfd74b7660a2b82ac6473af107849c7d545f/softgroup/data/custom.py#L92
    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, rot_y=False, shearing=False, scale=False, prob=1.0):
        import math

        m = torch.eye(3)
        if jitter and torch.rand(1).item() < prob:
            m += torch.rand(3, 3) * 0.1
        if flip and torch.rand(1).item() < prob:
            m[0][0] *= torch.randint(0, 2, (1,)).item() * 2 - 1
            m[2][2] *= torch.randint(0, 2, (1,)).item() * 2 - 1
        if rot and torch.rand(1).item() < prob:
            theta = torch.rand(1).item() * 2 * math.pi
            m = torch.matmul(m, torch.Tensor([[math.cos(theta), 0, math.sin(theta)],[0, 1, 0],
                              [-math.sin(theta), 0, math.cos(theta)]]))
        if rot_y and torch.rand(1).item() < prob:
            # theta [0,1] -> [-0.5, 0.5] -> [-0.1, 0.1] * math.pi
            theta = (torch.rand(1).item()-0.5) * 0.2 * math.pi
            m = torch.matmul(m, torch.Tensor([[math.cos(theta), math.sin(theta), 0],
                            [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]))
        if shearing and torch.rand(1).item() < prob:
            # theta [0,1] -> [-0.5, 0.5] -> [-0.2, 0.2]
            theta = (torch.rand(1).item()-0.5) * 0.4
            m = torch.matmul(m, torch.Tensor([[1, 0, 0],
                            [theta, 1, theta], [0, 0, 1]]))
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.05 * math.pi
            m = torch.matmul(m, torch.Tensor([[math.cos(theta), 0, math.sin(theta)],[0, 1, 0],
                              [-math.sin(theta), 0, math.cos(theta)]]))
        if scale and torch.rand(1).item() < prob:
            scale_factor = torch.FloatTensor(1).uniform_(0.95, 1.05).item()
            xyz = xyz * scale_factor
        return torch.matmul(torch.Tensor(xyz), m).numpy()


    def __len__(self):
        return self.n_files


    def __getsinglepointcloud__(self, neuron_idx):
        local_id = self.local_files[neuron_idx]
        n_points = self.n_points[local_id]

        # on_surface_coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.encoder_points,))
        on_surface_coords = self.coords[local_id][rand_idcs]
        on_surface_coords = torch.from_numpy(on_surface_coords).to(self.device).float()
        return on_surface_coords[None,:]


    def __getwholepointcloud__(self, neuron_idx):
        local_id = self.local_files[neuron_idx]
        on_surface_coords = self.coords[local_id]
        on_surface_coords = torch.from_numpy(on_surface_coords).to(self.device).float()
        return on_surface_coords[None,:]


    def __getitem__(self, idx):
        # each sample item contains points from only one neuron
        neuron_idx = torch.randint(0, len(self.local_ids), (1,), device=self.device).item()
        # random ID from all the ID's in the dataset

        neuron_id = self.local_ids[neuron_idx]
        local_id = self.local_files[neuron_idx]
        n_points = self.n_points[local_id]

        # encoder coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.encoder_points,))
        encoder_points = self.coords[local_id][rand_idcs]
        encoder_points = torch.from_numpy(encoder_points).to(self.device).float()

        # on_surface_coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.on_surface_points,))
        on_surface_coords = self.coords[local_id][rand_idcs]
        if self.data_augmentation:
            on_surface_coords = self.dataAugment(on_surface_coords, jitter=True, flip=True, rot=True, rot_y=True, shearing=True, scale=True, prob=0.3)

        on_surface_normals = self.normals[local_id][rand_idcs]
        on_surface_coords = torch.from_numpy(on_surface_coords).to(self.device).float()
        on_surface_normals = torch.from_numpy(on_surface_normals).to(self.device).float()

        # sample uniform in volume
        off_surface_coords = torch.rand(self.off_surface_points, 3, device=self.device) * 2 - 1

        if self.perturbed_points > 0:
            rand_idcs = torch.randint(0, n_points, (self.perturbed_points,))
            perturbed_off_surface_coords = self.coords[local_id][rand_idcs]
            perturbed_off_surface_normals = self.normals[local_id][rand_idcs]

            m = torch.distributions.log_normal.LogNormal(0.002, SCALE)
            perturbing = m.sample((self.perturbed_points,1)).numpy() * self.norm + 0.001
            perturbed_off_surface_coords = (torch.from_numpy(perturbed_off_surface_coords + perturbing * perturbed_off_surface_normals)).to(self.device).float()

            # off_surface + perturbed_off_surface
            off_surface_coords = torch.cat((off_surface_coords, perturbed_off_surface_coords), dim=0)
        
        if self.bb_points > 0:
            # off_surface_coords out of bb of neuron
            bb_sample_points = self.sample_in_bb(*self.bounding_boxes[local_id], self.bb_points)
            bb_off_surface_coords = bb_sample_points.to(self.device).float()
            off_surface_coords = torch.cat((off_surface_coords, bb_off_surface_coords), dim=0)

        off_surface_normals = torch.ones(self.total_samples-self.on_surface_points, 3, device=self.device) * -1

        occupancy = torch.zeros((self.total_samples, 1), device=self.device)  # on-surface = 0
        occupancy[self.on_surface_points:, :] = 1  # off-surface = 1

        coords = torch.cat((on_surface_coords, off_surface_coords), dim=0)
        normals = torch.cat((on_surface_normals, off_surface_normals), dim=0)

        # shuffle
        perm = torch.randperm(len(occupancy))
        occupancy, coords, normals = occupancy[perm], coords[perm], normals[perm]

        return {'coords': coords, 'pointcloud': encoder_points, 'neuron_id': neuron_id, 'occupancy': occupancy, 'normals': normals}


class OverfitOneShape(BaseDataset):
    """ Dataset for high resolution 3D object data, i.e. Lucy angel.
    """
    def __init__(self, pointcloud_path, pointcloud_name, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape=True, centering='mean', data_augmentation=None, rank=0, device='cuda:{}'):
        super().__init__(on_surface_points=on_surface_points, uniform_points=uniform_points, bb_points=bb_points, perturbed_points=perturbed_points, data_augmentation=data_augmentation)

        self.device = device.format(rank)
        # try:
        #     world_size = dist.get_world_size()
        # except RuntimeError:
        world_size = 1

        self.n_files = 1
        self.local_ids = np.arange(rank, self.n_files, world_size)
        self.local_files = [pointcloud_name] * world_size

        print(rank, self.local_ids)
        print('[INFO] load shape')

        shape = self.preprocess_shape(pointcloud_name, pointcloud_path, normalization_per_shape, centering)

        self.normals = {pointcloud_name:shape['normals']}
        self.coords = {pointcloud_name:shape['coords']}
        self.n_points = {pointcloud_name:shape['n_points']}

        print('[INFO] shape loaded')


class NeuronsDataset(BaseDataset):
    """ Dataset for neuronal morphologies of Microns data.
    """
    def __init__(self, pointcloud_path, all_segment_splits, n_shapes, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape=True, centering='mean', data_augmentation=None, rank=0, device='cuda:{}'):
        super().__init__(on_surface_points=on_surface_points, uniform_points=uniform_points, bb_points=bb_points, perturbed_points=perturbed_points, data_augmentation=data_augmentation)

        self.device = device.format(rank)
        # try:
        #     world_size = dist.get_world_size()
        # except RuntimeError:
        world_size = 1

        self.n_files = n_shapes
        self.local_ids = np.arange(rank, self.n_files, world_size)
        self.local_files = np.array(all_segment_splits)[self.local_ids]

        all_classes = np.array(['0']*self.n_files)
        self.all_labels = np.array(['neuron']*self.n_files)

        self.local_classes = all_classes[self.local_ids]
        self.local_labels = self.all_labels[self.local_ids]

        self.labels = {self.local_files[i]:self.local_labels[i] for i in range(len(self.local_files))}
        self.classes = {self.local_files[i]:self.local_classes[i] for i in range(len(self.local_files))}

        self.preprocessed = exists(join(pointcloud_path, 'preprocessed'))

        with open(join(pointcloud_path, 'idsplit2soma.pkl'), 'rb') as f:
            self.soma_dict = pickle.load(f)

        with open(join(pointcloud_path, 'normalizing_constants.pkl'), 'rb') as f:
            self.normalizing_constants = pickle.load(f)

        print(rank, self.local_ids)


    def bounding_box(self, coords, margin=0.1):
        """ Calculate bounding box of neuron (object)

        Parameters
        ----------
        coords : ndarray
            3D coords of neuron
        margin : float, optional
            margin around bounding box, by default 0.05

        Returns
        -------
        list
            min values in each dimension and max values in each dimension
        """
        mins, maxs = np.min(coords, axis=0), np.max(coords, axis=0)
        mins_m, maxs_m = mins-margin, maxs+margin
        return mins_m, maxs_m


    def sample_in_bb(self, mins, maxs, bb_samples):
        """ Uniformly sample points in bounding box of pointcloud

        Parameters
        ----------
        local_file : ndarray N x 3
            3D coordinates of pointcloud

        Returns
        -------
        ndarray off_surface_samples x 3
            3D coordinates of sampled points
        """
        x = torch.FloatTensor(1, bb_samples).uniform_(mins[0], maxs[0])
        y = torch.FloatTensor(1, bb_samples).uniform_(mins[1], maxs[1])
        z = torch.FloatTensor(1, bb_samples).uniform_(mins[2], maxs[2])
        off_surface_coords_bb = torch.vstack((x,y,z)).T
        return off_surface_coords_bb


class MicronsMinnieFastGPU(NeuronsDataset):
    """ Dataset for neuronal morphologies of Microns data.
    """
    def __init__(self, pointcloud_path, all_segment_splits, n_shapes, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape=True, centering='mean', data_augmentation=None, rank=0, device='cuda:{}'):
        super().__init__(pointcloud_path, all_segment_splits, n_shapes, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape, centering, data_augmentation, rank, device)

        print('[INFO] load all shapes')
        if not self.preprocessed:
            print("Achtung nicht preprocessed")
            all_shapes = [self.preprocess_shape(neuron_segment_split, pointcloud_path, normalization_per_shape, centering) for neuron_segment_split in self.local_files]
            self.coords = {self.local_files[i]:all_shapes[i]['coords'] for i in range(len(self.local_files))}
            self.bounding_boxes = {self.local_files[i]:self.bounding_box(self.coords[self.local_files[i]]) for i in range(len(self.local_files))}
        else:
            print("Im if")
            all_shapes = [np.load(join(pointcloud_path, 'preprocessed', f'{neuron_segment_split}.npz')) for neuron_segment_split in self.local_files]
            print("Post np load")
            self.coords = {self.local_files[i]:all_shapes[i]['coords'] for i in range(len(self.local_files))}
            print("Coords loaded")
            self.bounding_boxes = {self.local_files[i]:all_shapes[i]['bounding_box'] for i in range(len(self.local_files))}
        self.normals = {self.local_files[i]:all_shapes[i]['normals'] for i in range(len(self.local_files))}
        self.n_points = {self.local_files[i]:len(all_shapes[i]['coords']) for i in range(len(self.local_files))}

        print('[INFO] all shapes loaded')


class MicronsMinnie(NeuronsDataset):
    """ Dataset for neuronal morphologies of Microns data.
    """
    def __init__(self, pointcloud_path, all_segment_splits, n_shapes, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape=True, centering='mean', data_augmentation=None, rank=0, device='cuda:{}'):
        super().__init__(pointcloud_path, all_segment_splits, n_shapes, on_surface_points, uniform_points, bb_points, perturbed_points, normalization_per_shape, centering, data_augmentation, rank, device)

        self.pointcloud_path = pointcloud_path
        self.normalization_per_shape = normalization_per_shape
        self.centering = centering


    def __getsinglepointcloud__(self, neuron_idx):
        local_id = self.local_files[neuron_idx]
        if not self.preprocessed:
            shape = self.preprocess_shape(local_id, self.pointcloud_path, self.normalization_per_shape, self.centering)
        else:
            shape = np.load(join(self.pointcloud_path, 'preprocessed', f'{local_id}.npz'))
        coords = shape['coords']
        n_points = len(coords)

        # on_surface_coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.encoder_points,))
        on_surface_coords = coords[rand_idcs]
        on_surface_coords = torch.from_numpy(on_surface_coords).to(self.device).float()
        return on_surface_coords[None,:]


    def __getsingleboundingbox__(self, neuron_idx):
        local_id = self.local_files[neuron_idx]
        if not self.preprocessed:
            neuron = self.preprocess_shape(local_id, self.pointcloud_path, self.normalization_per_shape, self.centering)
            bounding_box = self.bounding_box(neuron['coords'])
        else:
            neuron = np.load(join(self.pointcloud_path, 'preprocessed', f'{local_id}.npz'))
            bounding_box = neuron['bounding_box']
        return bounding_box


    def __getitem__(self, idx):
        # each sample item contains points from only one neuron
        neuron_idx = torch.randint(0, len(self.local_ids), (1,), device=self.device).item()
        neuron_id = self.local_ids[neuron_idx]
        local_id = self.local_files[neuron_idx]

        if not self.preprocessed:
            neuron = self.preprocess_shape(local_id, self.pointcloud_path, self.normalization_per_shape, self.centering)
            # label = self.labels[neuron_segment_split]
            coords = neuron['coords']
            bounding_box = self.bounding_box(coords)
        else:
            neuron = np.load(join(self.pointcloud_path, 'preprocessed', f'{local_id}.npz'))
            # label = neuron['label']
            coords = neuron['coords']
            bounding_box = neuron['bounding_box']

        normals = neuron['normals']
        n_points = len(coords)

        # encoder coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.encoder_points,))
        encoder_points = coords[rand_idcs]
        encoder_points = torch.from_numpy(encoder_points).to(self.device).float()

        # on_surface_coords = random coords of surface points
        rand_idcs = torch.randint(0, n_points, (self.on_surface_points,))
        on_surface_coords = coords[rand_idcs]

        on_surface_normals = normals[rand_idcs]
        on_surface_coords = torch.from_numpy(on_surface_coords).to(self.device).float()
        on_surface_normals = torch.from_numpy(on_surface_normals).to(self.device).float()

        # sample uniform in volume
        off_surface_coords = torch.rand(self.off_surface_points, 3, device=self.device) * 2 - 1

        if self.perturbed_points > 0:
            rand_idcs = torch.randint(0, n_points, (self.perturbed_points,))
            perturbed_off_surface_coords = coords[rand_idcs]
            perturbed_off_surface_normals = normals[rand_idcs]

            m = torch.distributions.log_normal.LogNormal(0.002, SCALE)
            perturbing = m.sample((self.perturbed_points,1)).numpy() * self.norm + 0.001
            perturbed_off_surface_coords = (torch.from_numpy(perturbed_off_surface_coords + perturbing * perturbed_off_surface_normals)).to(self.device).float()

            # off_surface + perturbed_off_surface
            off_surface_coords = torch.cat((off_surface_coords, perturbed_off_surface_coords), dim=0)

        if self.bb_points > 0:
            # off_surface_coords out of bb of neuron
            bb_sample_points = self.sample_in_bb(*bounding_box, self.bb_points)
            bb_off_surface_coords = bb_sample_points.to(self.device).float()
            off_surface_coords = torch.cat((off_surface_coords, bb_off_surface_coords), dim=0)

        off_surface_normals = torch.ones(self.total_samples-self.on_surface_points, 3, device=self.device) * -1

        occupancy = torch.zeros((self.total_samples, 1), device=self.device)  # on-surface = 0
        occupancy[self.on_surface_points:, :] = 1  # off-surface = 1

        coords = torch.cat((on_surface_coords, off_surface_coords), dim=0)
        normals = torch.cat((on_surface_normals, off_surface_normals), dim=0)

        # shuffle
        perm = torch.randperm(len(occupancy))
        occupancy, coords, normals = occupancy[perm], coords[perm], normals[perm]

        return {'coords': coords, 'pointcloud': encoder_points, 'neuron_id': neuron_id, 'occupancy': occupancy, 'normals': normals}


class ShapeNetPointCloudGPU(BaseDataset):
    """ Dataset for ShapeNet
    """
    def __init__(self, pointcloud_path, on_surface_points, uniform_points, bb_points, perturbed_points, per_shape=True, centering='mean', n_files=1, only_class=None, data_augmentation=None, rank=0):
        super().__init__(on_surface_points=on_surface_points, uniform_points=uniform_points, bb_points=bb_points, perturbed_points=perturbed_points, data_augmentation=data_augmentation)

        import re

        # self.device = f'cuda:{rank}'
        # try:
        #     world_size = dist.get_world_size()
        # except RuntimeError:
        world_size = 1
    
        classes = sorted([c for c in os.listdir(pointcloud_path) if re.match(r'^[0-9]{8}$', c)])
        if only_class is not None:
            only_class = [int(i) for i in only_class.split(',')]
            classes = [classes[only_class]] if type(only_class) == int else [classes[oc] for oc in only_class]
        files_per_class = n_files // len(only_class)

        all_files = np.array([f.split('.')[0] for c in classes for f in sorted(os.listdir(join(pointcloud_path, c)))[:files_per_class]])
        all_classes = np.array([c for c in classes for _ in sorted(os.listdir(join(pointcloud_path, c)))[:files_per_class]])

        category_names = {}
        with open(join(pointcloud_path, 'synsetoffset2category.txt')) as f:
            inp = f.read().split('\n')
        for line in inp[:-1]:
            category_names[line.split('\t')[1]] = line.split('\t')[0]
        self.all_labels = np.array([category_names[c] for c in classes for _ in sorted(os.listdir(join(pointcloud_path, c)))[:files_per_class]])

        self.n_files = len(all_files)
        self.local_ids = np.arange(rank, self.n_files, world_size)

        self.local_files = all_files[self.local_ids]
        self.local_classes = all_classes[self.local_ids]
        self.local_labels = self.all_labels[self.local_ids]

        print("[INFO]", rank, self.local_ids)
        print("[INFO] Loading point clouds")

        local_shapes = [self.preprocess_shape(join(pointcloud_path, c, f'{f}.txt'), centering=centering, normalization_per_shape=per_shape) for c,f in zip(self.local_classes, self.local_files)]

        self.normals = {self.local_files[i]:local_shapes[i]['normals'] for i in range(len(self.local_files))}
        self.coords = {self.local_files[i]:local_shapes[i]['coords'] for i in range(len(self.local_files))}
        self.labels = {self.local_files[i]:self.local_labels[i] for i in range(len(self.local_files))}
        self.classes = {self.local_files[i]:self.local_classes[i] for i in range(len(self.local_files))}
        self.n_points = {self.local_files[i]:local_shapes[i]['n_points'] for i in range(len(self.local_files))}

        print('[INFO] all shapes loaded')


    def preprocess_shape(self, filename, centering, normalization_per_shape):
        a = open(filename).read()
        obj = np.array([[float(b) for b in x.split(' ')] for x in a.split('\n') if len(x) > 0])
        coords, normals = obj[:,:3], obj[:, 3:6]
        n_points = len(coords)

        # Reshape point cloud such that it lies in bounding box of (-1, 1)
        coords = self.center_coords(coords, center_type=centering)
        coords = self.norm_coords(coords, center_type=centering, per_shape=normalization_per_shape)

        return {'n_points':n_points, 'coords':coords, 'normals':normals}


class PerturbedSamplingScheduler():
    '''A simple wrapper class for scheduling the norm of perturbed sampling'''

    def __init__(self, dataset, n_epochs, min_norm, max_norm, verbose=False):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.verbose = verbose
        self.n_steps = 0

    def step_and_update_norm(self):
        "Step with the inner optimizer"
        self._update_norm()
        if self.verbose:
            print(f'[INFO] Update norm to {self.dataset.norm}.')

    def _get_norm_scale(self):
        scale = (self.min_norm - self.max_norm) / self.n_epochs
        return scale

    def _update_norm(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        norm = self.max_norm + self.n_steps * self._get_norm_scale()
        self.dataset.norm = norm


class TestDataset(Dataset):
    def __init__(self, n_files, ids, files, category_ids, category_names, data_dir):
        super().__init__()

        self.n_files = n_files     # number of eval meshes
        self.ids = ids             # corresponding to dataset.local_ids
        self.files = files         # corresponding to dataset.local_files
        self.category_ids = category_ids #if category_ids is not None else ['0']*self.n_files
        self.category_names = category_names #if category_names is not None else ['neuron']*self.n_files
        self.data_dir = data_dir   # path to points.npz, pointcloud.npz, mesh.off files

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx):
        points = self.load_points(model_path=join(self.data_dir, self.category_ids[idx], self.files[idx]), filename='points.npz')
        points_hard_neg = self.load_points(model_path=join(self.data_dir, self.category_ids[idx], self.files[idx]), filename='points_hard_neg.npz')
        pointcloud = self.load_pointcloud(model_path=join(self.data_dir, self.category_ids[idx], self.files[idx]), filename='pointcloud.npz')
        return {'idx':idx, 'model':self.files[idx], 'category':self.category_ids[idx], 'category_name':self.category_names[idx],
                'pointcloud_chamfer':pointcloud['points'], 'pointcloud_chamfer_normals':pointcloud['normals'],
                'points_iou':points['points'], 'points_iou_occ':points['occ'], 'points_hard_neg_iou':points_hard_neg['points'], 'points_hard_neg_iou_occ':points_hard_neg['occ']}


    def load_pointcloud(self, model_path, filename):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            filename (str): name of file
        '''
        file_path = join(model_path, filename)
        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            'points': points,
            'normals': normals,
        }
        return data


    def load_points(self, model_path, filename):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            filename (str): name of file
        '''
        file_path = os.path.join(model_path, filename)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        occupancies = occupancies.astype(np.float32)

        data = {
            'points': points,
            'occ': occupancies,
        }
        return data


def check_n_files(opt):
    if opt.n_shapes is None or opt.n_shapes == -1:
        if opt.dataset == 'microns_test' or opt.dataset == 'microns':
            id_list = np.load(join(opt.pointcloud_path, opt.id_list, 'train.npy'), allow_pickle=True)
            opt.n_shapes = len(id_list)
        elif opt.dataset == 'shapenet':
            import re
            from pathlib import Path

            classes = sorted([c for c in os.listdir(opt.pointcloud_path) if re.match(r'^[0-9]{8}$', c)])
            if opt.only_class is not None:
                only_class = [int(i) for i in opt.only_class.split(',')]
                classes = [classes[only_class]] if type(only_class) == int else [classes[oc] for oc in only_class]
            opt.n_shapes = sum([len(list(Path(join(opt.pointcloud_path, c)).glob('*'))) for c in classes])
    return opt


def create_dataset(mode, opt, rank, n_shapes=None, return_id_list=False):
    id_list = np.load(join(opt.pointcloud_path, opt.id_list, f'{mode}.npy'), allow_pickle=True)

    if n_shapes is not None:
        n_shapes = n_shapes
    elif mode == 'train':
        n_shapes = int(opt.n_shapes)
    else:
        n_shapes = len(id_list)

    if opt.dataset == 'microns_test':
        dataset = MicronsMinnieFastGPU(pointcloud_path=opt.pointcloud_path, all_segment_splits=id_list, n_shapes=n_shapes, on_surface_points=int(opt.on_surface_points), uniform_points=int(opt.uniform_points), bb_points=int(opt.bb_points), perturbed_points=int(opt.perturbed_points), normalization_per_shape=opt.normalization_per_shape, centering=opt.centering, data_augmentation=opt.data_augmentation, rank=rank)
    elif opt.dataset == 'microns':
        dataset = MicronsMinnie(pointcloud_path=opt.pointcloud_path, all_segment_splits=id_list, n_shapes=n_shapes, on_surface_points=int(opt.on_surface_points), uniform_points=int(opt.uniform_points), bb_points=int(opt.bb_points), perturbed_points=int(opt.perturbed_points), normalization_per_shape=opt.normalization_per_shape, centering=opt.centering, data_augmentation=opt.data_augmentation, rank=rank)
    elif opt.dataset == 'shapenet':
        dataset = ShapeNetPointCloudGPU(opt.pointcloud_path, on_surface_points=int(opt.on_surface_points), uniform_points=int(opt.uniform_points), bb_points=int(opt.bb_points), perturbed_points=int(opt.perturbed_points), per_shape=opt.normalization_per_shape, centering=opt.centering, n_files=n_shapes, only_class=opt.only_class, data_augmentation=opt.data_augmentation, rank=rank)
    elif opt.dataset == 'overfit1shape':
        dataset = OverfitOneShape(pointcloud_path=opt.pointcloud_path, pointcloud_name=opt.pointcloud_name, on_surface_points=int(opt.on_surface_points), uniform_points=int(opt.uniform_points), bb_points=int(opt.bb_points), perturbed_points=int(opt.perturbed_points), normalization_per_shape=opt.normalization_per_shape, centering=opt.centering, rank=rank)

    if return_id_list:
        return dataset, id_list[:n_shapes]
    else:
        return dataset