import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from openTSNE import TSNE
import tqdm
import math
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace
import pickle
import torch.optim as optim
import random
from collections import OrderedDict
import functools
import torcheval.metrics.functional as Feval
from torchvision.ops import sigmoid_focal_loss
from torcheval.metrics.functional import binary_f1_score
from torchmetrics.classification import BinaryJaccardIndex
from torch_receptive_field import receptive_field

from spconv.pytorch.utils import PointToVoxel
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule

from implicitmorph import utils, dataio, argument_parser

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from time import time
import wandb

seed = 1996
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

batch_size = 10

# Training split
id_list_path = '/scratch-emmy/usr/nimpede1/pcs_v7_volume_rotation/splits/train.npy'
# id_list_path = '/scratch-emmy/usr/nimpede1/pcs_v7_volume_rotation/splits/train.npy'
experiment_folder = 'voxel_neuron_morph/logs/test'
id_list = np.load(id_list_path, allow_pickle=True)
opt = utils.load_config(experiment_folder)
opt.use_amp=argument_parser.t_or_f(opt.use_amp)
opt.normalization_per_shape = argument_parser.t_or_f(opt.normalization_per_shape)
opt.curriculum_learning = argument_parser.t_or_f(opt.curriculum_learning)
opt.should_early_stop = argument_parser.t_or_f(opt.should_early_stop)

opt.bce_loss = argument_parser.t_or_f(opt.bce_loss)
opt.occ_boundary_loss = argument_parser.t_or_f(opt.occ_boundary_loss)
opt.inter_loss = argument_parser.t_or_f(opt.inter_loss)
opt.normals_loss = argument_parser.t_or_f(opt.normals_loss)
opt.grad_loss = argument_parser.t_or_f(opt.grad_loss)
opt.l1_loss = argument_parser.t_or_f(opt.l1_loss)
opt.latent_kld_loss = argument_parser.t_or_f(opt.latent_kld_loss)
opt.latent_kld_prior_loss = argument_parser.t_or_f(opt.latent_kld_prior_loss)
opt.latent_norm_loss = argument_parser.t_or_f(opt.latent_norm_loss)

opt.n_shapes = int(opt.n_shapes)
opt.num_epochs = int(opt.num_epochs)
opt.lr = float(opt.lr)
opt.num_hidden_layers = int(opt.num_hidden_layers)
opt.hidden_features = int(opt.hidden_features)
opt.shape_dim = int(opt.shape_dim)
opt.on_surface_points = int(opt.on_surface_points)
opt.resolution = int(opt.resolution)

opt.batch_size = int(batch_size)

dataset_train = dataio.MicronsMinnie(pointcloud_path=opt.pointcloud_path,
                                      all_segment_splits=id_list,
                                      n_shapes=41476,
                                      on_surface_points=int(opt.on_surface_points),
                                      uniform_points=int(opt.uniform_points),
                                      bb_points=int(opt.bb_points),
                                      perturbed_points=int(opt.perturbed_points),
                                      normalization_per_shape=opt.normalization_per_shape,
                                      centering=opt.centering,
                                      rank=0)

# Define the load function with default path
def load_model(state, optimizer, name):
    path = f"/home/nimlucmo/voxel_neuron_morph/{name}.pth"
    checkpoint = torch.load(path)
    state.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['num_iterations']
    state.eval()  # Set the model state to evaluation mode
    return state, optimizer, epoch


def to_dense(voxel_coords):
    voxel_coords = voxel_coords.long()
    dense_shape = torch.max(voxel_coords, dim=0).values + 1
    dense_tensor = torch.zeros(dense_shape.tolist(), dtype=torch.float32, device=DEVICE)
    dense_tensor[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0
    return dense_tensor


def voxelize_in_model(feats, batch_ids, batch_size, voxel_size, use_coords=True, use_feats=False, max_num_points_per_voxel=3, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0

    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]

        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon

        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size],
            coors_range_xyz= min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1],
            max_num_voxels=len(feats),
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=DEVICE)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        # voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=DEVICE)*i, voxel_coord), dim=1)

        # get mean feature of voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])
        
        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord)

    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = torch.cat(voxel_coords, dim=0).max(dim=0).values + 1
    # voxel_coords was originally a tensor of all neurons in a batch, now a list of that length

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]


def transform_input(pointclouds):
    bs, N, _ = pointclouds.shape
    coords = pointclouds.reshape(-1,3).to(DEVICE)
    feats = torch.randn(N*bs,3, device=DEVICE)
    batch_ids = torch.arange(bs).repeat_interleave(int(N)).to(DEVICE)
    return coords, feats, batch_ids, bs

# Data: batch x 100 000 x 3
def voxelize(data, voxel_size):
    coords, feats, batch_ids, batch_size = transform_input(data)
    voxel_feats, voxel_coords, v2p_maps , spatial_shape = voxelize_in_model(torch.hstack([coords, feats]), batch_ids, batch_size, voxel_size)
    return voxel_feats, voxel_coords, v2p_maps , spatial_shape


def set_padding(data, grid_size):
    # Dimensions are Depth x Height x Width

    pad_left = math.floor((grid_size - data.shape[2]) / 2)
    pad_right = math.ceil((grid_size - data.shape[2]) / 2)
    pad_top = math.floor((grid_size - data.shape[1]) / 2)
    pad_bot = math.ceil((grid_size - data.shape[1]) / 2)
    pad_front = math.floor((grid_size - data.shape[0]) / 2)
    pad_back = math.ceil((grid_size - data.shape[0]) / 2)

    return F.pad(data, (pad_left, pad_right, pad_top, pad_bot, pad_front, pad_back))

class ResidualBlock(nn.Module):
    """(Conv3d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        if in_channels == out_channels:
            self.i_branch = nn.Identity()
        else:
            self.i_branch = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        identity = self.i_branch(x)
        x = self.double_conv(x)
        x2 = x + identity
        return x2


class DoubleConv(nn.Module):
    """(Conv3d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, residual=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if residual:
            self.block = ResidualBlock(in_channels, out_channels, kernel_size)
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Downscaling with MaxPool3d then ResidualBlock"""

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, residual=residual)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    """Upscaling with Upsample then ResidualBlock"""

    def __init__(self, in_channels, out_channels, transpose=True, residual=False):
        super().__init__()

        if transpose:
            self.up = self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # Instead of nn.ConvTranspose3d: Better performance Upsample quicker, still weights through Conv3d, BUT only latent decoder does not work
                
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=1, residual=residual)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AE(nn.Module):
    """AE model which takes a 5D tensor as input"""

    def __init__(self, n_channels, n_classes, base_channels=32, num_blocks=4, transpose=True, residual=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # n_channels = n_classes -> input & ouput should have 1 channel dimension
        self.num_blocks = num_blocks
        self.residual = residual

        self.block_channels = [base_channels * (2 ** i) for i in range(num_blocks + 1)]
        # The + 1 at the end is need, because the incoming and outconv layer have to be accounter for as well

        self.incoming = DoubleConv(n_channels, self.block_channels[0], residual=residual)
        
        self.encoders = nn.ModuleList()
        for i in range(num_blocks):
            self.encoders.append(Encoder(self.block_channels[i], self.block_channels[i + 1], residual=residual))

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_blocks, 0, -1):
            self.decoders.append(Decoder(self.block_channels[i], self.block_channels[i - 1], transpose=transpose, residual=residual))
        
        # 1 because occupied or not
        self.outconv = OutConv(self.block_channels[0], n_classes)

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x, part_idxs=None):
        x = self.incoming(x)
        feats = [x]
        
        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            feats.append(x)

        latent_space = x

        if part_idxs is not None:
            latent_parts = [latent_space[:,
                                         :,
                                         part_idx[0]:part_idx[0]+1,
                                         part_idx[1]:part_idx[1]+1,
                                         part_idx[2]:part_idx[2]+1] for part_idx in part_idxs]
            latent_space = torch.cat(latent_parts, dim=0)

        x = latent_space

        # Decoder path
        for decoder in self.decoders:
            x = decoder(x)

        # Last/Output layer
        x = self.outconv(x)

        return x
    
    def get_occupied_indices(self, x):
        # Get the indices of occupied latent cubes
        for _ in range(self.num_blocks):
            x = self.maxpool(x)
        occupied = (x > 0).nonzero(as_tuple=False)[:, 2:]  
        return occupied


def get_random_part_idx(indices):
    # Random neuron ID
    idx = random.randint(0, len(indices) - 1)
    return idx, indices[idx]


def get_unique_part_indices(occupied_indices, latent_dim, num_latent_cubes):
    occupied_indices = occupied_indices.tolist()  # Convert tensor to list for easier manipulation
    if len(occupied_indices) <= 0:
        prob = 0
    else:
        prob = 0.5
    occupied_part_idxs = []
    random_part_idxs = []
    
    if num_latent_cubes == 1:
        if random.random() < prob:
            _, occupied_idx = get_random_part_idx(occupied_indices)
            occupied_part_idxs.append(occupied_idx)
        else:
            random_idx = torch.randint(0, latent_dim, (3,)).tolist()
            random_part_idxs.append(random_idx)
    else:
        num_occupied = min(num_latent_cubes // 2, len(occupied_indices))
        num_random = min(num_latent_cubes - num_occupied, (latent_dim ** 3) - num_occupied)
        # to limit the number of cubes to the maximum langth of the latent cube (in 8x8x8 cube=512)
        
        while len(occupied_part_idxs) < num_occupied:
            idx, individual_occupied_idx = get_random_part_idx(occupied_indices)
            occupied_part_idxs.append(individual_occupied_idx)
            occupied_indices.remove(individual_occupied_idx)
        
        while len(random_part_idxs) < num_random:
            random_idx = torch.randint(0, latent_dim, (3,)).tolist()
            if random_idx not in occupied_part_idxs and random_idx not in random_part_idxs:
                random_part_idxs.append(random_idx)

    combined_list = occupied_part_idxs + random_part_idxs
    return combined_list

def get_embeddings_encdec(model, dataset, id_list, grid_size):
    model.eval()
    embeddings = []

    for neuron_id in range(dataset.n_files):
        pointcloud = dataset.__getsinglepointcloud__(neuron_id)

        _, voxels_list, _ , _ = voxelize(pointcloud, voxel_size=2/grid_size)
        padded_list = []
        # To Dense Step - Goal: Tensor of [batch_size, channel_size=1, depth, height, width]
        for voxel_coords in voxels_list:
            dense_tensor = to_dense(voxel_coords)
            padded_tensor = set_padding(dense_tensor, grid_size)
            padded_list.append(padded_tensor.unsqueeze(0).unsqueeze(0)) # 2x unsqueeze(0) to add dims for batch_size & channels
    
        # concat all together & cast to float (required for model)
        voxels = torch.cat(padded_list, dim=0).float()

        x = model.incoming(voxels)
        for encoder in model.encoders:  # Sequentially pass through each encoder with trained weights
            x = encoder(x)

        # Apply max pooling operation over the channel dimension
        x_pooled = torch.max(x, dim=1, keepdim=True)[0]

        shape_code = x_pooled.detach().cpu().flatten().numpy()

        wandb.log({
            "model_type": num_latent_cubes,
            "neuron_id": neuron_id})

        embeddings.append(shape_code)

    embeddings = np.array(embeddings)
    return embeddings


pointcloud_path = '/scratch-emmy/usr/nimpede1/pcs_v7_volume_rotation'
id_list = 'splits'
mode = 'train'

path = '/home/nimlucmo/voxel_neuron_morph/embeddings/'

id_list = np.load(join(pointcloud_path, id_list, f'{mode}.npy'), allow_pickle=True)
label_df = pd.read_pickle(join(pointcloud_path, 'graphdino_assigned_layer.pkl'))

grid_size = 128
voxel_size = 2 / grid_size

wandb.init(
    project="Embeddings",
    name=f"{grid_size}"
)

print("128Dense")
num_latent_cubes = 0

lmodel = AE(n_channels=1,n_classes=1, base_channels=32, num_blocks=4, transpose=True, residual=True).to(DEVICE)
loptimizer = torch.optim.AdamW(lmodel.parameters(), lr=1e-3)

model_dense128, optim_dense128, n_iter_dense128 = load_model(lmodel, loptimizer, name="model_final/models/50k/128_bs10_nLat0_a0.75g2.2-lr0.001_F1")
print(n_iter_dense128)

out_dir = join(path, 'df_latent_label_128dense.pkl') 
print(out_dir)

latent_embeddings_128dense = get_embeddings_encdec(model_dense128, dataset_train, id_list, grid_size=grid_size)
# for each element in id_list, load neuron run through L-Block, extract latent cube, flatten, add to list

print(latent_embeddings_128dense.shape)

latent_df_dense = pd.DataFrame(data={'latent_emb':list(latent_embeddings_128dense),'segment_split': id_list})
latent_df_dense = latent_df_dense.merge(label_df, on='segment_split')
latent_df_dense.to_pickle(out_dir)

print("128nLat")
num_latent_cubes = 10

lmodel = AE(n_channels=1,n_classes=1, base_channels=32, num_blocks=4, transpose=True, residual=True).to(DEVICE)
loptimizer = torch.optim.AdamW(lmodel.parameters(), lr=1e-3)

model_nLat128, optim_nLat128, n_iter_nLat128 = load_model(lmodel, loptimizer, name="model_final/models/50k/128_bs10_nLat10_a0.75g2.2-lr0.001_F1")
print(n_iter_nLat128)

out_dir = join(path, 'df_latent_label_128nLat.pkl') 
print(out_dir)

latent_embeddings_128nLat = get_embeddings_encdec(model_nLat128, dataset_train, id_list, grid_size=grid_size)

print(latent_embeddings_128nLat.shape)

latent_df_nLat = pd.DataFrame(data={'latent_emb':list(latent_embeddings_128nLat),'segment_split': id_list})
latent_df_nLat = latent_df_nLat.merge(label_df, on='segment_split')
latent_df_nLat.to_pickle(out_dir)
