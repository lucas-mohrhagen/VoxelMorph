import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

import functools
from collections import OrderedDict
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.pytorch.utils import PointToVoxel

SINE_FACTOR = 30
SIGMA = 1e-3 #1e-4
VAR = 1


class Encoder(nn.Module):
    def __init__(self, encoder_channel, variational=True, w_var=1, w=1):
        super().__init__()

        self.variational = variational
        self.w_var = w_var
        self.w = w
        self.encoder_channel = encoder_channel

        if self.variational:
            self.hidden2mu = nn.Linear(self.encoder_channel, self.encoder_channel)
            self.hidden2logvar = nn.Linear(self.encoder_channel, self.encoder_channel)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def encode(self,x):
       hidden = self.inp2hidden(x)
       mu = self.hidden2mu(hidden)
       log_var = self.hidden2logvar(hidden)
       return mu,log_var

    def forward(self, input):
        if self.variational:
            mu, logvar = self.encode(input)
            hidden = self.reparameterize(mu,logvar) * self.w_var
            hidden_out = {'shape_code':hidden, 'mu':mu, 'logvar':logvar}
        else:
            hidden = self.inp2hidden(input) * self.w
            hidden_out = {'shape_code':hidden}
        return hidden_out
    
class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, kernel_size, indice_key=None):
        super().__init__()

        # residual connection either for unchanged number of channels or for increased number of channels
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        # 2 subsequent conv blocks, RF = 3x3x3 each
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=int((kernel_size-1)/2),
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class LBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, kernel_size, indice_key_id=1):
        super().__init__()
        blocks = {}

        first_block = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, kernel_size, indice_key=None)
            for i in range(block_reps)
        }
        first_block = OrderedDict(first_block)
        first_block = spconv.SparseSequential(first_block)
        blocks[f"block{0}"] = first_block

        for i in range(len(nPlanes) - 1):
            resblocks = [block(nPlanes[i+1], nPlanes[i+1], norm_fn, kernel_size, indice_key=None) for _ in range(block_reps)]

            blocks[f"block{i+1}"] = spconv.SparseSequential(
                norm_fn(nPlanes[i]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[i],
                    nPlanes[i+1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                 ),
                 *resblocks)

        self.blocks = spconv.SparseSequential(OrderedDict(blocks))

    def forward(self, x):
        output = self.blocks(x)
        return output


def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper


class VoxelEncoder(Encoder):

    def __init__(self,# variational=True, w_var=1e-3, w=1e-2,
            variational=False, w_var=1e-3, w=1e-2,
            channels=32, # number of feature channels after first unet contraction. after second and third number of channels is channels * 2 and channels * 3 respectively and so on...
            num_blocks=8, # number of unet contraction blocks
            kernel_size=3, # kernel size (everything except for 3 does not work unfortunately with the newest spconv versions. However on some older ones it worked for me)
            dim_coord=3, # dimensionality of input coordinates (basically always 3, I guess one could remove this from the init haha)
            dim_feat=1, # dimensionality of additional features to provided as input, e.g. rgb values or local geometry features
            fixed_modules=[], # name of modules to freeze
            use_feats=False, # whether to use feats as input features or not. if set to false, features are replaced with dummy values so that the network is trained without feature information
            use_coords=True, # whether to use coords as input features or not. This can basically always be set to false since the relative coord information is implicit in the sparse grid that is convolved over
            spatial_shape=None, # supply spatial_shape manually which is usually automatically calculated in the voxelize() function. You can leave it as None but it might lead to bugs sometimes. Ask me if you wanna know more.
            max_num_points_per_voxel=3, # number of points within a voxel for feature averaging. Once again, ask me if you wanna know what this does but it is not so important
            voxel_size=0.001, # voxel size to be used to transform input point cloud into regular sprase grid
            feature_dim=64,
            ):

        super().__init__(channels, variational, w_var, w)
        self.fixed_modules = fixed_modules
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.voxel_size = voxel_size
        self.todense = spconv.ToDense()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.lnet = LBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size)
        # head
        self.feature_linear = nn.Linear(channels*num_blocks, feature_dim)
        self.init_weights()

        # weight init
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()


    # manually set batchnorms in fixed modules to eval mode
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def transform_input(self, pointclouds):
        bs, N, _ = pointclouds.shape
        coords = pointclouds.reshape(-1,3)
        feats = torch.randn(N*bs,1)
        batch_ids = torch.arange(bs).repeat_interleave(int(N))
        return coords, feats, batch_ids, bs

    def inp2hidden(self, batch):
        coords, feats, batch_ids, batch_size = self.transform_input(batch)
        backbone_output, v2p_map = self.forward_backbone(coords, feats, batch_ids, batch_size)
        output = self.forward_head(backbone_output)
        return output

    @cuda_cast
    def forward_backbone(self, coords, feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, feats]), batch_ids, batch_size, self.voxel_size, self.use_coords, self.use_feats, max_num_points_per_voxel=self.max_num_points_per_voxel)
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        output = self.input_conv(input)
        output = self.lnet(output)
        return output, v2p_map


    def forward_head(self, backbone_output):
        output =  self.todense(backbone_output)
        B, C = output.shape[:2]
        output = output.reshape(B, C, -1).transpose(2, 1)
        output  = torch.mean(output, 1)
        output = self.feature_linear(output)
        return output


def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0
    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size],
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1],
            max_num_voxels=len(feats),
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device)*i, voxel_coord), dim=1)

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
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]


class Decoder(nn.Module):
    '''Siren auto-decoder network.'''

    def __init__(self, out_features=1, in_features=3, shape_dim=64,
                 mode='mlp', hidden_features=256, num_hidden_layers=3,
                 nonlinearity='sine', normalize_embeddings='none', use_amp=False):
        super().__init__()
        self.mode = mode
        self.use_amp = use_amp
        self.normalize_embeddings = normalize_embeddings

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable, special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_uniform_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None)}
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        # Net
        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features+shape_dim, hidden_features), nl
        ))

        for _ in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))

        self.net = nn.Sequential(*self.net)
        if nl_weight_init is not None:
            self.net.apply(nl_weight_init)

        # Apply special initialization to first layer, if applicable.
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)


    def forward(self, model_input):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Enables us to compute gradients w.r.t. coordinates
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org

            shape_embed = model_input['shape_codes']

            if self.normalize_embeddings == 'feature-wise':
                shape_embed = shape_embed - shape_embed.mean(0)
                shape_embed = shape_embed / shape_embed.std(0)

            shape_codes = torch.stack([shape_embed[i].repeat(model_input['coords'].size(1),1) for i in range(model_input['coords'].size(0))])

            if self.mode in ['nerf', 'fourier', 'sincos']:
                coords = self.positional_encoding(coords)

            inp = torch.cat([shape_codes, coords], dim=2)
            output = self.net(inp)

            return {'model_out': output, 'shape_codes':shape_codes}


class EncoderDecoder(nn.Module):
    def __init__(self, out_features=1, in_features=3, shape_dim=64,
                 mode='mlp', hidden_features=256, num_hidden_layers=3,
                 nonlinearity='sine', normalize_embeddings='none', encoder='voxel',
                 k=20, variational=False, use_amp=False):

        super().__init__()

        self.variational = variational
        
        self.encoder = VoxelEncoder(feature_dim=shape_dim, variational=variational)
        self.decoder = Decoder(nonlinearity=nonlinearity, mode=mode, in_features=in_features, out_features=out_features,hidden_features=hidden_features, num_hidden_layers=num_hidden_layers, shape_dim=shape_dim, normalize_embeddings=normalize_embeddings, use_amp=use_amp)


    def forward(self, model_input):
        hidden_out = self.encoder(model_input['pointcloud'])
        model_input['shape_codes'] = hidden_out['shape_code']
        output = self.decoder(model_input)

        if self.variational:
            output['means'] = hidden_out['mu']
            output['logvars'] = hidden_out['logvar']

        return output
            

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(SINE_FACTOR * input)


# INITIALIZATION

def init_weights_normal(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See siren supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / SINE_FACTOR, np.sqrt(6 / num_input) / SINE_FACTOR)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def first_layer_sine_uniform_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.uniform_(-1, 1)


class NeuronDecoder(torch.nn.Module):
    def __init__(self, opt, checkpoint_path):
        super().__init__()

        self.module = EncoderDecoder(nonlinearity=opt.nonlinearity, mode=opt.mode, hidden_features=opt.hidden_features, num_hidden_layers=opt.num_hidden_layers, shape_dim=opt.shape_dim, normalize_embeddings=opt.normalize_embeddings, encoder=opt.encoder, variational=opt.variational, use_amp=opt.use_amp)

        model_state_dict = torch.load(checkpoint_path)['model']
        new_state_dict = {k.replace('module.',''):v for k,v in model_state_dict.items()}
        self.module.load_state_dict(new_state_dict)
        self.module.cuda()

    def forward(self, model_input):
        return self.module(model_input)