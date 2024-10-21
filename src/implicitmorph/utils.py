import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from shutil import rmtree
import configargparse
import torch
import wandb

from implicitmorph import dataio, argument_parser, metrics, utils


def get_args():
    """ Create argument parser to parse config file

    Returns
    -------
    config
        parsed config file
    """
    p = configargparse.ArgumentParser()
    p = argument_parser.add_arguments_parser(p)
    opt = p.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(29500+np.random.randint(0,100,1)[0])

    opt.use_amp=argument_parser.t_or_f(opt.use_amp)
    opt.normalization_per_shape = argument_parser.t_or_f(opt.normalization_per_shape)
    opt.curriculum_learning = argument_parser.t_or_f(opt.curriculum_learning)
    opt.should_early_stop = argument_parser.t_or_f(opt.should_early_stop)
    opt.variational = argument_parser.t_or_f(opt.variational)

    opt.bce_loss = argument_parser.t_or_f(opt.bce_loss)
    opt.data_augmentation = argument_parser.t_or_f(opt.data_augmentation)

    opt.root_path = join(opt.logging_root, opt.experiment_name)
    
    if opt.lr_latent is None:
        opt.lr_latent = opt.lr

    return opt


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cond_rmdir(path):
    if os.path.exists(path):
        rmtree(path)


def check_git(config):
    """ Check if git repo clean and save git hash in config

    Parameters
    ----------
    config : config
        config file

    Returns
    -------
    config
        config file with git hash
    """
    import git
    import time
    
    repo = git.Repo(search_parent_directories=True)
    # repo = git.Repo(config['git_repo'])
    sha = repo.head.object.hexsha
    last_commit = time.strftime("%a, %d %b %Y %H:%M", time.gmtime(repo.head.commit.committed_date))
    config.git_hash = sha
    print('Git repo: {}: {}'.format(repo, sha)) 
    print('Last commit: {}'.format(last_commit))
    if repo.is_dirty(): 
        raise NameError('Git repo is not clean.')
    return config


def save_config(config):
    with open(join(config.root_path,'config.ini'), 'w') as f:
        for k,v in sorted(config.__dict__.items()):
            f.write(str(k)+' = '+str(v)+'\n')


def load_config(experiment_folder):
    opt = {}
    with open(join(experiment_folder, 'config.ini'), 'r') as f:
        for line in f.readlines():
            k,v = line.strip('\n').split('=')
            opt[k.strip(' ')] = v.strip(' ')
    
    from argparse import Namespace
    ns = Namespace(**opt)

    return ns


def get_label_df(df_labels_Microns_v7, label, segment_split):
    try:
        label = df_labels_Microns_v7[df_labels_Microns_v7['segment_split'] == segment_split][label].item()
    except ValueError:
        label = 'None'
    return label


def get_labels_microns_minnie_df(segment_splits, segmentsplit2label_file, labels):
    """ Load labels for neurons of Microns dataset for plotting

    Parameters
    ----------
    segment_splits : str
        segment ids and split indices for neurons
    segmentsplit2label_files : pkl file
        pandas dataframe containing segment-split of neuron and its label
    labels : str
        description of the label, i.e. 'cell_types', 'cell_types_ie' (inhibitory/excitatory)

    Returns
    -------
    dict
        with labels
    """
    import pandas as pd

    df_labels_Microns_v7 = pd.read_pickle(segmentsplit2label_file)
    labels_per_neuron = {}

    for label in labels:
        labels_per_neuron[label] = np.array([get_label_df(df_labels_Microns_v7, label, segment_split) for segment_split in segment_splits])
    return labels_per_neuron


def plot_clustering(labels, clustering):
    """ Plot the latent distribution of shape codes (e.g. t-SNE embedding) given labels

    Parameters
    ----------
    labels : dict
        labels for each object
    clustering : ndarray
        t-sne or PCA embedding

    Returns
    -------
    figure
        figure containing plot
    """
    fig = plt.figure(figsize=(15,5))
    u_labels = np.unique(labels)

    colors = plt.cm.tab10(np.arange(len(u_labels))) if len(u_labels) <= 10 else plt.cm.tab20(np.arange(len(u_labels)))
    for label, color in zip(u_labels, colors):
        plt.scatter(clustering[labels == label , 0] , clustering[labels == label , 1] , label = label, color=color)
    plt.legend(bbox_to_anchor=(1,1))
    return fig


def export_embeddings_encdec(model, dataset, id_list):
    embeddings = {}
    neuron_idxs = np.arange(len(id_list))

    for idx, segment_split in zip(neuron_idxs, id_list):
        pointcloud = dataset.__getsinglepointcloud__(idx)
        shape_code = model.module.encoder(pointcloud)
        # shape_code = model.model.encoder(pointcloud)
        embeddings[segment_split] = shape_code

    return embeddings


def get_embeddings_encdec(model, dataset, id_list):
    embeddings = []

    for neuron_id in range(dataset.n_files):
        pointcloud = dataset.__getsinglepointcloud__(neuron_id)
        shape_code = model.module.encoder(pointcloud)['shape_code'].detach().cpu().flatten().numpy()
        # shape_code = model.model.encoder(pointcloud).detach().cpu().flatten().numpy()
        embeddings.append(shape_code)

    embeddings = np.array(embeddings)
    return embeddings


def create_df(id_list, shape_codes, clustering, labels, summaries_dir, exp_name, perplexity):
    import pandas as pd

    out_df = pd.DataFrame()
    out_df['segment_split'] = id_list
    out_df['shape_codes'] = list(shape_codes)
    out_df['tsne'] = list(clustering)
    tsnes = np.array(list(clustering))
    tsne_x, tsne_y = tsnes[:,0], tsnes[:,1]
    out_df['tsne_x'] = tsne_x
    out_df['tsne_y'] = tsne_y

    for cluster_label, values in labels.items():
        out_df[cluster_label] = values

    out_df.to_pickle(join(summaries_dir, f'{exp_name}_{str(perplexity)}_df.pkl'))

    return out_df


def clustering_plot(decoder, dataset, id_list, opt, perplexity = 12, path=None):
    """

    Parameters
    ----------
    decoder : model
        trained model
    id_list : list
        segment ids of neurons
    info_files : str
        path to where label information is saved
    opt : config
        config file
    """
    from sklearn.manifold import TSNE

    decoder.eval()

    shape_codes = get_embeddings_encdec(decoder, dataset, id_list)
    info_files = opt.info_file

    if path is None:
        summaries_dir = join(opt.root_path, 'tsne')
    else:
        summaries_dir = path
    utils.cond_mkdir(summaries_dir)

    clustering = TSNE(n_components=2, perplexity=perplexity).fit_transform(shape_codes)
    exp_name = str(summaries_dir).split('/')[-2]

    if info_files == 'None':
        labels = []
    else:
        labels = get_labels_microns_minnie_df(id_list, info_files, opt.cluster_label.split(','))

    df = create_df(id_list, shape_codes, clustering, labels, summaries_dir, exp_name, perplexity)

    for cluster_label in opt.cluster_label.split(','):
        plot_tsne(df, cluster_label, savepath=join(summaries_dir, f'{exp_name}_{perplexity}_{cluster_label}_tsne.png'))
        if cluster_label in ['cell_type', 'layer']:
            exc_df = df[df['cell_type_coarse'] == 'exc']
            plot_tsne(exc_df, cluster_label, savepath=join(summaries_dir, f'{exp_name}_{perplexity}_{cluster_label}_exc_tsne.png'))


def _calc_mins_maxs(dataset, neuron_id):
    if isinstance(dataset, dataio.OverfitOneShape):
        mins, maxs = np.array([-1, -1, -1]), np.array([1, 1, 1])
    elif isinstance(dataset, dataio.MicronsMinnie):
        mins, maxs = dataset.__getsingleboundingbox__(dataset.local_ids[neuron_id])
    else:
        mins, maxs = dataset.bounding_boxes[dataset.local_files[dataset.local_ids[neuron_id]]]
    return mins, maxs


def calc_mins_maxs(dataset, neuron_id, neuron_id2=None):
    """ Calculate startpoint and sizes for reconstructing an object

    Parameters
    ----------
    dataset : dataset
        dataset
    neuron_id : int
        index of neuron in id list

    Returns
    -------
    ndarray, ndarray
        minimum and range value in each dimension of pointcloud
    """
    mins, maxs = _calc_mins_maxs(dataset, neuron_id)

    if neuron_id2 is not None:
        mins2, maxs2 = _calc_mins_maxs(dataset, neuron_id2)
        mins, maxs = np.min((mins, mins2), axis=0), np.max((maxs, maxs2), axis=0)

    return mins, maxs


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    xv, yv, zv = torch.meshgrid(pxs, pys, pzs)
    xv, yv, zv = xv.flatten(), yv.flatten(), zv.flatten()
    p = torch.stack([xv, yv, zv], dim=1)

    return p


def get_prediction_encdec(model, dataset, samples=None, N=64, max_batch=32 ** 3, neuron_id=0, neuron_id2 = None, shape_code=None):
    """ Get prediction of an object by querying the encoder-decoder model with a 3D grid

    Parameters
    ----------
    model : model
        trained model
    dataset : dataset
        dataset
    samples : Tensor
        sample points to be queried by model
    N : int, optional
        determines the number of samples i.e. the reconstruction quality, by default 256
    max_batch : int, optional
        number of points in batch, by default 64**3
    neuron_id : int, optional
        index for which pointcloud in dataset the mesh is created, by default 0
    shape_code : Tensor
        latent code

    Returns
    -------
    ndarray, N x 4
        array with x,y,z coordinates of grid points and the sdf value at this point
    """
    model.eval()

    # ENCODE
    if shape_code is None:
        pointcloud = dataset.__getsinglepointcloud__(neuron_id)
        shape_code = model.module.encoder(pointcloud)['shape_code']

    # DECODE
    if samples is not None:
        num_samples = len(samples)
        samples = torch.tensor(samples).to(torch.float32)
        samples = torch.cat((samples, torch.zeros(len(samples),1)), dim=1)
    else:
        mins, maxs = calc_mins_maxs(dataset, neuron_id, neuron_id2)
        num_samples = N ** 3
        samples = torch.zeros(num_samples, 4)
        samples[:,:3] = make_3d_grid(mins, maxs, (N,)*3)

    samples.requires_grad = False
    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            model.module.decoder({'coords':sample_subset.view(1, sample_subset.size(0), sample_subset.size(1)), 'shape_codes':shape_code.cuda()})['model_out']
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    return samples


def surface_plot(opt, model, dataset, total_steps, mode, neuron_id=0):
    """ Plot the reconstruction of the neuron (or object) in 2D in tensorboard

    Parameters
    ----------
    decoder : model
        trained model
    dataset : dataset
        dataset
    total_steps : int
        current epoch
    neuron_id : int, optional
        index for which pointcloud in dataset the mesh is created, by default 0
    """

    # plot gt
    neuron_id = dataset.local_ids[neuron_id]
    coords = dataset.__getsinglepointcloud__(neuron_id).squeeze().detach().cpu()
    fig = plt.figure(figsize=(4, 7))
    if isinstance(dataset, dataio.MicronsMinnieFastGPU) or isinstance(dataset, dataio.MicronsMinnie):
        plt.gca().invert_yaxis()
    xs, ys, _ = coords.T
    plt.scatter(xs, ys, c='green', marker='.', s=1)

    # plot prediction
    samples = get_prediction_encdec(model, dataset)

    surface = samples[samples[:,3]<=opt.surface_level]
    xs, ys, _ = surface[:,:3].T
    plt.scatter(xs, ys, c='red', marker='.', s=2)
    plt.axis('off')

    wandb.log({f'neuron reconstruction {mode}': wandb.Image(fig)})
    metrics.write_chamfer_dist(surface, coords, total_steps, mode)

    model.train()


def save_to_xyz(coords, normals, path, filename):
    if normals is not None:
        out = [np.concatenate((point, v_normal)) for point, v_normal in zip(coords, normals)]
    else:
        out = coords
    out = np.array(out)
    f = open(join(path,filename+'.xyz'), 'w')
    for row in out:
        for item in row:
            f.write(str(item))
            f.write(' ')
        f.write('\n')

# source: https://github.com/marissaweis/cluster_neuron/blob/main/neuron_cluster/visualize.py
def plot_tsne(df, name, grey_class=False, colors=None, savepath=None):
    ''' Plot tSNE embedding colored according to <name>.

        Args:
            df: pandas DataFrame with columns `tsne_latent_emb_x`,
                `tsne_latent_emb_y` and `<name>`.
            name: column name refering to column containing the labels
                for coloring (str).
            grey_class: If set to `True`, colors last label class in
                grey (only possible if colors were supplied).
            colors: List of colors to build palette from.
            savepath: If set, figure is saved under this path and
                closed.
    '''
    import seaborn as sns

    unique_labels = sorted(df[name].dropna().unique())
    n = len(unique_labels)

    if colors:
        if grey_class:
            palette = sns.color_palette(colors[:n-1] + ['#C0C0C0'], n_colors=n)
        else:
            palette = sns.color_palette(colors, n_colors=n)
    else:
        palette = 'Paired'

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(
            data=df,
            x='tsne_x',
            y='tsne_y',
            hue=name,
            # style=name,
            hue_order=unique_labels,
            ax=ax,
            palette=palette,
            alpha=0.75,
            linewidth=0,
            legend='full',
            s=50,
        )

    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title(name)
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()

    if savepath is not None:
        ax.set_title('')
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    from pointnet2_ops import pointnet2_utils

    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data