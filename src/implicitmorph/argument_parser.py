
def add_arguments_parser(p):
    """ Add arguments to argument parser

    Parameters
    ----------
    p : configargparse.ArgumentParser
        ArgumentParser

    Returns
    -------
    configargparse.ArgumentParser
        ArgumentParser with added arguments
    """
    p.add('-c', '--config', required=False, is_config_file=True, 
                help='Path to config file.')

    # general
    p.add_argument('--logging_root', type=str, default='./logs', 
                help='root for logging')
    p.add_argument('--root_path', type=str, default=None,
                help='Directory in loggin_root; is created in train.')
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument("--master_port", default=29500, type=int,
                help="Master node (rank 0)'s free port that needs to "
                "be used for communication during distributed "
                "training")
    p.add_argument('--git_hash', default=None, type=str)
    p.add_argument('--seed', default=42, type=int)

    # model
    p.add_argument('--model', type=str, default='sad')
    p.add_argument('--encoder', type=str, default='pointnet')
    p.add_argument('--variational', type=str, default=False,
                   help='Variational encoder; output vector embedding, mu and logvar if True, else vector embedding.')
    p.add_argument('--nonlinearity', type=str, default='sine')
    p.add_argument('--mode', type=str, default='mlp',
                help='Options are "mlp" or "nerf" or "fourier" or "sincos".')
    p.add_argument('--shape_dim', type=int, default=64,
                help='Dimension of the shape embedding.')
    p.add_argument('--normalize_embeddings', type=str, default='none',
                help='Normalization of embeddings. Options are "none" or "feature-wise".')
    p.add_argument('--num_hidden_layers', type=int, default=3)
    p.add_argument('--hidden_features', type=int, default=256)
    

    # dataset
    p.add_argument('--dataset', type=str, default='microns',
                help='Use "microns", "microns_test" or "shapenet" to train on.')
    p.add_argument('--pointcloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
                help='Name of directory where point clouds are saved.')
    p.add_argument('--id_list', type=str, default='labels_splits',
                help='List of neuron ids for training')
    p.add_argument('--pointcloud_name', type=str, default='lucy',
                help='Name of the point cloud.')
    p.add_argument('--preprocessed_path', type=str, default='data',
                help='Name of directly where preprocessed shapes should be saved.')
    p.add_argument('--n_shapes', type=int, default=None, 
                help='The number of shapes the model learns in training.')
    p.add_argument('--test_n_shapes', type=int, default=10,
                help='The number of shapes used for evaluating the meshes after training.')
    p.add_argument('--centering', type=str, default='mean',
                help='Centering method of the shapes. Options are "mean", "soma" and "soma_xz".')
    p.add_argument('--normalization_per_shape', type=str, default=True,
                help='Normalization method of the shapes. True: normalization per shape; False: Normalization over dataset.')
    p.add_argument('--on_surface_points', type=int, default=5000)
    p.add_argument('--uniform_points', type=int, default=5000)
    p.add_argument('--bb_points', type=int, default=2500,
                help='Off surface point sampling in the bounding box of the object.')
    p.add_argument('--perturbed_points', type=int, default=1250,
                help='Hard negative sampling of off surface points using perturbing method.')
    p.add_argument('--curriculum_learning', type=str, default=False,
                help='Hard negative sampling closer to surface with later epochs.')
    p.add_argument('--only_class', type=str, default=None,
                help='For Shapenet dataset; indices of classes to include in training')
    p.add_argument('--data_augmentation', type=str, default=False,
                help='Jitter, rotate, flip, scale on surface points.')

    # training
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4, 
                help='learning rate. default=1e-4')
    p.add_argument('--lr_latent', type=float, default=None, 
                help='learning rate for latent constraint.')
    p.add_argument('--num_epochs', type=int, default=10001,
                help='Number of epochs to train for.')
    p.add_argument('--use_amp', type=str, default="False",
                help='Use Automatic Mixed Precision (AMP), default false.')
    p.add_argument('--epochs_til_ckpt', type=int, default=100,
                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=100,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--resume_from', type=str, default=None, 
                help='Path to pretrained model checkpoint.')
    p.add_argument('--world_size', type=int, default=1,
                help='Number of processes participating in the job.')
    p.add_argument('--loss', type=str, default='loss',
                help='Options are "loss", "occupancy_kld", "occupancy_loss_wo_latent" and "sdf".')
    ## loss constraints
    p.add_argument('--bce_loss', type=str, default="True",
                help='Occupancy loss constraint. Binary cross entropy on occupancy.')
    p.add_argument('--occ_boundary_loss', type=str, default="False",
                help='Occupancy loss constraint. Boundary value constraint on occupancy.')
    p.add_argument('--inter_loss', type=str, default="False",
                help='Occupancy loss constraint. Penalize points off the surface that have an sdf value close to 0.')
    p.add_argument('--normals_loss', type=str, default="False",
                help='Occupancy loss constraint. Gradient of sdf and normal vector should align for points on surface.')
    p.add_argument('--grad_loss', type=str, default="False",
                help='Occupancy loss constraint. Eikonal boundary equation.')
    p.add_argument('--l1_loss', type=str, default="False",
                help='DeepSDF sdf constraint.')
    p.add_argument('--latent_kld_loss', type=str, default="False",
                help='Latent loss constraint. Kullback-Leibler Divergence on latent vectors.')
    p.add_argument('--latent_kld_prior_loss', type=str, default="False",
                help='Latent loss constraint. Kullback-Leibler Divergence on latent vectors.')
    p.add_argument('--latent_norm_loss', type=str, default="False",
                help='Latent loss constraint. Norm on latent vectors.')
    p.add_argument('--latent_norm_siren_loss', type=str, default="False",
                help='Latent loss constraint. Norm on latent vectors.')

    # inference
    p.add_argument('--checkpoint', default=None, 
                help='Checkpoint of trained model.')
    p.add_argument('--info_file', type=str, default=None,
                help='File with label information for the neurons.')
    p.add_argument('--cluster_label', type=str, default='cell_type',
                help='Label for clustering.')
    p.add_argument('--neuron_id', type=str, default='0', 
                help='The ids of the shapes to be reconstructed.')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument("--surface_level", type=float, default=0.075)

    #early stopping
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--should_early_stop", type=str, default="False")
    
    
    return p

def t_or_f(arg):
    """ Turn str in argument parser to boolean

    Parameters
    ----------
    arg : str
        argument of argument parser

    Returns
    -------
    bool
        boolean value of argument
    """
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False