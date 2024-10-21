import torch

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.spatial import KDTree
import seaborn as sns
import wandb

# SCALARS
def _calc_chamfer_dists_with_kdtree(pred_points, gt_points):
    # expects two point arrays as input
    # distances from pred to gt
    gt_kd_tree = KDTree(gt_points)
    distances, _ = gt_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(distances))
    # reverse
    pred_kd_tree = KDTree(pred_points)
    distances, _ = pred_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(distances))

    return pred_to_gt_chamfer + gt_to_pred_chamfer

def _calc_chamfer_dist(pred, gt):
    from extensions.pyTorchChamferDistance.chamfer_distance import ChamferDistance
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(pred, gt)
    dist = (torch.mean(dist1)) + (torch.mean(dist2))
    return dist


def write_chamfer_dist(surface, gt, total_steps, mode, use_kdtree=False):
    pred = surface[:,:3]

    if use_kdtree:
        pred, gt = np.array(pred), np.array(gt)
        chamfer_dist = _calc_chamfer_dists_with_kdtree(pred, gt)
    else:
        pred, gt = pred[None,:], gt[None,:]
        chamfer_dist = _calc_chamfer_dist(pred, gt)

    wandb.log({f"Chamfer distance {mode}": chamfer_dist})

def calc_chamferL1(completeness, accuracy):
    return 0.5 * (completeness + accuracy)

def calc_chamferL2(completeness2, accuracy2):
    return 0.5 * (completeness2 + accuracy2)

def calc_normals_correctness(completeness_normals, accuracy_normals):
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    return normals_correctness

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def calc_iou(pred, gt, surface_level):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        pred (tensor): first set of occupancy values [bs, n_points]
        gt (tensor): second set of occupancy values [bs, n_points]
        surface_level (float): threshold for surface extraction
    '''
    if isinstance(pred, np.ndarray):
        pred = torch.Tensor(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.Tensor(gt)

    # Convert to boolean values
    pred = (pred <= surface_level)
    gt = (gt <= surface_level)
    # Compute IOU
    area_union = torch.logical_or(pred, gt).sum(axis=-1)
    area_intersect = torch.logical_and(pred, gt).sum(axis=-1)
    iou = (area_intersect / area_union)
    return iou

def write_iou(opt, gt, model_output, total_steps, mode):
    bs, n_points, _ = gt['occupancy'].shape
    gt_occ = gt['occupancy'].reshape(bs, -1)
    pred_occ = model_output['model_out'].reshape(bs, -1)

    iou = calc_iou(pred_occ, gt_occ, opt.surface_level).mean()
    wandb.log({f'Intersection over Union (IoU) {mode}': iou})


def _calc_all_basic(pred, gt):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f1 = f1_score(gt, pred)
    balanced_accuracy = balanced_accuracy_score(gt, pred)

    return accuracy, precision, recall, f1, balanced_accuracy


def write_all_basic(opt, gt, model_output, total_steps, mode):
    bs, n_points, _ = gt['occupancy'].shape
    gt = gt_to_numpy_bool_1d(gt['occupancy'].reshape(bs, -1), opt.surface_level)
    pred = pred_to_numpy_bool_1d(model_output['model_out'].reshape(bs, -1), opt.surface_level)

    accuracy, precision, recall, f1, balanced_accuracy = _calc_all_basic(pred, gt)

    wandb.log({f'Accuracy {mode}': accuracy})
    wandb.log({f'Balanced accuracy {mode}': balanced_accuracy})
    wandb.log({f'F1 {mode}': f1})
    wandb.log({f'Precision {mode}': precision})
    wandb.log({f'Recall {mode}': recall})


# FIGURES
def write_distribution_plot(opt, gt, model_output, total_steps):
    bs, n_points, _ = gt['occupancy'].shape
    gt = gt['occupancy'].reshape(bs, -1)
    pred = model_output['model_out'].reshape(bs, -1)

    fig = _calc_distribution_plot(gt, pred, opt.surface_level)
    wandb.log({'Distribution plot': wandb.Image(fig)})


def _calc_distribution_plot(gt, pred, surface_level):
    # pred = torch.sigmoid(pred)
    gt = gt_to_numpy_bool_1d(gt, surface_level)
    pred = np.asarray(pred.cpu().detach().numpy(),dtype='f8').reshape(-1)

    fig = plt.figure(figsize=(4, 4))
    sns.kdeplot(pred[gt], color='blue', label='on_surface')
    sns.kdeplot(pred[~gt], color='orange', label='off_surface')
    plt.legend()
    return fig


# HELPER FUNCTIONS
def pred_to_numpy_bool_1d(pred, surface_level):
    if isinstance(pred,torch.Tensor):
        return (np.asarray(pred.cpu().detach().numpy()) <= surface_level).reshape(-1)
    else:
        return (np.asarray(pred) <= surface_level).reshape(-1)

def gt_to_numpy_bool_1d(gt, surface_level):
    if isinstance(gt,torch.Tensor):
        return (np.asarray(gt.cpu().detach().numpy()) <= surface_level).reshape(-1)
    else:
        return gt