import torch
import torch.nn.functional as F
from lib.loss.aaf import layers as nnx
import numpy as np


def affinity_loss(labels,
                  probs,
                  num_classes,
                  kld_margin):
    """Affinity Field (AFF) loss.

  This function computes AFF loss. There are several components in the
  function:
  1) extracts edges from the ground-truth labels.
  2) extracts ignored pixels and their paired pixels (the neighboring
     pixels on the eight corners).
  3) extracts neighboring pixels on the eight corners from a 3x3 patch.
  4) computes KL-Divergence between center pixels and their neighboring
     pixels from the eight corners.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    probs: A tensor of size [batch_size, height_in, width_in, num_classes],
      indicating segmentation predictions.
    num_classes: A number indicating the total number of valid classes.
    kld_margin: A number indicating the margin for KL-Divergence at edge.

  Returns:
    Two 1-D tensors value indicating the loss at edge and non-edge.
  """
    # Compute ignore map (e.g, label of 255 and their paired pixels).

    labels = torch.squeeze(labels, dim=1)  # NxHxW
    ignore = nnx.ignores_from_label(labels, num_classes, 1)  # NxHxWx8
    not_ignore = np.logical_not(ignore)
    not_ignore = torch.unsqueeze(not_ignore, dim=3)  # NxHxWx1x8

    # Compute edge map.
    one_hot_lab = F.one_hot(labels, depth=num_classes)
    edge = nnx.edges_from_label(one_hot_lab, 1, 255)  # NxHxWxCx8

    # Remove ignored pixels from the edge/non-edge.
    edge = np.logical_and(edge, not_ignore)
    not_edge = np.logical_and(np.logical_not(edge), not_ignore)

    edge_indices = torch.nonzero(torch.reshape(edge, (-1,)))
    not_edge_indices = torch.nonzero(torch.reshape(not_edge, (-1,)))

    # Extract eight corner from the center in a patch as paired pixels.
    probs_paired = nnx.eightcorner_activation(probs, 1)  # NxHxWxCx8
    probs = torch.unsqueeze(probs, dim=-1)  # NxHxWxCx1
    bot_epsilon = 1e-4
    top_epsilon = 1.0

    neg_probs = np.clip(
        1 - probs, bot_epsilon, top_epsilon)
    neg_probs_paired = np.clip(
        1 - probs_paired, bot_epsilon, top_epsilon)
    probs = np.clip(
        probs, bot_epsilon, top_epsilon)
    probs_paired = np.clip(
        probs_paired, bot_epsilon, top_epsilon)

    # Compute KL-Divergence.
    kldiv = probs_paired * torch.log(probs_paired / probs)
    kldiv += neg_probs_paired * torch.log(neg_probs_paired / neg_probs)
    edge_loss = torch.max(0.0, kld_margin - kldiv)
    not_edge_loss = kldiv

    not_edge_loss = torch.reshape(not_edge_loss, (-1,))
    not_edge_loss = torch.gather(not_edge_loss, 0, not_edge_indices)
    edge_loss = torch.reshape(edge_loss, (-1,))
    edge_loss = torch.gather(edge_loss, 0, edge_indices)

    return edge_loss, not_edge_loss

from lib.utils.tools.logger import Logger as Log

def adaptive_affinity_loss(labels,
                           one_hot_lab,
                           probs,
                           size,
                           num_classes,
                           kld_margin,
                           w_edge,
                           w_not_edge,
                           ignore_index=-1):
    """Adaptive affinity field (AAF) loss.

  This function computes AAF loss. There are three components in the function:
  1) extracts edges from the ground-truth labels.
  2) extracts ignored pixels and their paired pixels (usually the eight corner
     pixels).
  3) extracts eight corner pixels/predictions from the center in a
     (2*size+1)x(2*size+1) patch
  4) computes KL-Divergence between center pixels and their paired pixels (the 
     eight corner).
  5) imposes adaptive weightings on the loss.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    one_hot_lab: A tensor of size [batch_size, num_classes, height_in, width_in]
      which is the ground-truth labels in the form of one-hot vector.
    probs: A tensor of size [batch_size, num_classes, height_in, width_in],
      indicating segmentation predictions.
    size: A number indicating the half size of a patch.
    num_classes: A number indicating the total number of valid classes. The 
    kld_margin: A number indicating the margin for KL-Divergence at edge.
    w_edge: A number indicating the weighting for KL-Divergence at edge.
    w_not_edge: A number indicating the weighting for KL-Divergence at non-edge.
    ignore_index: ignore index

  Returns:
    Two 1-D tensors value indicating the loss at edge and non-edge.
  """
    # Compute ignore map (e.g, label of 255 and their paired pixels).
    labels = torch.squeeze(labels, dim=1)  # NxHxW
    ignore = nnx.ignores_from_label(labels, num_classes, size, ignore_index)  # NxHxWx8
    not_ignore = ~ignore
    not_ignore = torch.unsqueeze(not_ignore, dim=3)  # NxHxWx1x8

    # Compute edge map.
    edge = nnx.edges_from_label(one_hot_lab, size, ignore_index)  # NxHxWxCx8

    # Log.info('{} {}'.format(edge.shape, not_ignore.shape))

    # Remove ignored pixels from the edge/non-edge.
    edge = edge & not_ignore


    not_edge = ~edge & not_ignore

    edge_indices = torch.nonzero(torch.reshape(edge, (-1,)))
    # print(edge_indices.size())
    if edge_indices.size()[0] == 0:
        edge_loss = torch.tensor(0.0, requires_grad=False).cuda()
        not_edge_loss = torch.tensor(0.0, requires_grad=False).cuda()
        return edge_loss, not_edge_loss

    not_edge_indices = torch.nonzero(torch.reshape(not_edge, (-1,)))

    # Extract eight corner from the center in a patch as paired pixels.
    probs_paired = nnx.eightcorner_activation(probs, size)  # NxHxWxCx8
    probs = torch.unsqueeze(probs, dim=-1)  # NxHxWxCx1
    bot_epsilon = torch.tensor(1e-4, requires_grad=False).cuda()
    top_epsilon = torch.tensor(1.0, requires_grad=False).cuda()

    neg_probs = torch.where(1 - probs < bot_epsilon, bot_epsilon, 1 - probs)
    neg_probs = torch.where(neg_probs > top_epsilon, top_epsilon, neg_probs)

    neg_probs_paired = torch.where(1 - probs_paired < bot_epsilon, bot_epsilon, 1 - probs_paired)
    neg_probs_paired = torch.where(neg_probs_paired > top_epsilon, top_epsilon, neg_probs_paired)

    probs = torch.where(probs < bot_epsilon, bot_epsilon, probs)
    probs = torch.where(probs > top_epsilon, top_epsilon, probs)

    probs_paired = torch.where(probs_paired < bot_epsilon, bot_epsilon, probs_paired)
    probs_paired = torch.where(probs_paired > top_epsilon, top_epsilon, probs_paired)

    # neg_probs = np.clip(
    #     1-probs, bot_epsilon, top_epsilon)
    # neg_probs_paired = np.clip(
    #     1-probs_paired, bot_epsilon, top_epsilon)
    # probs = np.clip(
    #     probs, bot_epsilon, top_epsilon)
    # probs_paired = np.clip(
    #   probs_paired, bot_epsilon, top_epsilon)

    # Compute KL-Divergence.
    kldiv = probs_paired * torch.log(probs_paired / probs)
    kldiv += neg_probs_paired * torch.log(neg_probs_paired / neg_probs)
    edge_loss = torch.max(torch.tensor(0.0, requires_grad=False).cuda(), kld_margin - kldiv)
    not_edge_loss = kldiv

    # Impose weights on edge/non-edge losses.
    one_hot_lab = torch.unsqueeze(one_hot_lab, dim=-1)

    w_edge = torch.sum(w_edge * one_hot_lab.float(), dim=3, keepdim=True)  # NxHxWx1x1
    w_not_edge = torch.sum(w_not_edge * one_hot_lab.float(), dim=3, keepdim=True)  # NxHxWx1x1

    edge_loss *= w_edge.permute(0, 3, 1, 2, 4)
    not_edge_loss *= w_not_edge.permute(0, 3, 1, 2, 4)

    not_edge_loss = torch.reshape(not_edge_loss, (-1, 1))
    not_edge_loss = torch.gather(not_edge_loss, 0, not_edge_indices)
    edge_loss = torch.reshape(edge_loss, (-1, 1))
    edge_loss = torch.gather(edge_loss, 0, edge_indices)

    return edge_loss, not_edge_loss
