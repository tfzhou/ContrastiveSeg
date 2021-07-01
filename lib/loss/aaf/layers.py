import torch
import torch.nn.functional as F
import numpy as np


def eightway_activation(x):
    """Retrieves neighboring pixels/features on the eight corners from
  a 3x3 patch.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels, 8]
  """
    # Get the number of channels in the input.
    shape_x = list(x.shape)
    if len(shape_x) != 4:
        raise ValueError('Only support for 4-D tensors!')

    # Pad at the margin.
    x = F.pad(x,
              pad=(0, 0, 1, 1, 1, 1, 0, 0),
              mode='reflect')
    # Get eight neighboring pixels/features.
    x_groups = [
        x[:, 1:-1, :-2, :].clone(),  # left
        x[:, 1:-1, 2:, :].clone(),  # right
        x[:, :-2, 1:-1, :].clone(),  # up
        x[:, 2:, 1:-1, :].clone(),  # down
        x[:, :-2, :-2, :].clone(),  # left-up
        x[:, 2:, :-2, :].clone(),  # left-down
        x[:, :-2, 2:, :].clone(),  # right-up
        x[:, 2:, 2:, :].clone()  # right-down
    ]
    output = [
        torch.unsqueeze(c, dim=-1) for c in x_groups
    ]
    output = torch.cat(output, dim=-1)

    return output


def eightcorner_activation(x, size):
    """Retrieves neighboring pixels one the eight corners from a
  (2*size+1)x(2*size+1) patch.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]
    size: A number indicating the half size of a patch.

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels, 8]
  """
    # Get the number of channels in the input.
    shape_x = list(x.shape)
    if len(shape_x) != 4:
        raise ValueError('Only support for 4-D tensors!')
    n, c, h, w = shape_x

    # Pad at the margin.
    p = size
    x_pad = F.pad(x,
                  pad=(p, p, p, p, 0, 0, 0, 0),
                  mode='constant',
                  value=0)

    # Get eight corner pixels/features in the patch.
    x_groups = []
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue

            x_neighbor = x_pad[:, :, st_y:st_y + h, st_x:st_x + w].clone()
            x_groups.append(x_neighbor)

    output = [torch.unsqueeze(c, dim=-1) for c in x_groups]
    output = torch.cat(output, dim=-1)

    return output


def ignores_from_label(labels, num_classes, size, ignore_index):
    """Retrieves ignorable pixels from the ground-truth labels.

  This function returns a binary map in which 1 denotes ignored pixels
  and 0 means not ignored ones. For those ignored pixels, they are not
  only the pixels with label value >= num_classes, but also the
  corresponding neighboring pixels, which are on the the eight cornerls
  from a (2*size+1)x(2*size+1) patch.
  
  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    num_classes: A number indicating the total number of valid classes. The 
      labels ranges from 0 to (num_classes-1), and any value >= num_classes
      would be ignored.
    size: A number indicating the half size of a patch.

  Return:
    A tensor of size [batch_size, height_in, width_in, 8]
  """
    # Get the number of channels in the input.
    shape_lab = list(labels.shape)
    if len(shape_lab) != 3:
        raise ValueError('Only support for 3-D label tensors!')
    n, h, w = shape_lab

    # Retrieve ignored pixels with label value >= num_classes.
    # ignore = labels>num_classes-1 # NxHxW
    ignore = (labels == ignore_index)

    # Pad at the margin.
    p = size
    ignore_pad = F.pad(ignore,
                       pad=(p, p, p, p, 0, 0),
                       mode='constant',
                       value=1)

    # Retrieve eight corner pixels from the center, where the center
    # is ignored. Note that it should be bi-directional. For example,
    # when computing AAF loss with top-left pixels, the ignored pixels
    # might be the center or the top-left ones.
    ignore_groups = []
    for st_y in range(2 * size, -1, -size):
        for st_x in range(2 * size, -1, -size):
            if st_y == size and st_x == size:
                continue
            ignore_neighbor = ignore_pad[:, st_y:st_y + h, st_x:st_x + w].clone()
            mask = ignore_neighbor | ignore
            ignore_groups.append(mask)

    ig = 0
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            if st_y == size and st_x == size:
                continue
            ignore_neighbor = ignore_pad[:, st_y:st_y + h, st_x:st_x + w].clone()
            mask = ignore_neighbor | ignore_groups[ig]
            ignore_groups[ig] = mask
            ig += 1

    ignore_groups = [
        torch.unsqueeze(c, dim=-1) for c in ignore_groups
    ]  # NxHxWx1
    ignore = torch.cat(ignore_groups, dim=-1)  # NxHxWx8

    return ignore


def edges_from_label(labels, size, ignore_class=255):
    """Retrieves edge positions from the ground-truth labels.

  This function computes the edge map by considering if the pixel values
  are equal between the center and the neighboring pixels on the eight
  corners from a (2*size+1)*(2*size+1) patch. Ignore edges where the any
  of the paired pixels with label value >= num_classes.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    size: A number indicating the half size of a patch.
    ignore_class: A number indicating the label value to ignore.

  Return:
    A tensor of size [batch_size, height_in, width_in, 1, 8]
  """
    # Get the number of channels in the input.
    shape_lab = list(labels.shape)
    if len(shape_lab) != 4:
        raise ValueError('Only support for 4-D label tensors!')
    n, h, w, c = shape_lab

    # Pad at the margin.
    p = size
    labels_pad = F.pad(
        labels, pad=(0, 0, p, p, p, p, 0, 0),
        mode='constant',
        value=ignore_class)

    # Get the edge by comparing label value of the center and it paired pixels.
    edge_groups = []
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            if st_y == size and st_x == size:
                continue
            labels_neighbor = labels_pad[:, st_y:st_y + h, st_x:st_x + w]
            edge = labels_neighbor != labels
            edge_groups.append(edge)

    edge_groups = [
        torch.unsqueeze(c, dim=-1) for c in edge_groups
    ]  # NxHxWx1x1
    edge = torch.cat(edge_groups, dim=-1)  # NxHxWx1x8

    return edge
