# Exploring Cross-Image Pixel Contrast for Semantic Segmentation

![](figures/framework.png)

> [**Exploring Cross-Image Pixel Contrast for Semantic Segmentation**](https://arxiv.org/abs/2101.11939),            
> [Wenguan Wang](https://sites.google.com/view/wenguanwang/), [Tianfei Zhou](https://www.tfzhou.com/), [Fisher Yu](https://www.yf.io/), [Jifeng Dai](https://jifengdai.org/), [Ender Konukoglu](https://scholar.google.com/citations?user=OeEMrhQAAAAJ&hl=en) and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en) <br>
> *arXiv technical report ([arXiv 2101.11939](https://arxiv.org/abs/2101.11939))*  

## Abstract

Current semantic segmentation methods focus only on
mining “local” context, i.e., dependencies between pixels
within individual images, by context-aggregation modules
(e.g., dilated convolution, neural attention) or structureaware optimization criteria (e.g., IoU-like loss). However, they ignore “global” context of the training data, i.e.,
rich semantic relations between pixels across different images. Inspired by the recent advance in unsupervised contrastive representation learning, we propose a pixel-wise
contrastive framework for semantic segmentation in the
fully supervised setting. The core idea is to enforce pixel
embeddings belonging to a same semantic class to be more
similar than embeddings from different classes. It raises a
pixel-wise metric learning paradigm for semantic segmentation, by explicitly exploring the structures of labeled pixels, which are long ignored in the field. Our method can be
effortlessly incorporated into existing segmentation frameworks without extra overhead during testing.

We experimentally show that, with famous segmentation models (i.e.,
DeepLabV3, HRNet, OCR) and backbones (i.e., ResNet, HRNet), our method brings consistent performance improvements across diverse datasets (i.e., Cityscapes, PASCALContext, COCO-Stuff).

## Installation

This implementation is built on [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch). Many thanks to the authors for the efforts.

Please follow the [Getting Started](https://github.com/openseg-group/openseg.pytorch/blob/master/GETTING_STARTED.md) for installation and dataset preparation.

## Running

### Cityscapes

1.  Train ```DeepLabV3```

    ```bash scripts/cityscapes/deeplab/run_r_101_d_8_deeplabv3_train_contrast.sh train 'resnet101-deeplabv3-contrast'```

## Features (in progress)

- [x] Pixel-wise Contrastive Loss
- [x] Hard Anchor Sampling
- [ ] Memory Bank
- [ ] Hard Example Mining
- [ ] Model Zoo


## t-SNE Visualization

* Pixel-wise Cross-Entropy Loss
<p align="center">
  <img src="figures/tsne1.png" width="400">
</p>

* Pixel-wise Contrastive Learning Objective 
  
<p align="center">
  <img src="figures/tsne2.png" width="400">
</p>  

## Citation
```
@article{wang2021exploring,
  title   = {Exploring Cross-Image Pixel Contrast for Semantic Segmentation},
  author  = {Wang, Wenguan and Zhou, Tianfei and Yu, Fisher and Dai, Jifeng and Konukoglu, Ender and Van Gool, Luc},
  journal = {arXiv preprint arXiv:2101.11939},
  year    = {2021}
}
```
