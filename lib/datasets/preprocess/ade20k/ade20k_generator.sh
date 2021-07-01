#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


ORI_ROOT_DIR='/cluster/work/cvl/tiazhou/data/ADE20K/ADEChallengeData2016'
SAVE_DIR='/cluster/work/cvl/tiazhou/data/ADE20K/ADEChallengeData2016'


python ade20k_generator.py --ori_root_dir $ORI_ROOT_DIR \
                           --save_dir $SAVE_DIR