#!/usr/bin/env python
# coding=utf-8
'''
@Author: FishSzh
@Date: 2020-04-22 11:49:05
@LastEditTime: 2020-04-26 11:11:26
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def roi_num_demo(imgs, titles):
    nn = 2
    img_nums = len(imgs)-2
    fig = plt.figure(figsize=(4+4*img_nums//nn, 4), constrained_layout=True)
    gs = gridspec.GridSpec(6,1+img_nums//nn, figure=fig)
    
    ax = fig.add_subplot(gs[0:3,0])
    ax.imshow(imgs[0][:,:,::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[3:6,0])
    ax.imshow(imgs[1], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(img_nums):
        if i%nn == 0:
            ax = fig.add_subplot(gs[0,1+i//nn])
            ax.set_title(titles[i//nn])
            ax.imshow(imgs[2+i], cmap='gray')
        # elif i%nn == 1:
        #     ax = fig.add_subplot(gs[1,1+i//nn])
        #     ax.imshow(imgs[2+i], cmap='gray')
        else:
            ax = fig.add_subplot(gs[2:6,1+i//nn])
            ax.imshow(imgs[2+i][:,:,::-1])
        ax.set_xticks([])
        ax.set_yticks([])

    # for i in range(16):
    #     ax = fig.add_subplot(gs[1,16+i])
    #     ax.imshow(imgs[3][i], cmap='gray')
    plt.show()

    
