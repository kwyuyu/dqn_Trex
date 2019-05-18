import os
import matplotlib
import matplotlib.image as plimage
import torch


def save_image(img, file_name):
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    plimage.imsave(os.path.join('tmp', file_name), img)


def save_immediate_image(img, file_name):
    if not os.path.exists('image'):
        os.makedirs('image')

    img, _ = torch.max(img[0, :, :, :], dim = 0)

    plimage.imsave(os.path.join('image', file_name), img, cmap = matplotlib.cm.gray)
