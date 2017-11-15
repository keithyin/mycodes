import numpy as np
import torch
from torchvision import transforms


def img_process(img):
    assert len(img.shape) == 3
    # crop the dusk drive' img to  [85:595, 20:815]]
    res_img = img[85:595, 20:815, :]
    return res_img
