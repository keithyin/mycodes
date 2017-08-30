from skimage.io import imread
import os
import numpy as np

CLASSES_COLORS = [[0, 0, 0],
                  [0, 0, 128],
                  [0, 64, 128],
                  [224, 224, 192],
                  [128, 0, 0],
                  [192, 128, 128],
                  [128, 192, 0],
                  [192, 0, 0],
                  [64, 0, 128],
                  [128, 128, 0],
                  [0, 128, 0],
                  [128, 0, 128],
                  [128, 64, 0],
                  [192, 128, 0],
                  [192, 0, 128],
                  [64, 128, 128],
                  [0, 192, 0],
                  [64, 128, 0],
                  [128, 128, 128],
                  [64, 0, 0],
                  [0, 128, 128],
                  [0, 64, 0]]
NUM_CLASSES = 22

#
IMAGE_PATH = '/media/keith/Download/DataSets/VOCdevkit/VOC2012/SegmentationClass'

imgs = os.listdir(IMAGE_PATH)

imgs_path = [os.path.join(IMAGE_PATH, img) for img in imgs]

num_imgs = len(imgs_path)
classes_colors = []

# for img_index, img_path in enumerate(imgs_path):
#     print(img_index, "/", num_imgs)
#     img_bn = imread(img_path)
#     height, width, _ = img_bn.shape
#     for i in range(height):
#         for j in range(width):
#             color = list(img_bn[i, j])
#             if not (color in classes_colors):
#                 classes_colors.append(color)

labels = []
for color in CLASSES_COLORS:
    gray = 0.2125*color[0]+0.7154*color[1]+0.0721*color[2]
    labels.append(gray)
print(labels)

