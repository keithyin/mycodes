import numpy as np
import cv2

imgSize = (112, 96)

x_ = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299]
y_ = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]

src = np.array(zip(x_, y_)).astype(np.float32).reshape(1, 5, 2)

alignedFaces = []

# there might be more than one faces, hence
# multiple sets of points
for pset in points:
    img2 = img.copy()

    pset_x = pset[0:5]
    pset_y = pset[5:10]

    dst = np.array(zip(pset_x, pset_y)).astype(np.float32).reshape(1, 5, 2)

    transmat = cv2.estimateRigidTransform(dst, src, False)

    out = cv2.warpAffine(img2, transmat, (imgSize[1], imgSize[0]))

    alignedFaces.append(out)