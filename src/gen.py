import os
import random
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from label import Label
from projection_utils import find_T_matrix, getRectPts, perspective_transform
from utils import IOU_centre_and_dims, getWH, hsv_transform, im2single


class CCPDDataGen(tf.keras.utils.Sequence):
    def __init__(self, data_dir: str, batch_size, input_size=(224, 224), shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        img_exts = ["jpg", "jpeg", "png"]
        self.image_files = [
            os.path.join(self.data_dir, filename)
            for filename in os.listdir(self.data_dir)
            if filename.split(".")[-1] in img_exts
        ]
        self.n = len(self.image_files)

    def _load_data(self, batches: List[str]):
        images = []
        boxes = []
        for path in batches:
            img = cv2.imread(path)
            img = cv2.resize(img, self.input_size)

            img_name = os.path.basename(path)
            iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
            (rightdown, leftdown, leftup, rightup) = [[int(eel) for eel in el.split("&")] for el in iname[3].split("_")]
            ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
            bbox_coordinates = np.array(
                [
                    (point[0] / ori_w, point[1] / ori_h)
                    for point in (leftup, rightup, leftdown, rightdown)
                    # for point in (leftup, rightdown)
                ]
            ).reshape(2, 4)
            ROI, llp, pts = augment_sample(img, bbox_coordinates, img.shape[0])
            y = labels2output_map(llp, pts, img.shape[1], 16)

            boxes.append(y)
            images.append(ROI)
        return np.array(images), np.array(boxes)

    def __getitem__(self, index):
        batches = self.image_files[index * self.batch_size : (index + 1) * self.batch_size]
        x, y = self._load_data(batches)
        return x, y

    def __len__(self):
        return self.n // self.batch_size


def augment_sample(I, pts, dim):

    maxsum, maxangle = 120, np.array([80.0, 80.0, 45.0])
    angles = np.random.rand(3) * maxangle
    if angles.sum() > maxsum:
        angles = (angles / angles.sum()) * (maxangle / maxangle.sum())

    I = im2single(I)
    iwh = getWH(I.shape)

    whratio = random.uniform(2.0, 4.0)
    wsiz = random.uniform(dim * 0.2, dim * 1.0)

    hsiz = wsiz / whratio

    dx = random.uniform(0.0, dim - wsiz)
    dy = random.uniform(0.0, dim - hsiz)

    pph = getRectPts(dx, dy, dx + wsiz, dy + hsiz)
    pts = pts * iwh.reshape((2, 1))
    T = find_T_matrix(pts2ptsh(pts), pph)

    H = perspective_transform((dim, dim), angles=angles)
    H = np.matmul(H, T)

    Iroi, pts = project(I, H, pts, dim)

    hsv_mod = np.random.rand(3).astype("float32")
    hsv_mod = (hsv_mod - 0.5) * 0.3
    hsv_mod[0] *= 360
    Iroi = hsv_transform(Iroi, hsv_mod)
    Iroi = np.clip(Iroi, 0.0, 1.0)

    pts = np.array(pts)

    if random.random() > 0.5:
        Iroi, pts = flip_image_and_pts(Iroi, pts)

    tl, br = pts.min(1), pts.max(1)
    llp = Label(0, tl, br)

    return Iroi, llp, pts


def labels2output_map(label, lppts, dim, stride):

    side = ((float(dim) + 40.0) / 2.0) / stride  # 7.75 when dim = 208 and stride = 16

    outsize = int(dim / stride)
    Y = np.zeros((outsize, outsize, 2 * 4 + 1), dtype="float32")
    MN = np.array([outsize, outsize])
    WH = np.array([dim, dim], dtype=float)

    tlx, tly = np.floor(np.maximum(label.tl(), 0.0) * MN).astype(int).tolist()
    brx, bry = np.ceil(np.minimum(label.br(), 1.0) * MN).astype(int).tolist()

    for x in range(tlx, brx):
        for y in range(tly, bry):

            mn = np.array([float(x) + 0.5, float(y) + 0.5])
            iou = IOU_centre_and_dims(mn / MN, label.wh(), label.cc(), label.wh())

            if iou > 0.5:

                p_WH = lppts * WH.reshape((2, 1))
                p_MN = p_WH / stride

                p_MN_center_mn = p_MN - mn.reshape((2, 1))

                p_side = p_MN_center_mn / side

                Y[y, x, 0] = 1.0
                Y[y, x, 1:] = p_side.T.flatten()

    return Y


def pts2ptsh(pts):
    return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, dim):
    ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
    ptsh = np.matmul(T, ptsh)
    ptsh = ptsh / ptsh[2]
    ptsret = ptsh[:2]
    ptsret = ptsret / dim
    Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=0.0, flags=cv2.INTER_LINEAR)
    return Iroi, ptsret


def flip_image_and_pts(I, pts):
    I = cv2.flip(I, 1)
    pts[0] = 1.0 - pts[0]
    idx = [1, 0, 3, 2]
    pts = pts[..., idx]
    return I, pts
