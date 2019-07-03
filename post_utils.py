import scipy.ndimage as ndimg
import numpy as np
import math
from options import Options
from numpy.linalg import eig
import cv2

opt = Options()


def gcf_extract(hm_list):
    gcf_blocks = []
    gcf_cs = []
    gcf_rs = []
    for heatmap in hm_list:
        heatmap = heatmap.astype('uint8')
        msk = heatmap > 30
        lab, n = ndimg.label(msk)
        r, c = np.mgrid[:heatmap.shape[0], :heatmap.shape[1]]

        S00 = ndimg.sum(heatmap, lab, np.arange(1, n + 1))
        Sr1 = ndimg.sum(heatmap * r, lab, np.arange(1, n + 1))
        Sc1 = ndimg.sum(heatmap * c, lab, np.arange(1, n + 1))
        Srr = ndimg.sum(heatmap * r ** 2, lab, np.arange(1, n + 1))
        Scc = ndimg.sum(heatmap * c ** 2, lab, np.arange(1, n + 1))
        Src = ndimg.sum(heatmap * r * c, lab, np.arange(1, n + 1))
        rs = Sr1 / S00
        cs = Sc1 / S00
        Srr0 = Srr - S00 * rs ** 2
        Scc0 = Scc - S00 * cs ** 2
        Src0 = Src - S00 * rs * cs

        covs = np.hstack((Srr0, Src0, Src0, Scc0))
        covs = covs.reshape((4, -1)).T.reshape((-1, 2, 2))
        # convs is the covariance matrix of each area, (rs,cs) is the center of mass, S00 is the mass
        if list(S00) != []:
            blocks = list(zip(covs, rs, cs, S00))
            gcf_blocks_app = [block for block in blocks if block[3] > 10000]
            gcf_cs_app = [block[2] for block in blocks if block[3] > 10000]
            gcf_rs_app = [block[1] for block in blocks if block[3] > 10000]
            gcf_blocks += gcf_blocks_app
            gcf_cs += gcf_cs_app
            gcf_rs += gcf_rs_app
    gcf_blocks = gcf_blocks
    gcf_rs = gcf_rs
    gcf_cs = gcf_cs
    gcfs = []
    for gcf_index in range(len(gcf_blocks)):
        gcf_block = gcf_blocks[gcf_index]
        e, d = eig(gcf_block[0])
        r = gcf_block[1]
        c = gcf_block[2]
        s = gcf_block[3]
        dist1 = (d[0, 0] ** 2 + d[1, 0] ** 2) * (e[0] / s / 2) ** 2
        dist2 = (d[0, 1] ** 2 + d[1, 1] ** 2) * (e[1] / s / 2) ** 2
        if dist1 >= dist2:
            angle = math.atan(-d[0, 0] / d[1, 0] * opt.x_ratio / opt.y_ratio)
        else:
            angle = math.atan(-d[0, 1] / d[1, 1] * opt.x_ratio / opt.y_ratio)
        x = gcf_block[2] / opt.x_ratio
        y = gcf_block[1] / opt.y_ratio
        gcfs.append([x, y, angle])
    return gcfs


def vis_gcfs(gcfs,image):
    for gcf in gcfs:
        x = gcf[0]
        y = gcf[1]
        angle = gcf[2]

        point1 = (int(x - 40 * math.cos(angle)), int(y + 40 * math.sin(angle)))
        point2 = (int(x + 40 * math.cos(angle)), int(y - 40 * math.sin(angle)))
        cv2.line(image, point1, point2, (0, 0, 255), 5)
        cv2.circle(image, point1, 5, (0, 255, 255), 5)
        cv2.circle(image, point2, 5, (0, 255, 255), 5)
    return image

