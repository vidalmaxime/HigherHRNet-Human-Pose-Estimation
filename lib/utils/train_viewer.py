# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# Modified by Maxime Vidal
# ------------------------------------------------------------------------------

import argparse
import json
import os

import cv2
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage import io


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))


# Style
# (R,G,B)
color2 = [(169, 209, 142), (169, 209, 142),
          (20, 50, 90), (20, 50, 90),
          (169, 209, 142), (20, 50, 90),
          (0, 176, 240), (0, 176, 240), (0, 176, 240),
          (252, 176, 243), (252, 176, 243), (252, 176, 243), (252, 176, 243), (252, 176, 243),
          (240, 2, 127),
          (255, 255, 0),
          (240, 2, 127),
          (240, 2, 127),
          (255, 255, 0),
          (255, 255, 0)]

link_pairs2 = [
    [15, 17], [17, 19], [16, 18],
    [18, 20], [19, 8], [20, 8],
    [8, 7], [7, 6], [6, 1], [1, 2], [1, 3],
    [2, 3], [2, 4], [3, 5], [6, 13], [6, 14], [13, 11], [11, 9], [14, 12], [12, 10]
]

point_color2 = [(252, 176, 243), (252, 176, 243), (252, 176, 243), (252, 176, 243), (252, 176, 243),
                (0, 176, 240), (0, 176, 240), (0, 176, 240),
                (240, 2, 127),
                (255, 255, 0),
                (240, 2, 127),
                (255, 255, 0),
                (240, 2, 127),
                (255, 255, 0),
                (169, 209, 142),
                (20, 50, 90),
                (169, 209, 142),
                (20, 50, 90),
                (169, 209, 142),
                (20, 50, 90)]

style = ColorStyle(color2, link_pairs2, point_color2)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='/home/maxime/Documents/mathislab/bearproject/HigherHRNet-Human-Pose-Estimation/data/coco/images/val/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='/home/maxime/Documents/mathislab/bearproject/HigherHRNet-Human-Pose-Estimation/data/coco/annotations/animal_keypoints_val.json'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/cocoval/')

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i + 1
        joints_dict[id] = (x, y)

    return joints_dict


def plot(gt_file, img_path, save_path):
    coco = COCO(gt_file)
    catIds = coco.getCatIds(catNms=['animal'])
    imgIds = coco.getImgIds(catIds=catIds)
    #index = np.random.randint(0, len(imgIds))
    for index in range(len(imgIds)):
        img = coco.loadImgs(imgIds[index])[0]  # Load images with specified ids.
        print(img)
        i = io.imread(os.path.join(img_path, img['file_name']))
        plt.axis('off')
        plt.imshow(i)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.savefig(save_path + img["file_name"].split(".")[0] + '.png', format='png', bbox_inckes='tight', dpi=100)
        #plt.show()
        plt.close()



if __name__ == '__main__':

    args = parse_args()

    colorstyle = style
    save_path = args.save_path
    img_path = args.image_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))

    gt_file = args.gt_anno
    plot(gt_file, img_path, save_path)
