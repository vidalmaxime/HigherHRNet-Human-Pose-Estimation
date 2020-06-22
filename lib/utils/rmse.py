import argparse
import json
import numpy as np
from coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate RMSE')

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='/home/maxime/Documents/mathislab/bearproject/data/annotations_trainval2017/annotations/person_keypoints_val2017.json'
                        )

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        default='/home/maxime/Documents/mathislab/bearproject/data/annotations_trainval2017/keypointshuman.json')

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


def plot(data, gt_file):
    # joints
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(data)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval._prepare()
    gts_ = coco_eval._gts
    dts_ = coco_eval._dts

    p = coco_eval.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)

    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]
    joint_thres = 0.1

    mean_rmse_list = []
    mean_rmse_mask_list = []
    for catId in catIds:
        for imgId in p.imgIds:
            # dimension here should be Nxm
            gts = gts_[imgId, catId]
            dts = dts_[imgId, catId]
            if len(gts) != 0 and len(dts) != 0:
                npgt = np.array(gts[0]["keypoints"])
                npdt = np.array(dts[0]["keypoints"])
                mask = npdt[2::3] >= joint_thres
                maskgt = npgt[2::3] > 0

                RMSE = np.sqrt((npgt[0::3]*[maskgt] - npdt[0::3]*[maskgt]) ** 2 + (npgt[1::3]*[maskgt] - npdt[1::3]*[maskgt]) ** 2)

                RMSE_mask = RMSE[0][mask]
                print(maskgt)
                mean_rmse = np.nansum(RMSE[0])/np.nansum(maskgt)
                mean_rmse_mask = np.nansum(RMSE_mask)/np.nansum(maskgt)
                print(f"mean rmse: {mean_rmse}")
                print(f"mean rmse mask: {mean_rmse_mask}")
                mean_rmse_list.append(mean_rmse)
                mean_rmse_mask_list.append(mean_rmse_mask)

            inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
            dts = [dts[i] for i in inds]
            if len(dts) > p.maxDets[-1]:
                dts = dts[0:p.maxDets[-1]]
            if len(gts) == 0 or len(dts) == 0:
                continue

    print(f"total mean rmse: {np.nanmean(mean_rmse_list)}")
    print(f"total mean rmse mask: {np.nanmean(mean_rmse_mask_list)}")


if __name__ == '__main__':
    args = parse_args()

    with open(args.prediction) as f:
        data = json.load(f)
    gt_file = args.gt_anno
    plot(data, gt_file)
