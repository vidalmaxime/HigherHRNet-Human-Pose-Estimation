# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
from core.group import HeatmapParser
from core.inference import aggregate_results
from core.inference import get_multi_stage_outputs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import resize_align_multi_scale
from utils.utils import AverageMeter
from utils.vis import save_valid_image

logger = logging.getLogger(__name__)


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def validate(config, val_loader, val_dataset, model,  output_dir,
             tb_log_dir, writer_dict=None):
    model.eval()
    if config.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    # rmse = AverageMeter()

    parser = HeatmapParser(config)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(val_dataset)) if config.TEST.LOG_PROGRESS else None
    for i, (images, annos) in enumerate(val_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, config.DATASET.INPUT_SIZE, 1.0, min(config.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(config.TEST.SCALE_FACTOR, reverse=True)):
                input_size = config.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(config.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    config, model, image_resized, config.TEST.FLIP_TEST,
                    config.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    config, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(config.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, config.TEST.ADJUST, config.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        if config.TEST.LOG_PROGRESS:
            pbar.update()

        # if i % config.PRINT_FREQ == 0:
        #     prefix = '{}_{}'.format(os.path.join(output_dir, 'result_valid'), i)
        #     # logger.info('=> write {}'.format(prefix))
        #     save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=val_dataset.name)
        #     # save_debug_images(config, image_resized, None, None, outputs, prefix)

        all_preds.append(final_results)
        all_scores.append(scores)

    if config.TEST.LOG_PROGRESS:
        pbar.close()

    results, res_file = val_dataset.evaluate(
        config, all_preds, all_scores, output_dir
    )
    ##################################
    gt_file = val_dataset._get_anno_file_name()
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(res_file)
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
            if dts:
                npgt = np.array(gts[0]["keypoints"])
                npdt = np.array(dts[0]["keypoints"])
                mask = npdt[2::3] >= joint_thres
                RMSE = np.sqrt((npgt[0::3] - npdt[0::3]) ** 2 + (npgt[1::3] - npdt[1::3]) ** 2)
                RMSE_mask = RMSE[mask]
                mean_rmse = np.round(np.nanmean(RMSE.flatten()), 2)
                mean_rmse_mask = np.round(np.nanmean(RMSE_mask.flatten()), 2)
                mean_rmse_list.append(mean_rmse)
                mean_rmse_mask_list.append(mean_rmse_mask)
    print(f"Mean RMSE: {np.mean(mean_rmse_list)}")
    print(f"Mean RMSE mask: {np.mean(mean_rmse_mask_list)}")
    global_steps = writer_dict['valid_global_steps']
    writer_dict["writer"].add_scalar(
        "rmse",
        np.mean(mean_rmse_list),
        global_steps
    )
    writer_dict["writer"].add_scalar(
        "rmse_mask",
        np.mean(mean_rmse_mask_list),
        global_steps
    )
    writer_dict['valid_global_steps'] = global_steps + 1
    return np.mean(mean_rmse_list)
