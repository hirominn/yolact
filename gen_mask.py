from pickletools import uint8
from yolact.data import COCODetection, get_label_map, MEANS, COLORS
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from yolact.utils.functions import MovingAverage, ProgressBar
from yolact.layers.box_utils import jaccard, center_size, mask_iou
from yolact.utils import timer
from yolact.utils.functions import SavePath
from yolact.layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from yolact.data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2


def acquire_mask(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    score_threshold = 0.15
    top_k = 15

    # start = time.perf_counter()

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, 
                                        #visualize_lincomb = args.display_lincomb,
                                        visualize_lincomb = False,
                                        # crop_masks        = args.crop,
                                        crop_masks        = True,
                                        # score_threshold   = args.score_threshold)
                                        score_threshold   = score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break
    # stop = time.perf_counter()
    # print("-- -- Acquiring Duration:", (stop - start) * 1000, "ms")
    return masks.cpu().numpy()[:num_dets_to_consider], boxes[:num_dets_to_consider], classes[:num_dets_to_consider]


def evalimage(net:Yolact, image):
    # start = time.perf_counter()
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    # stop = time.perf_counter()
    # print("-- Evaluation Duration:", (stop - start) * 1000, "ms")
    img_numpy = acquire_mask(preds, frame, None, None, undo_transform=False)
    return img_numpy

def evaluate(net:Yolact, image):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False
    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    return evalimage(net, image)

def prepare_net(trained_model):
    model_path = SavePath.from_str(trained_model)
    config = model_path.model_name + '_config'
    set_cfg(config)

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    print(' Done.')
    net = net.cuda()

    return net

def gen_mask(net, image):
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        return evaluate(net, image)

def get_class_names():
    return cfg.dataset.class_names

if __name__ == '__main__':
    image = cv2.imread('data/yolact_example_0.png')
    trained_model='weights/yolact_plus_resnet50_54_800000.pth'

    net = prepare_net(trained_model)

    for i in range(20):
        # start = time.perf_counter()
        gen_mask(net, image)
        # stop = time.perf_counter()
        # print("Generating Duration:", (stop - start) * 1000, "ms")