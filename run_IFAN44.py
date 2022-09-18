import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import pdb
import os
import numpy as np

from models import create_model
from utils import *
from ckpt_manager import CKPT_Manager
import importlib

from eval import *


def read_frame(path, norm_val = None, rotate = None):
    if norm_val == (2**16-1):
        frame = cv2.imread(path, -1)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / norm_val
        frame = frame[...,::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / 255.
    return np.expand_dims(frame, axis = 0)


def refine_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]
    
    
def IFAN44_create(rootpath):
    
    # load config
    config_lib = importlib.import_module('configs.{}'.format('config_IFAN_44'))
    config = config_lib.get_config('IFAN_CVPR2021', 'IFAN_44', 'config_IFAN_44')
    config.is_train = False
    
    config.network = 'IFAN'
    config.EVAL.ckpt_name = None
    config.EVAL.ckpt_abs_name = os.path.join(rootpath, 'ckpt/IFAN_44.pytorch')
    config.EVAL.ckpt_epoch = None
    config.EVAL.load_ckpt_by_score = False
    
    model = create_model(config)
    network = model.get_network().eval()
    
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}'.format(ckpt_name, config.mode, load_state))
    
    return config, network


def run_IFAN44(path, rootpath):
    
    # load config
    config, network = IFAN44_create(rootpath)
    
    C = refine_image(read_frame(path, config.norm_val, None), config.refine_val)
    C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
    
    _, _, H, W = C.shape
    res_h, res_w = 1200, 1600
    C = C[:, :, H//2-res_h//2:H//2+res_h//2, W//2-res_w//2:W//2+res_w//2]
    
    with torch.no_grad():
        out = network(C=C, is_train=False)
        output = out['result']
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0) #[0, 1]
    return output_cpu
        
        
if __name__ == '__main__':
    output = run_IFAN44('./IFAN/demo/input3/ex3.png', './IFAN')
    output = cv2.cvtColor(output * 255, cv2.COLOR_BGR2RGB)
    # output = cv2.medianBlur(output, 3)
    output = cv2.bilateralFilter(output, -1, 100, 10)
    cv2.imwrite('ex.png', output)
    
