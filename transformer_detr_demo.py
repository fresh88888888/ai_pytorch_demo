
import sys
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
import cv2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
import keras as keras

from keras.applications import DenseNet121

################# DETR FUCNTIONS FOR LOSS########################
sys.path.extend(['/Users/zcj/py_workspace/hello/detr/'])
#################################################################

n_folds = 5
seed = 42
null_class_coef = 0.5
num_classes = 1
num_queries = 100
BATCH_SIZE = 8
LR = 5e-5
lr_dict = {'backbone': 0.1, 'transformer': 1, 'embed': 1, 'final': 5}
EPOCHS = 2
max_norm = 0
model_name = 'detr_resnet50'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return (self.sum / self.count) if self.count > 0 else 0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed)
