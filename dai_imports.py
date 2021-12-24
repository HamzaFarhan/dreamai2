# Imports

import torch
import torchaudio
import torchvision
from torch import mm
import torch.nn as nn
import torch.optim as optim
# import torch.tensor as tensor
from torchvision.ops import nms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets.folder import*
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import RandomSampler
from torchvision import datasets, models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, mask_rcnn
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_resnet101
from torchvision.models import (vgg16,vgg16_bn,densenet121,densenet169,resnet18,resnet34,resnet50,resnet101,resnet152,
                                resnext50_32x4d,resnext101_32x8d)

import re
import io
import os
import cv2
import json
import math
import copy
import time
import rawpy
import heapq
import shutil
import pickle
import random
import optuna
import adaptdl
import inspect
# import skimage
import logging
import pathlib
import requests
import colorsys
import traceback
import itertools
import matplotlib
import numpy as np
import pandas as pd
# from tqdm import tqdm
from math import sqrt
from scipy import stats
from pathlib import Path
# import horovod.torch as hvd
import adaptdl.torch as adt
from functools import wraps
from ast import literal_eval
from functools import partial
import albumentations as albu
from datetime import datetime
from matplotlib import colors
# import pytorch_lightning as pl
import moviepy.editor as editor
from collections import Counter
import matplotlib.pyplot as plt
from pprint import PrettyPrinter
from torchsummary import summary
from os.path import isfile, join
# from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
# from livelossplot import PlotLosses
from collections import OrderedDict
from collections import defaultdict
from argparse import ArgumentParser
from optuna.trial import TrialState
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from itertools import product as product
from albumentations import pytorch as AT
# import segmentation_models_pytorch as smp
from efficientnet_pytorch import EfficientNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pytorch_lightning.callbacks.base import Callback
# from pytorch_lightning import callbacks as pl_callbacks
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, mean_squared_error
