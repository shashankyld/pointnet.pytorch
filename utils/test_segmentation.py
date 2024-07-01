# This script is to test the segmentation model on the test dataset.
# Visualize for every datapoint

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data 
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
