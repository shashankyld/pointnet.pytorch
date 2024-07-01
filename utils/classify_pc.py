
# python classify_pc.py --dataset /home/shashank/Documents/UniBonn/Sem4/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/ --nepoch=4 --model /home/shashank/Documents/UniBonn/Sem4/pointnet.pytorch/utils/seg/seg_model_Chair_3.pth



from __future__ import print_function
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
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
    print(f"Loaded model from {opt.model}")

classifier.cuda()

shape_ious = []

for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points, target = points.cuda(), target.cuda()
    points = points.transpose(2, 1)
    classifier = classifier.eval()

    with torch.no_grad():
        pred, trans, trans_feat = classifier(points)

    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0] - 1
    loss = F.nll_loss(pred, target)
    print('loss', loss.item())
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('correct', correct.item())
    print('accuracy', correct.item() / float(opt.batchSize * 2500))
    
    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    
    for shape_idx in range(opt.batchSize):
        parts = range(num_classes)  # assuming all parts are included
        part_ious = []
        for part in parts:
            pred_part = (pred_np == part)
            target_part = (target_np == part)
            if np.sum(target_part) == 0 and np.sum(pred_part) == 0:
                iou = 1.0
            else:
                iou = np.sum(pred_part & target_part) / float(np.sum(pred_part | target_part))
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        
    for i in range(opt.batchSize):
        points_np = points[i].transpose(1, 0).cpu().numpy()
        pred_choice_np = pred_choice.view(opt.batchSize, -1)[i].cpu().numpy()
        target_np = target.view(opt.batchSize, -1)[i].cpu().numpy()
        np.save(os.path.join(opt.outf, f'points_{i}.npy'), points_np)
        np.save(os.path.join(opt.outf, f'pred_{i}.npy'), pred_choice_np)
        np.save(os.path.join(opt.outf, f'target_{i}.npy'), target_np)
    break

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
