from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pointnet.model import PointNetDenseCls

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='path to the trained model')
parser.add_argument('--datafolder', type=str, required=True, help='folder containing .txt files of point clouds')
parser.add_argument('--outputfolder', type=str, default='output', help='output folder for visualizations')
parser.add_argument('--num_classes', type=int, default=4, help='number of segmentation classes')
parser.add_argument('--batchSize', type=int, default=16, help='batch size for inference')

opt = parser.parse_args()
print(opt)

# Create output folder if not exists
os.makedirs(opt.outputfolder, exist_ok=True)

# Load point cloud data from .txt files
def load_point_clouds(datafolder):
    point_clouds = []
    file_paths = [os.path.join(datafolder, f) for f in os.listdir(datafolder) if f.endswith('.txt')]
    for file_path in file_paths:
        points = np.loadtxt(file_path, delimiter=' ')
        point_clouds.append(torch.tensor(points, dtype=torch.float32))
    return point_clouds, file_paths

point_clouds, file_paths = load_point_clouds(opt.datafolder)
print(f"Found {len(point_clouds)} point clouds.")

# Load the PointNet model
classifier = PointNetDenseCls(k=opt.num_classes, feature_transform=False)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()
classifier.cuda()

# Function to visualize point cloud with predicted labels
def visualize_point_cloud(points, labels, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = plt.get_cmap("tab10")(labels / 10)[:, :3]  # Use tab10 colormap for visualization
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_path, pcd)  # Save visualization as a point cloud file

# Process point clouds in batches and perform inference
for i in range(0, len(point_clouds), opt.batchSize):
    points_batch = point_clouds[i:i + opt.batchSize]
    file_paths_batch = file_paths[i:i + opt.batchSize]
    points_batch = torch.stack(points_batch).cuda()
    points_batch = points_batch.transpose(2, 1)

    with torch.no_grad():
        pred, _, _ = classifier(points_batch)

    pred_choice = pred.data.max(2)[1].cpu().numpy()  # Predicted labels

    # Visualize each point cloud with its predicted labels
    for j, (points_np, pred_choice_np) in enumerate(zip(points_batch.cpu().numpy(), pred_choice)):
        visualize_point_cloud(points_np.transpose(), pred_choice_np, os.path.join(opt.outputfolder, f'visualization_{i+j}.pcd'))

print(f"Visualizations saved in {opt.outputfolder}.")

