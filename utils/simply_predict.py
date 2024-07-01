# import torch
# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# from pointnet.model import PointNetDenseCls

# # Load point cloud data from a .txt file
# def load_point_cloud(file_path):
#     points = np.loadtxt(file_path, delimiter=' ')
#     return points

# # Visualize point cloud with predicted labels using Open3D
# def visualize_point_cloud_with_labels(points, labels, num_classes):
#     points = points.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
#     labels = labels.cpu().numpy().squeeze()   # Convert labels to numpy and remove batch dimension

#     # Create Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Map labels to colors
#     cmap = plt.get_cmap("tab10")  # Colormap for visualization
#     colors = cmap(labels / (num_classes - 1))[:, :3]  # Normalize labels and get RGB colors
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([pcd])

# # Load point cloud from file
# point_cloud_file = '/home/shashank/Documents/UniBonn/Sem4/Alisha/data/dlrvdata/points/Nr1.txt'
# points_np = load_point_cloud(point_cloud_file)

# # Convert numpy array to Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_np)

# # Visualize the original point cloud
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# # Remove outliers using statistical outlier removal
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# # Extract the outlier removed point cloud
# points_np_clean = np.asarray(cl.points)

# # Visualize the cleaned point cloud
# pcd_clean = o3d.geometry.PointCloud()
# pcd_clean.points = o3d.utility.Vector3dVector(points_np_clean)
# o3d.visualization.draw_geometries([pcd_clean], window_name="Cleaned Point Cloud")

# # Segment foreground points based on distance threshold (adjust threshold as needed)
# distance_threshold = 0.3
# plane_model, inliers = pcd_clean.segment_plane(distance_threshold, ransac_n=3, num_iterations=1000)
# inlier_cloud = pcd_clean.select_by_index(inliers)
# outlier_cloud = pcd_clean.select_by_index(inliers, invert=True)

# # Extract the foreground points
# points_np_foreground = np.asarray(inlier_cloud.points)

# # Visualize the foreground point cloud
# pcd_foreground = o3d.geometry.PointCloud()
# pcd_foreground.points = o3d.utility.Vector3dVector(points_np_foreground)
# o3d.visualization.draw_geometries([pcd_foreground], window_name="Foreground Point Cloud")

# # Load the trained model
# num_classes = 4  # Adjusted to match the number of classes in the checkpoint
# model_path = '/home/shashank/Documents/UniBonn/Sem4/pointnet.pytorch/utils/seg/seg_model_Chair_3.pth'

# # Initialize the model
# model = PointNetDenseCls(k=num_classes, feature_transform=False)
# model.load_state_dict(torch.load(model_path))
# model.eval()

# # Convert foreground point cloud to torch tensor and add batch dimension
# point_cloud_tensor = torch.tensor(points_np_foreground, dtype=torch.float32).unsqueeze(0)

# # Move point cloud tensor to GPU if available
# if torch.cuda.is_available():
#     model.cuda()
#     point_cloud_tensor = point_cloud_tensor.cuda()

# # Predict per-point class labels
# with torch.no_grad():
#     point_cloud_tensor = point_cloud_tensor.transpose(2, 1)  # Transpose to match model input shape
#     pred, _, _ = model(point_cloud_tensor)
#     pred_choice = pred.data.max(2)[1]  # Get predicted labels
#     print(f"Predicted labels shape: {pred_choice.shape}")

# # Convert predicted labels to numpy for easier manipulation/visualization
# predicted_labels = pred_choice.cpu().numpy().squeeze()

# # Print the predicted labels
# print("Predicted per-point class labels:", predicted_labels)

# # Visualize the point cloud with predicted labels
# visualize_point_cloud_with_labels(point_cloud_tensor.transpose(2, 1), pred_choice, num_classes)


##########################
##########################



import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pointnet.model import PointNetDenseCls

# Load point cloud data from a .txt file
def load_point_cloud(file_path):
    points = np.loadtxt(file_path, delimiter=' ')
    return points

# Visualize point cloud with predicted labels using Open3D
def visualize_point_cloud_with_labels(points, labels, num_classes):
    points = points.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
    labels = labels.cpu().numpy().squeeze()   # Convert labels to numpy and remove batch dimension

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Map labels to colors
    cmap = plt.get_cmap("tab10")  # Colormap for visualization
    colors = cmap(labels / (num_classes - 1))[:, :3]  # Normalize labels and get RGB colors
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Load point cloud from file
# point_cloud_file = '/home/shashank/Documents/UniBonn/Sem4/Alisha/data/dlrvdata/points/Nr1.txt'
point_cloud_file = '/home/shashank/Documents/UniBonn/Sem4/Alisha/random sample_txt/Nr59.txt'
points_np = load_point_cloud(point_cloud_file)

# Convert numpy array to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_np)

# Visualize the original point cloud
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# Remove outliers using statistical outlier removal
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Extract the outlier removed point cloud
points_np_clean = np.asarray(cl.points)

# Visualize the cleaned point cloud
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points_np_clean)
o3d.visualization.draw_geometries([pcd_clean], window_name="Cleaned Point Cloud")

# Load the trained model
num_classes = 4  # Adjusted to match the number of classes in the checkpoint
model_path = '/home/shashank/Documents/UniBonn/Sem4/pointnet.pytorch/utils/seg/seg_model_Chair_3.pth'

# Initialize the model
model = PointNetDenseCls(k=num_classes, feature_transform=False)
model.load_state_dict(torch.load(model_path))
model.eval()

# Convert cleaned point cloud to torch tensor and add batch dimension
point_cloud_tensor = torch.tensor(points_np_clean, dtype=torch.float32).unsqueeze(0)

# Move point cloud tensor to GPU if available
if torch.cuda.is_available():
    model.cuda()
    point_cloud_tensor = point_cloud_tensor.cuda()

# Predict per-point class labels
with torch.no_grad():
    point_cloud_tensor = point_cloud_tensor.transpose(2, 1)  # Transpose to match model input shape
    pred, _, _ = model(point_cloud_tensor)
    pred_choice = pred.data.max(2)[1]  # Get predicted labels
    print(f"Predicted labels shape: {pred_choice.shape}")

# Convert predicted labels to numpy for easier manipulation/visualization
predicted_labels = pred_choice.cpu().numpy().squeeze()

# Print the predicted labels
print("Predicted per-point class labels:", predicted_labels)

# Visualize the point cloud with predicted labels
visualize_point_cloud_with_labels(point_cloud_tensor.transpose(2, 1), pred_choice, num_classes)



#########################
#########################



# import pyntcloud
# import numpy as np
# import torch
# import open3d as o3d
# import matplotlib.pyplot as plt
# from pointnet.model import PointNetDenseCls

# # Load point cloud data from a file (assuming it's in PLY format)
# point_cloud_file = '/path/to/your/point_cloud.ply'
# cloud = pyntcloud.PyntCloud.from_file(point_cloud_file)

# # Remove outliers based on the distance from the centroid
# centroid = cloud.xyz.mean(axis=0)
# distances = np.linalg.norm(cloud.xyz - centroid, axis=1)
# outlier_mask = distances < np.percentile(distances, 95)  # Adjust percentile as needed
# cloud = cloud.extract_info("points", outlier_mask)

# # Estimate edges
# edges = cloud.add_scalar_field("edges", k=10)  # Adjust `k` parameter as needed

# # Convert point cloud to torch tensor
# points = torch.tensor(cloud.points.values, dtype=torch.float32).unsqueeze(0)

# # Load the trained model
# num_classes = 4  # Adjusted to match the number of classes in the checkpoint
# model_path = '/path/to/your/seg_model.pth'

# # Initialize the model
# model = PointNetDenseCls(k=num_classes, feature_transform=False)
# model.load_state_dict(torch.load(model_path))
# model.eval()

# # Move model and point cloud to GPU if available
# if torch.cuda.is_available():
#     model.cuda()
#     points = points.cuda()

# # Predict per-point class labels
# with torch.no_grad():
#     points = points.transpose(2, 1)  # Transpose to match model input shape
#     pred, _, _ = model(points)
#     pred_choice = pred.data.max(2)[1]  # Get predicted labels

# # Convert predicted labels to numpy for easier manipulation/visualization
# predicted_labels = pred_choice.cpu().numpy().squeeze()



# def visualize_point_cloud_with_labels_and_edges(points, labels, num_classes, edges):
#     points = points.transpose()  # Adjust to correct shape for Open3D visualization
#     labels = labels.cpu().numpy().squeeze()   # Convert labels to numpy and remove batch dimension

#     # Create Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Map labels to colors
#     cmap = plt.get_cmap("tab10")  # Colormap for visualization
#     colors = cmap(labels / (num_classes - 1))[:, :3]  # Normalize labels and get RGB colors
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Visualize the point cloud with edges
#     lines = []
#     for edge in edges.points.values:
#         lines.append([edge[0], edge[1]])

#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(points),
#         lines=o3d.utility.Vector2iVector(lines),
#     )

#     # Create visualization
#     o3d.visualization.draw_geometries([pcd, line_set])


# # Visualize the point cloud with predicted labels and edges
# points_np = points.cpu().numpy().squeeze().transpose()
# visualize_point_cloud_with_labels_and_edges(points_np, pred_choice, num_classes, edges)
