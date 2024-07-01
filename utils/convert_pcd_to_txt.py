import os
import open3d as o3d
import numpy as np

# Define input and output directories
input_dir = '/home/shashank/Documents/UniBonn/Sem4/Alisha/random sample'
output_dir = input_dir + '_txt'

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

def convert_pcd_to_txt(input_path, output_path):
    # Load PCD file
    pcd = o3d.io.read_point_cloud(input_path)
    
    # Get the points as a numpy array
    points = np.asarray(pcd.points)
    
    # Write points to TXT file
    with open(output_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")

# Process each PCD file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.pcd'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        convert_pcd_to_txt(input_path, output_path)
        print(f"Converted {filename} to TXT format.")
