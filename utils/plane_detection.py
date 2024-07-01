import open3d as o3d
import numpy as np
import random
import time

# Utility functions

def ReadPlyPoint(fname):
    """ Read points from a PLY file.

    Args:
        fname (str): Path to the PLY file.

    Returns:
        np.ndarray: N x 3 array of point coordinates.
    """
    pcd = o3d.io.read_point_cloud(fname)
    return PCDToNumpy(pcd)

def NumpyToPCD(xyz):
    """ Convert a numpy array to Open3D point cloud.

    Args:
        xyz (np.ndarray): N x 3 array of point coordinates.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def PCDToNumpy(pcd):
    """ Convert an Open3D point cloud to numpy array.

    Args:
        pcd (o3d.geometry.PointCloud): Open3D point cloud object.

    Returns:
        np.ndarray: N x 3 array of point coordinates.
    """
    return np.asarray(pcd.points)

def RemoveNoiseStatistical(pc, nb_neighbors=20, std_ratio=2.0):
    """ Remove noise from a point cloud using statistical outlier removal.

    Args:
        pc (np.ndarray): N x 3 array of point coordinates.
        nb_neighbors (int, optional): Number of neighbors to consider. Defaults to 20.
        std_ratio (float, optional): Standard deviation ratio. Defaults to 2.0.

    Returns:
        np.ndarray: Cleaned N x 3 array of point coordinates.
    """
    pcd = NumpyToPCD(pc)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return PCDToNumpy(cl)

def PlaneRegression(points, threshold=0.01, init_n=3, iter=1000):
    """ Fit a plane to points using RANSAC.

    Args:
        points (np.ndarray): N x 3 array of point coordinates.
        threshold (float, optional): RANSAC distance threshold. Defaults to 0.01.
        init_n (int, optional): Initial number of inliers. Defaults to 3.
        iter (int, optional): Maximum iterations. Defaults to 1000.

    Returns:
        tuple: Plane equation coefficients (4 x 1 array), inlier indices (list).
    """
    pcd = NumpyToPCD(points)
    w, index = pcd.segment_plane(threshold, init_n, iter)
    return w, index

def DrawResult(points, colors):
    """ Visualize points with assigned colors.

    Args:
        points (np.ndarray): N x 3 array of point coordinates.
        colors (np.ndarray): N x 3 array of RGB colors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def edge_points(planes_tuple):
    # Subsitute the plane points of one plane to the plane equation of other planes.
    # Points that are in other planes are considered as edge points. (threshold = 0.01)
    edge_points = []
    for i in range(len(planes_tuple)):
        plane = planes_tuple[i][1]
        plane_eq = planes_tuple[i][0]
        for j in range(len(planes_tuple)):
            if i != j:
                other_plane = planes_tuple[j][1]
                for point in plane:
                    if abs(np.dot(plane_eq[:3], point) + plane_eq[3]) < 0.001:
                        edge_points.append(point)
    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    print('Number of edge points:', len(edge_points)) 

    return np.array(edge_points), edge_pcd

def DetectMultiPlanes(points, min_ratio=0.1, threshold=0.01, iterations=10, min_points_per_plane=10):
    """ Detect multiple planes from given point clouds.

    Args:
        points (np.ndarray): N x 3 array of point coordinates.
        min_ratio (float, optional): Minimum ratio of remaining points to stop detection. Defaults to 0.1.
        threshold (float, optional): RANSAC distance threshold. Defaults to 0.01.
        iterations (int, optional): Maximum iterations for RANSAC. Defaults to 10.
        min_points_per_plane (int, optional): Minimum points per plane to consider it valid. Defaults to 10.

    Returns:
        list: List of tuples containing plane equation coefficients (np.ndarray) and plane points (np.ndarray).
    """
    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(target, threshold=threshold, init_n=3, iter=iterations)
        count += len(index)
        
        # Filter out planes with fewer points than min_points_per_plane
        if len(index) >= min_points_per_plane:
            plane_list.append((w, target[index]))
        else:
            print(f"Ignoring plane with {len(index)} points.")
        
        target = np.delete(target, index, axis=0)

    print('Number of planes detected:', len(plane_list))
    return plane_list

if __name__ == "__main__":
    # Example usage
    # Read point cloud from PCD file
    pcd = o3d.io.read_point_cloud('/home/shashank/Documents/UniBonn/Sem4/Alisha/random sample_filtered/Nr12.pcd')
    
    # Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Convert to numpy array
    points = PCDToNumpy(pcd)
    print("Loaded point cloud with {} points".format(points.shape[0]))
    
    # Pre-processing
    points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)
    
    # Detecting planes
    t0 = time.time()
    print("Detecting Planes...")
    results = DetectMultiPlanes(points, min_ratio=0.05, threshold=4, iterations=2000, min_points_per_plane=8000)
    edge_points_, edge_pcd_ = edge_points(results)
    # Set size of edge pcd points to 2
    edge_pcd_.paint_uniform_color([0, 0, 1])
    sphere_size = 0.01
    # Create sphere at each edge point
    spheres = []
    flag = 0
    for point in edge_points_:
        flag += 1
        # Every 1000 points are considered for visualization
        if flag % 1000 == 0:
                
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
            sphere.translate(point)
            spheres.append(sphere)
    o3d.visualization.draw_geometries([edge_pcd_] + spheres)

    
    

    print('Time:', time.time() - t0)
    
    # Visualize results
    planes = []
    colors = []
    for _, plane in results:
        r = random.random()
        g = random.random()
        b = random.random()
        
        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b
        
        planes.append(plane)
        colors.append(color)
    
    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)
    DrawResult(planes, colors)
