import yaml
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import rospy
import tf



def load_from_config(yaml_file):
    '''
    Load a specific key field from a YAML file.
    '''
    with open(yaml_file, 'r') as file:
        # Load the YAML file
        data = yaml.safe_load(file)
        return data
    

def segment_plane_ransac(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Segment the largest plane from a set of points using RANSAC.
    
    Args:
        points: A numpy array of shape (N, 3) representing the point cloud.
        distance_threshold: Maximum distance a point can have to be considered part of the plane.
        ransac_n: Minimum number of points needed to estimate a plane (typically 3 for a plane).
        num_iterations: Number of RANSAC iterations to run.
    
    Returns:
        plane_model: Coefficients [A, B, C, D] of the plane equation Ax + By + Cz + D = 0.
        inliers: The indices of points that belong to the segmented plane.
        outliers: The indices of points that do not belong to the plane.
    """
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Apply RANSAC to segment the plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    
    # Extract inlier and outlier points
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    return plane_model, inlier_cloud, outlier_cloud
    

def create_point_cloud2(header, points):
    """
    Create a PointCloud2 message from a list of points.
    Each point is assumed to be of the format [x, y, z].
    """
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    # Create PointCloud2 message from the points
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg


# def visualize_depth_image(self, depth_image, K):
    #     pass
    #     # for i in range(0, depth_image.shape[0]):
    #     #     for j in range(0, depth_image.shape[1]):
    #     #         depth_image[i, j] = depth_image[i, j] 
        
    #     # Normalize the depth image to display it better
    #     # depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    #     # Convert the normalized depth image to a format suitable for display
    #     # depth_display = np.uint8(depth_normalized)
    #     # rospy.loginfo(depth_display)
    #     # Display the depth image
    #     # cv2.imshow('Depth Image', depth_display)
    #     # cv2.waitKey(1)

 
def compute_ICP(self, source_pc, target_pc, voxel_size = 0.02, trans_init = np.identity(4)):
    
    radius_normal = voxel_size * 2
    # estimate norm
    source_pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    target_pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    
    # trans_init = np.identity(4)
    rospy.logdebug(f"init trans: {trans_init}")
    distance_threshold = voxel_size * 1.5
    reg_result = o3d.pipelines.registration.registration_icp(
    source_pc, target_pc, distance_threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # print("Has converged:", reg_result.converged)
    print("Fitness:", reg_result.fitness)
    print("RMSE:", reg_result.inlier_rmse)
    print("Transformation matrix:")
    print(reg_result.transformation)
    
    source_pc.transform(reg_result.transformation)

    o3d.visualization.draw_geometries([source_pc, target_pc])
    

def publish_transform(translation = (1.0, 0.0, 0.5), rotation = (0, 0, 0, 1), parent_frame = "world", child_frame = "base_link"):
        
        # Create a TransformBroadcaster object
        br = tf.TransformBroadcaster()

        # Publishing rate (10 Hz)
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Get current time
            current_time = rospy.Time.now()

            # Define the translation (x, y, z) and rotation (as a quaternion)
            # translation = (1.0, 0.0, 0.5)  # Translation in meters (e.g., x=1, y=0, z=0.5)
            # rotation = tf.transformations.quaternion_from_euler(0, 0, math.radians(90))  # Rotation in radians
            
            # Publish the transform between "base_link" and "camera_link"
            # rotation = tf.transformations.matrix_from_quater(rotation)
            
            br.sendTransform(translation,
                            rotation,
                            current_time,
                            child_frame,  # Child frame
                            parent_frame)    # Parent frame

            # Sleep to maintain the loop rate
            rate.sleep()
            

def project_pixels_to_3d(pixels, depth_values, K):
        
        """
        Projects 2D pixel coordinates into 3D space using camera intrinsics and depth.

        :param pixels: NumPy array of shape (N, 2), where N is the number of pixels.
                    Each row is (u, v) pixel coordinates.
        :param depth_values: NumPy array of shape (N,), depth values corresponding to each pixel.
        :param K: Camera intrinsic matrix of shape (3, 3).
        :return: NumPy array of shape (N, 3), 3D points in camera coordinate frame.
        """
        # Extract intrinsics
        K = K.reshape(3,3)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Get pixel coordinates
        u = pixels[:, 0]
        v = pixels[:, 1]
        Z = depth_values

        # Compute normalized coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy

        # Compute 3D points
        X = x * Z
        Y = y * Z

        # Stack into (N, 3) array
        points_3d = np.stack((X, Y, Z), axis=-1)
        
        return points_3d

def est_board_plane(points, h, w):
    centroid = np.mean(points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    distance_threshold = 0.01  # Adjust based on your data scale
    ransac_n = 3               # Minimum number of points required to fit a plane
    num_iterations = 1000      # Number of RANSAC iterations for robust fitting

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.4f} * x + {b:.4f} * y + {c:.4f} * z + {d:.4f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])   # Paint inliers red
    # o3d.visualization.draw_geometries([inlier_cloud])


def detect_chessboard_reflec(self, reflected_image):
    pass
    # cv2.imshow('Reflected Image', reflected_image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # left_ret, left_corners = cv2.findChessboardCorners(reflected_image, self.chessboard_size, None)
    # cv2.drawChessboardCorners(reflected_image, self.chessboard_size, left_corners, left_ret)
    # self.reflec_detected_image_pub.publish(self.bridge.cv2_to_imgmsg(reflected_image, "gray"))
    # self.left_chessboard_detected_pub.publish(True)
    # left_corners_str = ', '.join([str(c) for c in left_corners])
    # self.left_chessboard_corners_pub.publish(left_corners_str)

def numpy_to_pointcloud(points_np, frame_id = "world"):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Flatten the Nx3 array into a list of tuples (x, y, z)
        point_list = points_np.tolist()
        new_point_list = []
        for point in point_list:    
            x, y, z = point
            new_point_list.append((x, y, z))
        
        # Create PointCloud2 message using sensor_msgs.point_cloud2.create_cloud
        point_cloud_msg = pc2.create_cloud(header, fields, new_point_list)
        
        return point_cloud_msg


def concatenate_images(self, left_image, right_image):
    """
    Concatenate the left and right images side by side for visualization.
    """
    # Ensure that both images have the same height
    if left_image.shape[0] != right_image.shape[0]:
        rospy.logerr("Left and right images have different heights, cannot concatenate.")
        return None

    # Concatenate the images horizontally
    concatenated_image = np.hstack((left_image, right_image))
    
    return concatenated_image

def get_depth(self, depth_image, point):
        """
        Get the depth value of a point in the depth image.
        """
        depth_array = np.array(depth_image, dtype=np.float32)
        x, y = point
        depth = depth_array[y, x]
        rospy.loginfo("Depth value at point (%d, %d): %f", x, y, depth)
        return depth