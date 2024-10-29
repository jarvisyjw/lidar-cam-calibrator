#!/usr/bin/env python

# solve the import issue
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header
import open3d as o3d
import tf
from geometry_msgs.msg import TransformStamped
import math
from sklearn.decomposition import PCA

from utils import load_from_config, create_point_cloud2, segment_plane_ransac, publish_transform, numpy_to_pointcloud, project_pixels_to_3d, est_board_plane


import tf.transformations


class ChessboardDetector:
    '''Detect a chessboard in a stereo image pair
    and compute the 3D coordinates of the
    chessboard corners. Segment the plane of the
    chessboard and filter the point cloud to only
    include points on the plane.
    '''
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('stereo_chessboard_detector', log_level=rospy.DEBUG, anonymous=True)
        
        pwd = os.path.dirname(__file__)
        config = load_from_config(os.path.join(pwd, 'config/RPF_camera_lidar.yaml'))
        
        # Chessboard parameters
        chessboard_config = config['chessboard']
        h = chessboard_config['height']
        w = chessboard_config['width']        
        self.chessboard_size = (h, w)  # Change this based on your chessboard
        
        self.init_ext = np.asarray(config['init_ext'])
        
        self.bridge = CvBridge()

        # Subscribers
        rgb_image_sub = message_filters.Subscriber("/zed2/zed_node/rgb/image_rect_color/compressed", CompressedImage)
        depth_image_sub = message_filters.Subscriber("/zed2/zed_node/depth/depth_registered", Image)
        camera_info_sub = message_filters.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo)
        point_cloud_sub = message_filters.Subscriber('/ouster/points', PointCloud2)

        # Publishers
        self.filtered_cloud_pub = rospy.Publisher('/detected_point_cloud', PointCloud2, queue_size=10)
        self.image_chessboard_image_pub = rospy.Publisher("/image_chessboard_image", Image, queue_size=10)
        self.image_chessboard_corners_pc_pub = rospy.Publisher("/image_chessborad_pointcloud", PointCloud2, queue_size=10)

        # Synchronize left and right image topics
        ts = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub, camera_info_sub, point_cloud_sub], 100, 0.1)
        ts.registerCallback(self.callback)
        
        # tf listener
        self.listener = tf.TransformListener()
        
        # Set range limits for pointcloud filtering
        self.x_min, self.x_max = 0.2, 6.0
        self.y_min, self.y_max = -2.5, 2.5
        self.z_min, self.z_max = -0.25, 1.8
        rotation_matrix = np.array([[-1, 0, 0],
                        [0, -1, 0],
                     [0, 0, 1]])
        homo_rotation = np.eye(4)
        # homo_rotation[:3, :3] = rotation_matrix
        print(homo_rotation.shape)
        # rotation = tf.transformations.quaternion_from_matrix(homo_rotation)
        # init_ext = np.linalg.inv(self.init_ext)
        # print(init_ext)
        rotation = tf.transformations.quaternion_from_matrix(self.init_ext)
        # print(self.init_ext)
        trans = self.init_ext[:3, 3]
        print(trans)
        
        publish_transform(translation = trans, rotation = rotation, parent_frame = "os_sensor", child_frame = "zed2_left_camera_optical_frame")
        # publish_transform(translation = (0.0, 0.0, 0.0), rotation = (0, 0, 0, 1), parent_frame = "world", child_frame = "os_sensor")

        rospy.loginfo("Stereo Chessboard Detector initialized.")
        
    def project_points_onto_plane(self, points, normal, d):
        """ Project 3D points onto a plane with a given normal and offset d. """
        projected_points = []
        normal = np.array(normal)
        for p in points:
            p = np.array(p)
            distance = (np.dot(normal, p) + d) / np.linalg.norm(normal)
            projection = p - distance * normal
            projected_points.append(projection)
        return np.array(projected_points)
    
    def project_onto_2d_basis(self, point, v1, v2):
        u = np.dot(point, v1)
        v = np.dot(point, v2)
        return np.array([u, v])
    
    def convert_2d_to_3d(self, u, v, v1, v2, point_on_plane):
            return u * v1 + v * v2 + point_on_plane


    # def point_cloud_callback(self, point_cloud_msg):
    #     # Extract points from the point cloud
    #     filtered_points = []

    #     for point in pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
    #         x, y, z = point[:3]

    #         # Filter points based on x, y, z ranges
    #         # if (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max) and (self.z_min <= z <= self.z_max):
    #         #     filtered_points.append([x, y, z])
    #         if ((self.x_min <= x <= self.x_max) and (self.z_min <= z)):
    #             filtered_points.append([x, y, z])
        
    #     rospy.logdebug(f"filtered points: {filtered_points}")
    #     plane_model, inlier_cloud, outlier_cloud = segment_plane_ransac(np.array(filtered_points))

    #     # Create the header for the new PointCloud2 message
    #     header = Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = point_cloud_msg.header.frame_id

    #     # Create a new PointCloud2 message with the filtered points
    #     # filtered_cloud_msg = self.create_point_cloud2(header, filtered_points)
    #     filtered_cloud_msg = create_point_cloud2(header, inlier_cloud.points)

    #     # find vertexes of the plane
    #     # project the filtered points onto the plane
    #     # projected_points = self.project_points_onto_plane(np.asarray(inlier_cloud.points), plane_model[:3], plane_model[3])
        
    #     # normal = plane_model[:3]
    #     # # Define two orthogonal vectors on the plane
    #     # v1 = np.array([-normal[1], normal[0], 0])
    #     # v1 /= np.linalg.norm(v1)  # Normalize the first vector
    #     # v2 = np.cross(normal, v1)  # Compute the second vector by crossing with the normal
        
    #     # points_2d = np.array([self.project_onto_2d_basis(p, v1, v2) for p in projected_points])

    #     # # Find the minimum and maximum u and v coordinates
    #     # u_min, u_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
    #     # v_min, v_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])

    #     # # The four corners of the rectangle in 2D
    #     # vertices_2d = np.array([
    #     #     [u_min, v_min],
    #     #     [u_min, v_max],
    #     #     [u_max, v_min],
    #     #     [u_max, v_max]
    #     # ])
        
    #     # Assume we use the centroid of the projected points as a point on the plane
    #     # centroid = np.mean(projected_points, axis=0)

    #     # # Convert each 2D vertex back into 3D space
    #     # vertices_3d = np.array([self.convert_2d_to_3d(u, v, v1, v2, centroid) for u, v in vertices_2d])

    #     # # Visualize the rectangle using the four vertices
    #     # rect_pcd = o3d.geometry.PointCloud()
    #     # rect_pcd.points = o3d.utility.Vector3dVector(vertices_3d)

    #     # Visualize the original point cloud and the rectangle
    #     # o3d.visualization.draw_geometries([inlier_cloud, rect_pcd])
        
    #     # Publish the filtered point cloud
    #     self.filtered_cloud_pub.publish(filtered_cloud_msg)
    #     rospy.loginfo("Filtered point cloud published.")
        
    #     return inlier_cloud
    
    
    def callback(self, img_msg, depth_msg, camera_info_msg, raw_pointcloud_msg):
        # get camera intrinsic matrix
        K = np.array(camera_info_msg.K)
        # rospy.logdebug(f"camera intrinsics {K}")
        
        try:
            rospy.loginfo("Received images.")
            # compressed image to cv2 image
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # depth image to cv2 image, the pixel value is the distance to the camera (meters)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return
        
        self.detect_chessboard(image_np, depth_image, K, raw_pointcloud_msg)
    
    
    def project_points_2d(self, points, K, depth):
        """
        Porject 2D points to 3D using camera intrinsic matrix.
        """
        # Convert the points to homogeneous coordinates
        points_homogeneous = np.hstack((points, depth.reshape(-1,1) * np.ones((points.shape[0], 1))))
        rospy.logdebug(f"points homogeneous shape: {points_homogeneous.shape}")
        # Project the points to 3D using the camera intrinsic matrix
        points_3d = np.linalg.inv(K.reshape(3,3)).dot(points_homogeneous.T).T
        rospy.logdebug(f"points 3d shape: {points_3d.shape}")
        return points_3d
        

    def detect_chessboard(self, rgb_image, depth_image, K, pc_msg):
        # Convert both images to grayscale
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_image, self.chessboard_size, None)
        depth_array = np.array(depth_image, dtype=np.float32)
        
        if ret:
            rospy.loginfo("Chessboard detected.")
            cv2.drawChessboardCorners(rgb_image, self.chessboard_size, corners, ret)
            corners_reshaped = np.array(np.squeeze(corners.reshape((-1,1,2))))
            corners_depth = []
            for c in corners_reshaped:
                x, y = c
                depth = depth_array[int(y), int(x)]
                corners_depth.append(depth)
                
            corners_depth = np.array(corners_depth, dtype=np.float32)
            z_avg = np.nanmean(corners_depth)
            z_min = np.nanmin(corners_depth)
            z_max = np.nanmax(corners_depth)
            self.image_chessboard_image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_image, "bgr8"))
            
            corners_3d = project_pixels_to_3d(corners_reshaped, corners_depth, K)
            chessboard_corners = est_board_plane(corners_3d, 1, 2)
            corners_x = corners_3d[:,0]
            corners_y = corners_3d[:,1]
            x_min = np.nanmin(corners_x)
            x_max = np.nanmax(corners_x)
            y_min = np.nanmin(corners_y)
            y_max = np.nanmax(corners_y)
            
            corners_pc_msg = numpy_to_pointcloud(corners_3d, "zed2_left_camera_optical_frame")
            self.image_chessboard_corners_pc_pub.publish(corners_pc_msg)
            rospy.loginfo("Chessboard point cloud published.")
            
            self.z_min = -y_min - 0.5
            self.z_max = -y_max + 0.5
            self.y_min = -x_min - 0.5
            self.y_max = -x_max + 0.5
            self.x_min = z_min - 0.2
            self.x_max = z_max + 0.2
                        
            filtered_points = []
            for point in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point[:3]

                # Filter points based on x, y, z ranges
                # if (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max) and (self.z_min <= z <= self.z_max):
                #     filtered_points.append([x, y, z])
                if ((self.x_min <= x <= self.x_max) and (self.z_min <= z)):
                    filtered_points.append([x, y, z])
            
            rospy.logdebug(f"filtered points: {filtered_points}")
            plane_model, inlier_cloud, outlier_cloud = segment_plane_ransac(np.array(filtered_points))

            # Create the header for the new PointCloud2 message
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = pc_msg.header.frame_id

            # Create a new PointCloud2 message with the filtered points
            # filtered_cloud_msg = self.create_point_cloud2(header, filtered_points)
            filtered_cloud_msg = create_point_cloud2(header, inlier_cloud.points)
            self.filtered_cloud_pub.publish(filtered_cloud_msg)
            rospy.loginfo("Filtered point cloud published.")
            
            # corners_pcd = o3d.geometry.PointCloud()

            # # Assign points to the PointCloud
            # corners_pcd.points = o3d.utility.Vector3dVector(corners_3d)
            
            # try:
            #     (trans, rot) = self.listener.lookupTransform('os_sensor', 'zed2_left_camera_optical_frame', rospy.Time(0))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     rospy.logerr("Transform not available")
            #     pass
            
            # rot_matrix = tf.transformations.quaternion_matrix(rot)
            # rot_matrix[0:3, 3] = trans
            
                        
            # self.compute_ICP(corners_pcd, inliers_cloud, trans_init=self.init_ext)
            
            # plane_range = np.concatenate(x_min,x_max,y_min, y_max, z_min,z_max)
            
            # return plane_range
            
            
            # return corners_reshaped, corners_depth

if __name__ == '__main__':
    try:
        # Initialize the stereo chessboard detector
        detector = ChessboardDetector()

        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Stereo chessboard detector node terminated.")
    finally:
        cv2.destroyAllWindows()