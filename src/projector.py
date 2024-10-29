#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.dirname(__file__))

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
from utils import load_from_config, segment_plane_ransac, create_point_cloud2
import cv2


class PointCloudProjectionNode:
    def __init__(self):
        
        rospy.init_node('projection_node', anonymous=True)
        
        # constants for filtering points
        self.x_min, self.x_max = 0.2, 5.0
        self.y_min, self.y_max = -2.5, 2.5
        self.z_min, self.z_max = -0.25, 5.0
        
        # Get camera intrinsic matrix K
        cam_info_sub = rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo, self.cam_info_callback)

        # Get extrinsic parameters
        pwd = os.path.dirname(__file__)
        self.ext = load_from_config(os.path.join(pwd, 'config/RPF_camera_lidar.yaml'), 'init_ext')

        # Subscribe to the PointCloud2 and image topic
        self.pointcloud_sub = message_filters.Subscriber('/ouster/points', PointCloud2)
        self.rgb_image_sub = message_filters.Subscriber("/zed2/zed_node/rgb/image_rect_color/compressed", CompressedImage)
        
        self.pointcloud_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=10)
        
        # Message filter to synchronize the two topics
        ts = message_filters.ApproximateTimeSynchronizer([self.pointcloud_sub, self.rgb_image_sub], 100, 0.1)
        ts.registerCallback(self.pointcloud_callback)
        
        # Prepare a publisher for the output image
        self.image_pub = rospy.Publisher("/projected_image", Image, queue_size=10)

        # For image conversion
        self.bridge = CvBridge()
    
    
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
    #     plane_model, inlier_cloud, outlier_cloud = self.segment_plane_ransac(np.array(filtered_points))

    #     # Create the header for the new PointCloud2 message
    #     header = Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = point_cloud_msg.header.frame_id

    #     # Create a new PointCloud2 message with the filtered points
    #     # filtered_cloud_msg = self.create_point_cloud2(header, filtered_points)
    #     filtered_cloud_msg = self.create_point_cloud2(header, inlier_cloud.points)

    
    def pointcloud_callback(self, pc_msg, img_msg):        
        # Load Image
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   
        
        # Extract points from the point cloud
        filtered_points = []

        for point in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point[:3]

            # Filter points based on x, y, z ranges
            # if (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max) and (self.z_min <= z <= self.z_max):
            #     filtered_points.append([x, y, z])
            if ((self.x_min <= x <= self.x_max) and (self.z_min <= z)):
                filtered_points.append([x, y, z])
        
        plane_model, inlier_cloud, outlier_cloud = segment_plane_ransac(np.array(filtered_points))
        
        # Create the header for the new PointCloud2 message
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = pc_msg.header.frame_id

        # Create a new PointCloud2 message with the filtered points
        # filtered_cloud_msg = self.create_point_cloud2(header, filtered_points)
        filtered_cloud_msg = create_point_cloud2(header, inlier_cloud.points)
        
        self.pointcloud_pub.publish(filtered_cloud_msg)
            
        # segmented_points = np.array(inlier_cloud.points)

        # # Create a blank depth image (for demonstration)
        # # Initialize depth image with a high value (e.g., 100m, assuming max depth is much smaller)

        # # Loop through each point in the point cloud
        # for point in segmented_points:
        #     x, y, z = point[:3]  # 3D coordinates (in world frame)

        #     # Convert to homogeneous coordinates for projection
        #     P_world = np.array([[x], [y], [z], [1]])

        #     # Apply extrinsic transformation: R * P_world + t
        #     P_camera = np.dot(self.ext, P_world)

        #     # Project the 3D point onto the 2D image plane
        #     if P_camera[2] > 0:  # Only project points in front of the camera
        #         P_image = np.dot(self.K, P_camera[:3])

        #         # Normalize by z (to convert homogeneous coordinates)
        #         u = int(P_image[0] / P_image[2])
        #         v = int(P_image[1] / P_image[2])

        #         # Check if the projected point is within the image boundaries
        #         if 0 <= u < image_np.shape[1] and 0 <= v < image_np.shape[0]:
        #             # Set the depth value in the depth image (only keep the minimum depth for each pixel)
        #             continue
        #         cv2.circle(image_np, (u, v), 2, (255, 0, 0), -1)

        # Normalize the depth values for visualization purposes (optional)
        # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the depth image to 8-bit format for visualization
        # depth_image_8bit = depth_image_normalized.astype(np.uint8)

        # Convert the depth image to a ROS Image message
        # projected_image_msg = self.bridge.cv2_to_imgmsg(image_np)

        # # Publish the depth image
        # self.image_pub.publish(projected_image_msg)
    
    
    def callback(self, pointcloud_msg, image_msg):
        # Convert PointCloud2 message to 3D points
        points = pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Create a blank image (for demonstration)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Loop through each point in the point cloud
        for point in points:
            x, y, z = point[:3]  # 3D coordinates (in world frame)

            # Convert to homogeneous coordinates for projection
            P_world = np.array([[x], [y], [z], [1]])

            # Apply extrinsic transformation: R * P_world + t
            P_camera = np.dot(self.ext, P_world)

            if P_camera[2] > 0:  # Only project points in front of the camera
                P_image = np.dot(self.K, P_camera[:3])

                # Normalize by z (to convert homogeneous coordinates)
                # reverse for the image coordinates
                u = int(P_image[1] / P_image[2])
                v = int(P_image[0] / P_image[2])

                # Check if the projected point is within the image boundaries
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    # Draw the projected point on the image
                    cv2.circle(image, (u, v), 2, (0, 255, 0), -1)

        # Convert the image to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

        # Publish the projected image
        self.image_pub.publish(image_msg)
        
        
    def cam_info_callback(self, msg):
        # Get camera intrinsic matrix K from the CameraInfo message
        self.K = np.array(msg.K).reshape(3, 3)


if __name__ == '__main__':
    try:
        PointCloudProjectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass