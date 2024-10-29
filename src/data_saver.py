#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
# import keyboard
import os
import cv2
# from inputs import get_key
import pygame

class DataSaver:
    def __init__(self):
        rospy.init_node('data_saver', anonymous=True)

        # Create a directory to save the data
        self.save_dir = rospy.get_param('~save_dir', './saved_data')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.image_callback)
        self.pointcloud_sub = rospy.Subscriber('/ouster/points', PointCloud2, self.pointcloud_callback)

        self.current_image = None
        self.current_pointcloud = None
        
        self.indx = 0

        rospy.loginfo("DataSaver node initialized. Press 's' to save data.")

    def image_callback(self, msg):
        self.current_image = msg

    def pointcloud_callback(self, msg):
        self.current_pointcloud = msg

    def save_data(self):
        if self.current_image and self.current_pointcloud:
            # Save image
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='bgr8')
            image_filename = os.path.join(self.save_dir, "img",f"{self.indx:05d}.png")
            if not os.path.exists(os.path.dirname(image_filename)):
                os.makedirs(os.path.dirname(image_filename))
            cv2.imwrite(image_filename, cv_image)

            # Save point cloud (you can modify the saving method as needed)
            pc_filename = os.path.join(self.save_dir, "pcd", f"{self.indx:05d}.pcd")
            if not os.path.exists(os.path.dirname(pc_filename)):
                os.makedirs(os.path.dirname(pc_filename))
            with open(pc_filename, 'wb') as f:
                f.write(self.current_pointcloud.data)  # Save raw data for simplicity

            rospy.loginfo(f"Saved image to {image_filename} and point cloud to {pc_filename}")
            self.indx += 1
            
        else:
            rospy.logwarn("No data to save.")

    def run(self):
        pygame.init()
        WIDTH=600
        HEIGHT=480
        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
        while not rospy.is_shutdown():
           for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        print(f'save data')
                        self.save_data()

if __name__ == '__main__':
    data_saver = DataSaver()
    data_saver.run()
