# LiDAR-Camera Extrinsic Calibrator

```python
self.init_ext=
    np.asarray([
        [ 0.00349169, 0.24438457,   0.9696721,   0.05704335],
        [-0.99996054, -0.00706836,  0.00538218,  0.12145698],
        [ 0.00816931, -0.96965262,  0.24435025, -0.06868874],
        [ 0.,          0.,          0.,          1.        ]
        ])

rotation = tf.transformations.quaternion_from_matrix(self.init_ext)
trans = self.init_ext[:3, 3]
publish_transform(translation = trans, rotation = rotation, parent_frame = "os_sensor", child_frame = "zed2_left_camera_optical_frame")


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
```