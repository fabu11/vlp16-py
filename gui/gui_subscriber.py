from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from threading import Lock
from PySide6.QtCore import Signal, QObject
import pandas as pd
import tensorflow as tf
import numpy as np
import os

class VelodyneSignals(QObject):
    data = Signal(object)  

class VelodyneSubscriber(Node):
    def __init__(self, qt_signals):
        super().__init__('velodyne_subscriber')
        
        # ROS2 SUBSCIRBER
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',  
            self.pointcloud_callback,
            10)
        
        # VOXEL CLASSIFIER
        #self.label_to_location = {0: 'frost_large', 1: 'capstone_lab', 2: 'baker_not_curved', 3: 'chumash', 4: 'embedded_lab', 5: 'open_lab'}
        self.label_to_location = {0: 'open_lab', 1: 'embedded_lab', 2: 'baker_not_curved', 3: 'chumash', 4: 'baker_hallway', 5: 'cs_classroom', 6: 'capstone_lab', 7: 'frost_large', 8: 'e_hall', 9: 'cs_hall'}
        self.voxel_grid_size = (64, 64, 64)
        self.model_filepath = "../voxel_classifier.keras"
        try: 
            self.model = tf.keras.models.load_model(self.model_filepath) # pyright: ignore
        except FileNotFoundError:
            errmsg = f"Classifier file not found! {self.model_filepath}"
            print(errmsg)
            exit(1)
        
        # QT SIGNALS
        self.qt_signals = qt_signals
        self.lock = Lock()
        self.point_count = 0

    def pointcloud_callback(self, msg):
        with self.lock:
            # Read points from message
            # Convert generator to list of tuples
            points = read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True) # pyright: ignore
            df = pd.DataFrame(points, columns=['x', 'y', 'z']) # pyright: ignore
            # Check for empty scans
            if df.empty:
                self.get_logger().warn("Received an empty LiDAR scan.")
                return
            # Convert DataFrame to voxel grid
            voxel_grid = self.pointcloud_to_voxel(df)
            # Reshape for model input (batch and channel dimensions)
            voxel_input = voxel_grid[np.newaxis, ..., np.newaxis]
            # Run inference
            prediction = self.model.predict(voxel_input)[0]
            predicted_label = int(np.argmax(prediction))
            predicted_location = self.label_to_location.get(predicted_label, "Unknown")
            confidence = prediction[predicted_label]

            obj = {
                    "points": points,
                    "pred": f"Predicted: {predicted_location} (#{predicted_label}) - ({confidence}%)",
                    "proba": 0
                    }
            self.qt_signals.data.emit(obj)  # Send original point cloud for visualization

    def pointcloud_to_voxel(self, df):
        """Convert pandas DataFrame (LiDAR scan) into a voxel grid."""

        min_bounds = df[['x', 'y', 'z']].min().values
        max_bounds = df[['x', 'y', 'z']].max().values

        # Compute voxel size
        voxel_size = (max_bounds - min_bounds) / np.array(self.voxel_grid_size)
        voxel_size[voxel_size == 0] = 1e-6  # Prevent division by zero

        # Initialize empty voxel grid
        voxel_grid = np.zeros(self.voxel_grid_size, dtype=np.float32)

        # Compute voxel indices
        indices = ((df[['x', 'y', 'z']].values - min_bounds) / voxel_size).astype(int)
        indices = np.clip(indices, 0, np.array(self.voxel_grid_size) - 1)

        # Set occupied voxels to 1
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        return voxel_grid
