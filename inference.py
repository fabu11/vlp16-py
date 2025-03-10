import math
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import pandas as pd
import numpy as np
import tensorflow as tf


class MinimalSubscriber(Node):
    def __init__(self, model):
        super().__init__('minimal_subscriber')

        # Subscribe to /velodyne_points topic
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10)
        self.subscription  # Prevent linter warnings

        # Load trained model
        self.model = model

        self.label_to_location = {0: 'frost_large', 1: 'capstone_lab', 2: 'baker_not_curved', 3: 'chumash', 4: 'embedded_lab', 5: 'open_lab'}

        # Define voxel grid size (must match training setup)
        self.voxel_grid_size = (64, 64, 64)

    def listener_callback(self, msg):
        """ROS2 callback to process incoming LiDAR scans."""
        # Convert PointCloud2 to DataFrame
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
        # Log the result
        self.get_logger().info(f"Predicted Location: {predicted_location} (Label {predicted_label}), Confidence: {confidence:.2f}")

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


def _get_flag_val(f):
    """Utility function to parse command-line flags."""
    args = sys.argv[1:]
    if f in args:
        idx = args.index(f)
        if idx + 1 < len(args):
            return args[idx + 1]
        print(f"Flag {f} requires a value.")
    return None


def load_model():
    """Load trained TensorFlow model from disk."""
    model_path = _get_flag_val('-m') or './voxel_classifier.keras'
    return tf.keras.models.load_model(model_path) # pyright: ignore


def main(args=None):
    """Main function to initialize ROS2 node."""
    print("Starting ROS2 LiDAR Classifier...")
    rclpy.init(args=args)
    model = load_model()
    minimal_subscriber = MinimalSubscriber(model)
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


