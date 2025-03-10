import math, sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf


class MinimalSubscriber(Node):
    def __init__(self, m, e):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10)
        self.subscription

        # Model / Encoder
        self.model = m
        self.encoder = e

        self.shortest_distance = {               
                        "range": 9999,
                        "index": 0,
                        }

    def calculate_range(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def pre_process(self, points):
        # Convert the generator to a numpy array.
        df = pd.DataFrame(points[1:], columns=['x', 'y', 'z', 'intensity']) # pyright: ignore
        df.loc[df.isnull().any(axis=1)] = [0, 0, 0, 0]
        target_rows = 29183
        current_rows = len(df)
        if(current_rows < target_rows):
            rows_to_add = target_rows - current_rows
            padding_df = pd.DataFrame(0, index=range(rows_to_add), columns=df.columns)
            df = pd.concat([df, padding_df], ignore_index=True)
        elif current_rows > target_rows:
            df = df.iloc[:target_rows]
        return np.array(df).flatten()

    def reorder_points(self, points):
        points = np.array(list(points))
        
        for i, p in enumerate(points):
            if(any(math.isnan(item) for item in p)):
                points[i] = (0, 0, 0, 0)
            c = self.calculate_range(p[0], p[1], p[2]) 
            if c < self.shortest_distance["range"]:
                self.shortest_distance = {
                    "range": c,
                    "index": i,
                }
        
        # Fix the concatenate issue by using a tuple
        points = np.concatenate((points[self.shortest_distance["index"]:], points[:self.shortest_distance["index"]]))
        return points.tolist()

    def listener_callback(self, msg):
        # Extract the point cloud from the ROS message

        # Read points from /velodyne_points msg
        points = read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False) # pyright: ignore

        # pre-procecss points st they are rearranged (shortest distance first) and turn NaN to 0
        points = self.reorder_points(points)
        points = self.pre_process(points)
        points = np.expand_dims(points, axis=0)

        prediction = self.model.predict(points)
        predicted_labels = self.encoder.inverse_transform(prediction)

        print(f"Prediction: {predicted_labels}")



def _get_flag_val(f):
    args = sys.argv[1:]
    if f in args:
        m_index = args.index(f)
        val = args[m_index + 1]
        if not val:
            print(f"flag: {f} has usage error")
        return val
        


def load_model_and_encoder():
    # Default paths
    model_path = './lidar_classifier.keras'
    encoder_path = './label_encoder.pkl'

    m_f = _get_flag_val('-m')
    e_f = _get_flag_val('-e')

    if(m_f):
        model_path = m_f
    if(e_f):
        encoder_path = '-e'

    # load tf model
    loaded_model = tf.keras.models.load_model(model_path) # pyright: ignore

    # load encoder_path 
    with open(encoder_path, 'rb') as f:
        loaded_encoder = pickle.load(f)
        
    return loaded_model, loaded_encoder



def main(args=None):
    print("start")
    rclpy.init(args=args)
    m, e = load_model_and_encoder()
    minimal_subscriber = MinimalSubscriber(m, e)
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
