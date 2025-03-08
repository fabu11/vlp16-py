# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# must have ros sourced in bash

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import pandas as pd
import pickle
from collections import Counter

with open("./lidar_classifier.sav", "rb") as f:
    loaded_model = pickle.load(f)

LOADED_MODEL = loaded_model
MAX_POINTS = 29183 

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        # Extract the point cloud from the ROS message
        points = read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False)
        
        # Convert the points into a Pandas DataFrame
        df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity'])

        # Handle NaNs by replacing rows with [999, 999, 999, 999]
        df.loc[df.isna().any(axis=1)] = [999, 999, 999, 999]
        
        # Trim excess points if too many
        df = df.iloc[:MAX_POINTS]
        
        # Pad with zeros if the number of points is less than MAX_POINTS
        pad_width = MAX_POINTS - df.shape[0]
        if pad_width > 0:
            padding = np.zeros((pad_width, 4))  # Zero-padding for x, y, z, intensity
            df = pd.concat([df, pd.DataFrame(padding, columns=['x', 'y', 'z', 'intensity'])])
        
        # Flatten the DataFrame to match the input shape (1, 116732) as required
        df = df.to_numpy().flatten()[:116732].reshape(1, -1)

        # Predict the probabilities for each class
        probabilities = LOADED_MODEL.predict_proba(df)
        probability = probabilities.max(axis=1).mean()
        print(probability)
        # probability = probabilities[0].max()  # Get the max probability for the first (and only) sample

        # Get the class prediction
        predictions = LOADED_MODEL.predict(df)
        cntrs = Counter(predictions)
        prediction = cntrs.most_common(1)[0][0]

        # If probability is less than 95, mark the prediction as unsure
        if probability < 0.95:
            prediction = f"UNKNOWN (Guessed: {prediction})"

        print(f"Predicted: {prediction} ({probability})")


def main(args=None):
    print("start")
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
