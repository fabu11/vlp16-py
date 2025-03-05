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
        points = read_points(cloud=msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False) # pyright: ignore
        df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity']) # pyright: ignore

        try:
            with open('lidar_classifier.sav', 'rb') as f:
                loaded_model = pickle.load(f)
        except FileNotFoundError:
            self.get_logger().error("Classifier file not found. Please make sure 'lidar_classifier.sav' is in the correct location.")
            return

        # get probabilities
        probabilities = loaded_model.predict_proba(df[['x', 'y', 'z', 'intensity']])
        probability = probabilities.max(axis=1).mean()

        # get predictions
        predictions = loaded_model.predict(df[['x', 'y', 'z', 'intensity']])
        cntrs = Counter(predictions)
        prediction = cntrs.most_common(1)[0][0]

        # if probability is less than 95, then it is unsure
        if probability < .95:
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
