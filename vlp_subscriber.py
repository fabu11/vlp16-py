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
import time, os, math

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10)
        self.subscription

        self.directory = "./output"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)


    def calculate_range(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)


    def listener_callback(self, msg):
        points = read_points(cloud=msg, field_names=('x', 'y', 'z'), skip_nans=False) # pyright: ignore
        point_array = np.array([[self.calculate_range(p[0], p[1], p[2]),
                                 p[0], p[1], p[2]] for p in points], dtype=np.float32)
        point_array[np.isnan(point_array)] = 999.0
        timestamp = int(time.time() * 1000)
        filename = f'{self.directory}/scan_{timestamp}.txt'
        np.savetxt(filename, point_array, fmt='%.4f', header='x\ty\tz')
        self.get_logger().info(f'Saved point cloud to {filename} with {point_array.shape[0]} points')

def main(args=None):
    print("start")
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
