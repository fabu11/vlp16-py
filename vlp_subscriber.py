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

cnt = 0

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Shortest distance used in sort
        self.shortest_distance = {               
                        "range": 9999,
                        "index": 0,
                        }
        self.storage = {}
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
        points = list(read_points(cloud=msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False)) # pyright: ignore
        for i, p in enumerate(points):
            # Distance is shorter, need to update origin
            c = self.calculate_range(p[0], p[1], p[2]) 
            if(c < self.shortest_distance["range"]):
                new_shortest = {
                        "range": c,
                        "index": i,
                        }
                self.shortest_distance = new_shortest
        new_p = points[self.shortest_distance["index"]:] + points[:self.shortest_distance["index"]]
        new_p = np.array(new_p)
        np.nan_to_num(new_p, nan=0)
        timestamp = int(time.time() * 1000)
        filename = f'{self.directory}/scan_{timestamp}.txt'
        np.savetxt(filename, new_p, fmt='%.4f', header='x\ty\tz')
        self.get_logger().info(f'Saved point cloud to {filename} with {len(new_p)} points')
        

def main(args=None):
    print("start")
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
