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
import time, os, math, sys, glob


class MinimalSubscriber(Node):
    def __init__(self, output_dir):
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
        self.first = True
        # default output to ./output
        self.directory = "./output"
        if output_dir:
            self.directory = output_dir
        print(f"[I/O]: Ouptut directory set to {self.directory}")
        # Check if output dir is empty, if not ask if we want to remove all files
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else:
            if(len(os.listdir(self.directory)) != 0):
                abs_path = os.path.join(os.getcwd(), output_dir)
                if(input(f"[I/O]: {abs_path} is not empty. Do you want to clear? Y/n\t").lower() == 'y'):
                    files = glob.glob(os.path.join(abs_path, '*'))
                    for f in files:
                        print(f"[I/O]: removing {f}")
                        os.remove(f)
                    print(f"[I/O]: removed {len(files)} files")

    def calculate_range(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def listener_callback(self, msg):
        if(self.first):
            print("[Listener]: Subscribed")
            self.first = False
        # Get points from /velodyne_points
        points = list(read_points(cloud=msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False)) # pyright: ignore
        # Update shortest point
        for i, p in enumerate(points):
            # Distance is shorter, need to update origin
            c = self.calculate_range(p[0], p[1], p[2]) 
            if(c < self.shortest_distance["range"]):
                new_shortest = {
                        "range": c,
                        "index": i,
                        }
                self.shortest_distance = new_shortest
        # Reorder points so that shortest is always first, keep order
        new_p = points[self.shortest_distance["index"]:] + points[:self.shortest_distance["index"]]
        new_p = np.array(new_p)
        # Set NaN to 0
        np.nan_to_num(new_p, nan=0)
        # Save to output directory
        timestamp = int(time.time() * 1000)
        filename = f'{self.directory}/scan_{timestamp}.txt'
        np.savetxt(filename, new_p, fmt='%.4f', header='x\ty\tz')
        self.get_logger().info(f'[Listener]: Saved point cloud to {filename} with {len(new_p)} points')
        

def main(args=None):
    args = sys.argv[1:]
    output_dir = None
    # get sys arg flags
    if '-o' in args:
        o_index = args.index('-o')
        output_dir = args[o_index + 1]
        if not output_dir:
            print("[Error]: with output_dir")
            print(f"Usage: {sys.argv[0]} -o <outputfile>")
            exit(-1)
    print("Start:")
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber(output_dir)
    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        print("\nShutting Down...")
    finally:
        if rclpy.ok():
            minimal_subscriber.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
