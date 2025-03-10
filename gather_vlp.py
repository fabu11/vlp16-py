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
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import time, os, math, sys, glob


class MinimalSubscriber(Node):
    def __init__(self, output_dir):
        super().__init__('minimal_subscriber')

        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points',
            self.listener_callback,
            10)
        self.count = 0
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


    def listener_callback(self, msg):
        if(self.first):
            print("[Listener]: Subscribed")
            self.first = False
        # Get points from /velodyne_points
        points = read_points(cloud=msg, field_names=('x', 'y', 'z'), skip_nans=True) # pyright: ignore
        df = pd.DataFrame(points, columns=['x', 'y', 'z']) # pyright: ignore
        filename = f'{self.directory}/scan_{self.count}.csv'
        df.to_csv(filename, index=False)
        self.count += 1
        self.get_logger().info(f'[Listener]: Saved point cloud to {filename} with {len(df)} points')
        

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
