import pickle
import sys
from PySide6 import QtCore, QtWidgets 
from PySide6.QtCore import Qt, QMetaObject, Q_ARG
from rclpy.node import Node
import rclpy
import subprocess
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import pandas as pd
import numpy as np
from collections import Counter
class Gui(Node, QtWidgets.QWidget):
    def __init__(self, model):
        Node.__init__(self, 'gui')
        QtWidgets.QWidget.__init__(self)

        # Saved Model
        self.loaded_model = model

        # GUI Window Settings
        self.setWindowTitle("VLP LiDAR Classifier")
        self.setGeometry(100, 100, 1024, 768)

        # init layouts
        main_layout = QtWidgets.QHBoxLayout()
        l_layout = QtWidgets.QVBoxLayout()
        r_layout = QtWidgets.QVBoxLayout()

        #### LEFT
        self.list_widget = QtWidgets.QListWidget()
        self.refresh_button = QtWidgets.QPushButton("Refresh Topics")
        self.refresh_button.clicked.connect(self.update_topics)
        self.launch_button = QtWidgets.QPushButton("Launch Rviz2 - velodyne_points")
        self.launch_button.clicked.connect(self.launch_rviz2)
        self.launch_button.setDisabled(True)

        l_layout.addWidget(self.list_widget)
        l_layout.addWidget(self.refresh_button)
        l_layout.addWidget(self.launch_button)


        #### RIGHT
        self.desc = QtWidgets.QLabel("CPE Capstone: Lidar Classifier\nLorem ipsum dolor sit amet...")
        self.sub_out = QtWidgets.QLabel("... Waiting for /velodyne_points topic. Try refreshing.")

        r_layout.addWidget(self.desc)
        r_layout.addWidget(self.sub_out)

        # Add Left and Right Layouts to Main Layout
        main_layout.addLayout(l_layout)
        main_layout.addLayout(r_layout)
        self.setLayout(main_layout)

        # ROS2 Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            'velodyne_points', 
            self.listener_callback,
            10)

        self.update_topics()

    def listener_callback(self, msg):
        points = read_points(cloud=msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False) # pyright: ignore
        df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity']) # pyright: ignore
        probabilities = self.loaded_model.predict_proba(df[['x', 'y', 'z', 'intensity']])
        probability = probabilities.max(axis=1).mean()

        # get predictions
        predictions = self.loaded_model.predict(df[['x', 'y', 'z', 'intensity']])
        cntrs = Counter(predictions)
        prediction = cntrs.most_common(1)[0][0]

        # if probability is less than 95, then it is unsure
        if probability < .95:
            prediction = f"UNKNOWN (Guessed: {prediction})"
        fn = f"Predicted: {prediction} ({probability})"
        QMetaObject.invokeMethod(self.sub_out, "setText", Q_ARG(str, fn))

        
    def update_topics(self):
        self.list_widget.clear()
        topic_list = self.get_topic_names_and_types()
        if any(topic == "/velodyne_points" for topic, _ in topic_list):  
            self.launch_button.setDisabled(False)
        else:
            self.launch_button.setDisabled(True)
        for topic, _ in topic_list:
            self.list_widget.addItem(topic)


    def launch_rviz2(self):
        rviz2_config = "./velodyne.rviz"
        self.rviz2_process = subprocess.Popen(["rviz2", "-f", "velodyne", "-d", rviz2_config])


def _get_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
            return loaded_model
    except FileNotFoundError:
        print("Classifier file not found. Please make sure 'lidar_classifier.sav' is in the correct location.")
        exit(1)


def main():
    rclpy.init()
    #TODO option to get from cli
    file_path = 'lidar_classifier.sav'
    app = QtWidgets.QApplication(sys.argv)
    window = Gui(_get_model(file_path))
    window.show()
    sys.exit(app.exec())


if(__name__ == '__main__'):
    main()






