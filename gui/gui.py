import sys
import numpy as np
from threading import Thread
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QLabel 
from PySide6.QtCore import Slot, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import * # pyright: ignore 
from OpenGL.GLU import * # pyright: ignore 
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QSplitter, QSizePolicy

import rclpy
from gui_subscriber import VelodyneSubscriber, VelodyneSignals

import subprocess


from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.empty((0, 3), dtype=np.float32)  
        self.angle_x = -45  
        self.angle_y = 0
        self.last_x = 0 
        self.last_y = 0
        self.is_dragging = False  # Flag to check if mouse is being dragged
        self.zoom_factor = 50.0  # Initial zoom factor (higher means zoomed out)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)  # Enable depth testing
        glPointSize(2)  # Set point size
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / max(height, 1), 0.1, 500.0)  # Perspective view
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the buffer # pyright: ignore
        glLoadIdentity()

        # Apply rotations to the camera
        glTranslatef(0.0, 0.0, -self.zoom_factor)  # Zoom out by increasing negative z-value
        glRotatef(self.angle_x, 1, 0, 0)  
        glRotatef(self.angle_y, 0, 1, 0)  

        # Draw LiDAR points
        glBegin(GL_POINTS)
        glColor3f(1, 1, 0)  
        for x, y, z in self.points:
            glVertex3f(x, y, z)
        glEnd()

    def update_points(self, points):
        self.points = np.vstack((points['x'], points['y'], points['z'])).T  # Combine x, y, z into one array
        self.update()  # Refresh display

    def set_angle(self, angle_x, angle_y):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.update()  # Refresh the display with new angle

    # Mouse press event to start dragging
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton: # pyright: ignore
            self.is_dragging = True
            self.last_x = event.position().x()
            self.last_y = event.position().y()

    # Mouse move event to update rotation angles
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            dx = event.position().x() - self.last_x
            dy = event.position().y() - self.last_y

            self.angle_x += dy * 0.5
            self.angle_y += dx * 0.5

            self.last_x = event.position().x()
            self.last_y = event.position().y()

            self.update()

    # Mouse release event to stop dragging
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton: # pyright: ignore
            self.is_dragging = False

    # Wheel event for zooming in and out
    def wheelEvent(self, event):
        # Get the wheel delta, which indicates the scroll amount
        delta = event.angleDelta().y()  # Positive for scrolling up, negative for scrolling down

        # Zoom in (scroll up) or out (scroll down)
        if delta > 0:
            self.zoom_factor = max(1.0, self.zoom_factor - 5.0)  # Zoom in (decrease the zoom factor)
        else:
            self.zoom_factor += 5.0  # Zoom out (increase the zoom factor)

        self.update()  # Refresh the display


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle("Velodyne LiDAR Viewer")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.gl_widget = GLWidget(self)  # 3D point cloud visualizer
        self.launch_button = QPushButton("Launch Rviz2 - velodyne_points")
        self.launch_button.clicked.connect(self.launch_rviz2)

        left_layout.addWidget(self.gl_widget)
        left_layout.addWidget(self.launch_button)

        right_layout = QVBoxLayout()
        self.status_label = QLabel("[Status] Waiting for data...")
        self.pred_label = QLabel("[Prediction] Waiting for data...")
        self.points_label = QLabel("Points: 0")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_subscriber)
        self.stop_button.clicked.connect(self.stop_subscriber)

        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.pred_label)
        right_layout.addWidget(self.points_label)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal) # pyright: ignore
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Set the initial size of the left widget to be larger
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # pyright: ignore
        right_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred) # pyright: ignore

        # Set stretch factor to prioritize the left widget's expansion
        splitter.setSizes([600, 200])  # Adjust the size ratio of left and right widgets (larger left side)

        # Add splitter to the main layout
        main_layout.addWidget(splitter)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.signals = VelodyneSignals()
        self.signals.data.connect(self.update_pointcloud_info)

        self.node = None
        self.ros_thread = None
        self.running = False

    @Slot()
    def start_subscriber(self):
        if not self.running:
            self.running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Connecting to ROS2...")

            self.ros_thread = Thread(target=self.run_ros)
            self.ros_thread.daemon = True
            self.ros_thread.start()

    @Slot()
    def stop_subscriber(self):
        if self.running:
            self.running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("[Status] Stopped")

            if self.node is not None:
                rclpy.shutdown()
                self.node = None

    def run_ros(self):
        rclpy.init()
        self.node = VelodyneSubscriber(self.signals)
        self.status_label.setText("Connected. Waiting for data...")

        while self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    @Slot(list) # pyright: ignore
    def update_pointcloud_info(self, data):
        points = data["points"]
        pred = data["pred"]
        proba = data["proba"]
        self.status_label.setText("[Status] Receiving data")
        self.pred_label.setText(f"[Prediction] {pred}")
        self.points_label.setText(f"Points: {len(points)}")
        self.gl_widget.update_points(points)

    def closeEvent(self, event):
        self.stop_subscriber()
        event.accept()

    def launch_rviz2(self):
        rviz2_config = "./velodyne.rviz"
        self.rviz2_process = subprocess.Popen(["rviz2", "-f", "velodyne", "-d", rviz2_config])

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

