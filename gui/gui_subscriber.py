import pickle
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from threading import Lock
from PySide6.QtCore import Signal, QObject
from collections import Counter
import pandas as pd
import numpy as np

class VelodyneSignals(QObject):
    data = Signal(object)  

class VelodyneSubscriber(Node):
    def __init__(self, qt_signals):
        super().__init__('velodyne_subscriber')
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',  # Default Velodyne topic, adjust if needed
            self.pointcloud_callback,
            10)
        
        self.qt_signals = qt_signals
        self.lock = Lock()
        self.point_count = 0
        self.model_filepath = "../lidar_classifier.sav"
        try: 
            with open(self.model_filepath, 'rb') as f:
                self.loaded_model = pickle.load(f)
        except FileNotFoundError:
            errmsg = f"Classifier file not found! {self.model_filepath}"
            print(errmsg)
            self.qt_signals.pred.emit(errmsg)
            exit(1)

    def pointcloud_callback(self, msg):
        MAX_POINTS = 29183  # This should match the max_points from your preprocessing
        
        with self.lock:
            # Read points from message
            # Convert generator to list of tuples

            points = read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=False) # pyright: ignore
            
            # Convert the points into a Pandas DataFrame
            df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity']) # pyright: ignore

            # Handle NaNs by replacing rows with [999, 999, 999, 999]
            df.loc[df.isna().any(axis=1)] = [999, 999, 999, 999]
            
            # Trim excess points if too many
            df = df.iloc[:MAX_POINTS]
            
            # Pad with zeros if the number of points is less than MAX_POINTS
            pad_width = MAX_POINTS - df.shape[0]
            if pad_width > 0:
                padding = np.zeros((pad_width, 4))  # Zero-padding for x, y, z, intensity
                df = pd.concat([df, pd.DataFrame(padding, columns=['x', 'y', 'z', 'intensity'])]) # pyright: ignore
            
            # Flatten the DataFrame to match the input shape (1, 116732) as required
            df = df.to_numpy().flatten()[:116732].reshape(1, -1)

            # Predict the probabilities for each class
            probabilities = self.loaded_model.predict_proba(df)
            probability = probabilities.max(axis=1).mean()
            # probability = probabilities[0].max()  # Get the max probability for the first (and only) sample

            # Get the class prediction
            predictions = self.loaded_model.predict(df)
            cntrs = Counter(predictions)
            prediction = cntrs.most_common(1)[0][0]

            # If probability is less than 95, mark the prediction as unsure
            if probability < 0.95:
                prediction = f"UNKNOWN (Guessed: {prediction})"

            
            locs = ['baker_no_curve', 'capstone_lab', 'embedded_lab', 'frost_large', 'gym', 'open_lab']
            if(prediction in range(0, len(locs))):
                prediction = locs[int(prediction)] 
            obj = {
                    "points": points,
                    "pred": f"Predicted: {prediction} ({probability})",
                    "proba": 0
                    }
            self.qt_signals.data.emit(obj)  # Send original point cloud for visualization
