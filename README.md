# VLP16-py
Python package to read live data from VLP-16 sensor.

##### [Note] This project uses git-lfs
- Install [here](https://git-lfs.com/)
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

- Learn more [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)

## Install ROS2 Humble (requires Ubuntu 22.04 Jammy):
[Follow these instructions from ROS Docs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

## Install Velodyne ROS Packages
```
sudo apt install ros-humble-velodyne
```

## Optional shell alias
```
vim ~/.bashrc && source ~/.bashrc
```
Add the following to the end of your shell's rc
```
alias startros="source /opt/ros/humble/setup.bash"
alias vlplaunch="ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py"
```
- If Network is set up [see doc](https://wiki.ros.org/velodyne/Tutorials/Getting%20Started%20with%20the%20Velodyne%20VLP16), then  vlplaunch will start reading packets from VLP-16 sensor

## Playing Bags
##### Setup
- [Docs](https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)
##### Visualizing bag
- Rviz2
```
sudo apt install ros-humble-rviz2
```
```
rviz2 -f velodyne
```
- Once rviz2 opens, go click 'Add' on bottom left > then click 'By topic' > then click the /velodyne_points PointCloud2 option.


## Python Subscriber
```
sudo apt install ros-humble-sensor-msgs
sudo apt install ros-humble-sensor-msgs-py
```






