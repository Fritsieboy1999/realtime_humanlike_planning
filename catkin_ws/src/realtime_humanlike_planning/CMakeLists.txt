cmake_minimum_required(VERSION 3.0.2)
project(realtime_humanlike_planning)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  visualization_msgs
  controller_manager_msgs
  std_msgs
  std_srvs
  actionlib
  actionlib_msgs
  tf2_ros
  tf2_geometry_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    visualization_msgs
    controller_manager_msgs
    std_msgs
    std_srvs
)

## Install Python scripts
catkin_install_python(PROGRAMS
  iiwa_planning_interface.py
  safety_monitor.py
  example_usage.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.rviz"
)

## Install Python modules
install(DIRECTORY planning/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/planning
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY parameters/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/parameters
  FILES_MATCHING PATTERN "*.py"
)
