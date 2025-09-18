source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
export XDG_RUNTIME_DIR=/temp/xdg
sudo mkdir -p $XDG_RUNTIME_DIR
sudo chown $(id -u):$(id -g) /temp/xdg

sudo chmod 0700 $XDG_RUNTIME_DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/kuka_dependencies/RBDyn/build/src

# adding line for avoiding using MESA libraries on GPU
export LIBGL_ALWAYS_SOFTWARE=1

# defining alias for sourcing ROS and catkin workspace
alias noetic="source /opt/ros/noetic/setup.bash"

# defining alias to set ROS parameters when communicating with the robot
connect_robot()
{
    export ROS_MASTER_URI=http://192.180.1.5:11311/
    export ROS_IP=192.180.1.5
}
