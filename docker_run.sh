#!/bin/bash
# Run script for IIWA Docker container
# This script should be run from the project root directory

echo "Starting IIWA Docker container..."
echo "Mounting catkin_ws from: $(pwd)/catkin_ws"

# Check if container already exists
if [ "$(docker ps -aq -f name=my_container)" ]; then
    echo "Container 'my_container' already exists. Starting it..."
    docker start -i my_container
else
    echo "Creating new container 'my_container'..."
    docker run -it --user kuka_cor --name my_container --network=host --ipc=host \
        -v $PWD/catkin_ws:/catkin_ws -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --env=DISPLAY iiwa-imp-cntrl
fi

echo ""
echo "Container commands:"
echo "- Build catkin workspace: cd catkin_ws && catkin build"
echo "- Source workspace: source catkin_ws/devel/setup.bash"
echo "- Exit container: Ctrl+D"
echo "- Open new terminal in container: docker exec -it my_container bash"
