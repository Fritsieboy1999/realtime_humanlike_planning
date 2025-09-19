#!/bin/bash
# Run script for IIWA Docker container
# This script should be run from the project root directory

echo "Starting IIWA Docker container..."
echo "Mounting catkin_ws from: $(pwd)/catkin_ws"

# Check if Docker daemon is running and user has permissions
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running or you don't have permissions."
    echo "Please run: sudo usermod -aG docker \$USER"
    echo "Then log out and log back in, or run: newgrp docker"
    exit 1
fi

# Check if container already exists
if [ "$(docker ps -aq -f name=my_container)" ]; then
    echo "Container 'my_container' already exists. Starting it..."
    docker start -i my_container
else
    echo "Creating new container 'my_container'..."
    echo "Note: First run will show 'setup.bash not found' - this is normal!"
    docker run -it --user kuka_cor --name my_container --network=host --ipc=host \
        -v $PWD/catkin_ws:/catkin_ws -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --env=DISPLAY kuka-planning-image
fi

echo ""
echo "Container commands:"
echo "- Build catkin workspace: cd catkin_ws && catkin build"
echo "- Source workspace: source catkin_ws/devel/setup.bash"
echo "- Exit container: Ctrl+D"
echo "- Open new terminal in container: docker exec -it my_container bash"
echo ""
echo "For simulation only (no real robot):"
echo "- Run once: cd catkin_ws/src/iiwa_ros/iiwa_driver && touch CATKIN_IGNORE"
