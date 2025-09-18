#!/bin/bash
# Build script for IIWA Docker container
# This script should be run from the project root directory

echo "Building IIWA Docker container..."
echo "Building from: catkin_ws/src/iiwa-impedance-control-docker/"

# Check if Docker daemon is running and user has permissions
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running or you don't have permissions."
    echo "Please run: sudo usermod -aG docker \$USER"
    echo "Then log out and log back in, or run: newgrp docker"
    exit 1
fi

cd catkin_ws/src/iiwa-impedance-control-docker/
echo "Current directory: $(pwd)"
echo "Building Docker image..."

docker build -t iiwa-imp-cntrl .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "Next steps:"
    echo "1. To run the container:"
    echo "   ./docker_run.sh"
    echo ""
    echo "2. Or manually from project root:"
    echo "   docker run -it --user kuka_cor --name my_container --network=host --ipc=host \\"
    echo "     -v \$PWD/catkin_ws:/catkin_ws -v /tmp/.X11-unix:/tmp/.X11-unix:rw \\"
    echo "     --env=DISPLAY iiwa-imp-cntrl"
    echo ""
    echo "3. If working in simulation only, first run:"
    echo "   cd catkin_ws/src/iiwa_ros/iiwa_driver && touch CATKIN_IGNORE"
else
    echo "❌ Docker build failed!"
    exit 1
fi
