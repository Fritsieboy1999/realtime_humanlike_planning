#!/bin/bash
# Build script for IIWA Docker container
# This script should be run from the project root directory

echo "Building IIWA Docker container..."
echo "Building from: catkin_ws/src/iiwa-impedance-control-docker/"

cd catkin_ws/src/iiwa-impedance-control-docker/
docker build -t iiwa-imp-cntrl .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "Next steps:"
    echo "1. To run the container:"
    echo "   ./docker_run.sh"
    echo ""
    echo "2. Or manually run with:"
    echo "   docker run -it --user kuka_cor --name my_container --network=host --ipc=host \\"
    echo "     -v \$PWD/catkin_ws:/catkin_ws -v /tmp/.X11-unix:/tmp/.X11-unix:rw \\"
    echo "     --env=DISPLAY iiwa-imp-cntrl"
else
    echo "❌ Docker build failed!"
    exit 1
fi
