- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Container](#Container)
- [Run](#run)
- [Troubleshooting](#troubleshooting)

# Introduction
This is a self-contained package aimed at running the code from [`iiwa-impedance-control`](https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control) using [Docker](https://www.docker.com/). This should save you from having to install manually all the dependencies, and allow you to run the controller code in simulation (with Gazebo) as well as on the KUKA robot. It is a starting point to then add your own higher-level packages in a flexible way.

# Prerequisite
You should have a working Ubuntu distribution ready on your computer (e.g. Ubuntu 22.04 LTS). Please follow the instruction on [Docker](https://www.docker.com/) and install Docker. 

# Installation
Go to your home folder and install this repo by 
```console
git clone git@gitlab.tudelft.nl:kuka-iiwa-7-cor-lab/iiwa-impedance-control-docker.git
cd iiwa-impedance-control-docker
```
Follow the instructions on cloning the repos to use the controllers. **You don't need to build any of the packages, the dockerfile will build them as it builds the docker image.** However, please make sure to switch to the correct branch on each repo. 

For kuka_fri (used for the real robot). If you don't have access to it, you can skip the next instruction block, and still work in simulation.
```console
mkdir kuka_dependencies
cd kuka_dependencies
git clone git@gitlab.tudelft.nl:nickymol/kuka_fri.git
```

Get the other dependencies
```console
git clone --recurse-submodules https://github.com/jrl-umi3218/SpaceVecAlg.git
git clone --recurse-submodules https://github.com/jrl-umi3218/RBDyn.git
git clone --recurse-submodules https://github.com/jrl-umi3218/mc_rbdyn_urdf.git
git clone --recurse-submodules git@github.com:mosra/corrade.git
cd corrade
git checkout 0d149ee9f26a6e35c30b1b44f281b272397842f5
cd ..
git clone --recurse-submodules git@github.com:epfl-lasa/robot_controllers.git
cd ..
```

Let's set up the catkin workspace for our container, and clone there the EPFL LASA implementation of [`iiwa_ros`](https://github.com/epfl-lasa/iiwa_ros), and the [`iiwa-impedance-control`](https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control) repository from Nicky Mol at TU Delft.
**Note**: make sure that you checkout to the correct branches that you want to work on! E.g., if you modified `iiwa_impedance_control`, you should checkout to your branch in this step too.

```console
cd catkin_ws
mkdir src && cd src
git clone --recurse-submodules git@github.com:epfl-lasa/iiwa_ros.git
git clone --recurse-submodules git@gitlab.tudelft.nl:nickymol/iiwa_impedance_control.git
cd ../..
```

Now you can build you docker image by
```console
docker build -t iiwa-imp-cntrl .
```
this can take a while, grab a coffee, make some tea, and wait. After the image is built, you can check your list of available docker images by
``` console
docker image list
```
You should see information about the image `iiwa-imp-cntrl` which you just created!

In the docker file, we created a user `kuka_cor` with `USER_UID` and `USER_GID` both at 1000, same as your host. This mean that you can share volumes and files between docker image and host.

# Container
Here, you create the actual docker container that will allow to run the code.
You should put packages you wish to run on kuka into the catkin workspace's source folder (`catkin_ws/src`), where `iiwa-ros` and `iiwa-impedance-control` are. Here, you will also install the packages/projects that you need, so that they are built and can be accessed in the Docker container as well.

If you are working only in simulation (i.e., you don't have access to the `kuka_fri` package), run the following:
```console
cd catkin_ws/src/iiwa_ros/src/iiwa_driver
touch CATKIN_IGNORE
cd ../../..
```

Run the docker image and load directory `catkin_ws` as a volume
```console
docker run -it --user kuka_cor --name my_container --network=host --ipc=host -v $PWD/catkin_ws:/catkin_ws -v /temp/.X11-unix:/temp/.X11-unix:rw --env=DISPLAY iiwa-imp-cntrl
```

This will create a container named `my_container` with the image `kuka_ros`, and at any point, you can exit the container with `ctrl+ d`.

*Note*: the first time you run this you should see this error `bash: /catkin_ws/devel/setup.bash: No such file or directory`. This is because we have not run `catkin build` yet, so don't worry!

You can start it again with 
```console
docker start -i my_container
```
or you can also open up a different terminal in the same container by 
```console
docker exec -it my_container bash
```
Firstly, you should build your working package by 
```console
cd catkin_ws
catkin build
cd ..
source catkin_ws/devel/setup.bash
```
Any changes made by `catkin build` will also show up in the host directory so you don't need to rebuild the package every time you rerun your image. And sourcing of your package is taken care of in bashrc. 

# Run



# Troubleshooting

## Nvidia Issues [Docker]
If you are having problem using your Nvidia GPU in docker GUIs, please take a look at [Nvidia Container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) and follow their instruction. Generally, adding args into the `docker run` command helps, e.g. `--runtime=nvidia` and `--gpus all`. 
For example, the run command will be like
```console
    docker run -it --runtime=nvidia --gpus all --user kuka_cor --name my_container --network=host --ipc=host -v $PWD/catkin_ws:/catkin_ws -v /temp/.X11-unix:/temp/.X11-unix:rw --env=DISPLAY kuka_ros
```

## Missing GPU [Docker]
If your system does not have a GPU, you will see an error like the following when trying to run some applications (e.g. Gazebo):
```console
libGL error: MESA-LOADER: failed to retrieve device information
Segmentation fault (core dumped)
```

You should make sure that OpenGL uses software rendering instead, by running:
```console
export LIBGL_ALWAYS_SOFTWARE=1
```
You need to run this in every terminal, so add this line to the `~/.bashrc` in your container, to set it on every terminal session.