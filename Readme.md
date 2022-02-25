This repository provides a list of relevant ROS2 packages and other open source packages in the autonomous driving space. There are other great lists with slightly different scope for [ROS2](https://fkromer.github.io/awesome-ros2/) and general [robotics](http://jslee02.github.io/awesome-robotics-libraries/). 

## Software stacks
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Autoware | Apache 2.0 | [here](https://www.autoware.org/) | C++  | Full software stack |
| AWS DeepRacer | Apache 2.0 |[here](https://github.com/orgs/aws-deepracer/repositories) | C++ |  Collection of packages for AWS DeepRacer including ros2 integration |
| Apollo | Apache 2.0 |  [here](https://github.com/ApolloAuto/apollo) | C++ | Full software stack |

## Ground Filtering
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| CSF | Apache 2.0 | [here](https://github.com/jianboqi/CSF) | C++ |Cloth Simulation Filtering from aerial lidar
| Patchwork | MIT  | [here](https://github.com/LimHyungTae/patchwork) | C++ |  Filtering, 43Hz with Ouster 0S0
| Cascaded Ground Segmentation  | Apache 2.0  | [here](https://github.com/wangx1996/Cascaded-Lidar-Ground-Segmentation) | C++ | Slope based filtering
| GndNet | MIT | [here](https://github.com/anshulpaigwar/GndNet)| Python| ground removal with neural network	| 
| Fast Ground Segmentation of 3D Point Clouds |  GPL 3.0 | [here](https://github.com/Amsterdam-AI-Team/3D_Ground_Segmentation)| C++ | makes use of scan line properties
| Ground Removal | None | [here](https://github.com/SilvesterHsu/LiDAR_ground_removal)| Python | also check out original paper, only python code	

## Sensor Calibration
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| ILCC | BSD2 |  [here](https://github.com/mfxox/ILCC) | Python | 3D-LiDAR and camera extrinsic calibration|
| mi-extrinsic-calib | LGPL |  [here](https://github.com/anlif/mi-extrinsic-calib) | C++ | 3D-LiDAR and camera extrinsic calibration |
| automatic_lidar_camera_calibration | None |  [here](https://github.com/xmba15/automatic_lidar_camera_calibration) | C++ | 3D-LiDAR and camera extrinsic calibration  |

## Object detection
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Apollo | Apache 2.0 | [here](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/lidar/README.md) | C++ | Mask-Pillars Lidar Object Detection |
| Depth Clustering | MIT | [here](https://github.com/PRBonn/depth_clustering) | C++ | Clustering for Detection|
| mmdetection3d | Apache 2.0 | [here](https://github.com/open-mmlab/mmdetection3d) | Python | Variety of Lidar Object Detection Models |
| Open3D-ML | MIT | [here](https://github.com/isl-org/Open3D-ML) | Python | Variety of Lidar Object Detection Models |
| OpenPCDet | Apache 2.0 | [here](https://github.com/open-mmlab/OpenPCDet) | Python | Variety of Lidar Object Detection Models  |
| CFR-Net for Object detection | Apache 2.0 | [here](https://github.com/TUMFTM/CameraRadarFusionNet) | Python | Object detection with a neural net based on camera and radar data |

## Lane detection
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Papers with Code | |  [here](https://paperswithcode.com/task/lane-detection) | | Overview Page with Benchmarks |
| TuSimple Benchmark | Apache 2.0 | [here](https://github.com/TuSimple/tusimple-benchmark) | Python | Lane Line Dataset Loader |
|  CU Lane Dataset | Non-comercial| [here](https://xingangpan.github.io/projects/CULane.html) | | Lane Line Dataset |
| lanedet | Apache 2.0 | [here](https://github.com/Turoad/lanedet) | Python | Collection of Multiple Lane Detectors |
|  RESA | Apache 2.0 | [here](https://github.com/ZJULearning/resa) | Python | RESA: Recurrent Feature-Shift Aggregator for Lane Detection |
| lanenet-lane-detection | Apache 2.0 | [here](https://github.com/MaybeShewill-CV/lanenet-lane-detection) | Python | Multi-Stage Lane Detection |
| Ultra-Fast-Lane-Detection | MIT | [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection) | Python | 300+ FPS in Benchmark|
| Codes-for-Lane-Detection | MIT | [here](https://github.com/cardwing/Codes-for-Lane-Detection) | Python | Several Networks |
| LaneAF | None | [here](https://github.com/sel118/LaneAF) | Python |Robust Multi-Lane Detection with Affinity Fields |

## 3D segmentation
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| lidar_bonnetal | MIT | [here](https://github.com/PRBonn/lidar-bonnetal) | Python | Lidar Range Image Segmentation |
| SalsaNext | MIT | [here](https://github.com/Halmstad-University/SalsaNext) | Python | Uncertainty-aware Lidar Segmenation |
| Cylinder3D | Apache 2.0 | [here](https://github.com/xinge008/Cylinder3D) | Python | Cylindric Convolutions for Lidar Segmentation |

## General perception 
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Awesome Multiple Object Tracking | | [here](https://github.com/luanshiyinyang/awesome-multiple-object-tracking) | | Meta repository with papers and code
| easy_perception_deployment | Apache 2.0 | [here](https://github.com/ros-industrial/easy_perception_deployment) | C++ | Deployment of CV models
| ros2_object_analytics | Apache 2.0 | [here](https://github.com/intel/ros2_object_analytics) | C++ | General purpose object detection and tracking packages
| vision_opencv | None | [here](https://github.com/ros-perception/vision_opencv/tree/ros2) | C++ | Interface to OpenCV
| slam_gmapping | None | [here](https://github.com/Project-MANAS/slam_gmapping) | C++ | Wrapper for gmapping SLAM algorithm
| cartographer | Apache 2.0 | [here](https://github.com/ros2/cartographer) | C++ | Wrapper for cartographer SLAM algorithm
| lidarslam_ros2 | BSD-2-Clause | [here](https://github.com/rsasaki0109/lidarslam_ros2) | C++ | NDT scan matching SLAM algorithm
| slam_toolbox | LGPL-2.1 | [here](https://github.com/SteveMacenski/slam_toolbox) | C++ | Holistic library with a variety of SLAM options
| perception_pcl | None | [here](https://github.com/ros-perception/perception_pcl) | C++ | Functions for handling pointcloud data with PCL
| image_common | None | [here](https://github.com/ros-perception/image_common/tree/noetic-devel) | C++ | Functions for handling images
| ros_msft_onnx | MIT | [here](https://github.com/ms-iot/ros_msft_onnx) | C++ | Wrapper for inference with ONNX models

## Prediction
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Wale-Net Prediction Network | LGPL-3.0 | [here](https://github.com/TUMFTM/Wale-Net) | Python | Encoder-decoder neural network for vehicle trajectory prediction with uncertainties | 

## Planning algorithms
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Global racetrajectory optimization | LGPL-3.0 | [here](https://github.com/TUMFTM/global_racetrajectory_optimization) | Python | Various approaches from minimum curvature to minimum time | 
Graph-based local trajectory planner | LGPL-3.0 | [here](https://github.com/TUMFTM/GraphBasedLocalTrajectoryPlanner) | Python | Local trajectory planner with independent graph-based path and velocity planning
Velocity optimization module | LGPL-3.0 | [here](https://github.com/TUMFTM/velocity_optimization) | Python | Optimization-based vleocity planning module considering external parameters (e.g., the friction coefficient or a maximum power limit)
Trajectory supervisor | LGPL-2.1 | [here](https://github.com/TUMFTM/TrajectorySupervisor) | Python | Online verification framework for autonomous vehicle trajectory planning

## Control algorithms
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
Vehicle motion control | LGPL-3.0 | [here](https://github.com/TUMFTM/mod_vehicle_dynamics_control) | Simulink | Motion control (LQR) and state estimation for autonomous vehicles

## Hardware drivers
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| Ouster LIDAR | Apache 2.0 | [here](https://github.com/ros-drivers/ros2_ouster_drivers) | C++ | Supports all models
| Velodyne LIDAR | Apache 2.0 | [here](https://github.com/ros-drivers/velodyne) | C++ | Supports 64E(S2, S2.1, S3), the 32E, the 32C, and the VLP-16. Additional functionality available. 
| Smartmicro Radar | Apache 2.0 | [here](https://github.com/smartmicro/smartmicro_ros2_radars) | C++ | Supports UMRR-96
| CAN interface | Apache 2.0 | [here](https://github.com/autowarefoundation/ros2_socketcan) | C++ | Interfaces the socketcan API of linux
| OXTS INS | Apache 2.0 | [here](https://github.com/OxfordTechnicalSolutions/oxts_ros2_driver) | C++ | Supports all OXTS units
| Novatel INS | MIT | [here](https://github.com/novatel/novatel_oem7_driver/tree/ros2-dev) | C++ | Supports OEM7 units

## Tools & Simulation
| Name | License | Link | Language | Short description |
|---------------|--------------|----------------|-------------|---------------------------|
| CARLA | MIT | [here](https://github.com/carla-simulator/carla) | | Full stack simulation environment based on the Unreal Game Engine
| Foxglove | MPL-2.0 | [here](https://github.com/foxglove/studio) | | Web-based visualization and diagnosis tool
| Webviz | Apache 2.0 | [here](https://github.com/cruise-automation/webviz) | | Web-based visualization and diagnosis tool
| PlotJuggler | LGPL-3.0 | [here](https://github.com/facontidavide/PlotJuggler) | | Time-series visualization tool

## Papers & Articles
| Name | Link | Short description |
|---------------|----------------|---------------------------|
| Racing overview | [here](https://github.com/JohannesBetz/AutonomousRacing_Literature) | Meta repository with papers and code
| Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle Racing | [here](https://www.researchgate.net/publication/358632526_Autonomous_Vehicles_on_the_Edge_A_Survey_on_Autonomous_Vehicle_Racing) | Survey paper on autonomous vehicle racing
| Oops! It's Too Late. Your Autonomous Driving System Needs a Faster Middleware | [here](https://www.semanticscholar.org/paper/Oops!-It's-Too-Late.-Your-Autonomous-Driving-System-Wu-Wu/a60c0ccf8c3cec22fc401277626b0c67e551e5ea) | Analysis of ROS1 / ROS2 / Cyber with respect to latency and performance |
| A Self-Driving Car Architecture in ROS2 | [here](https://ieeexplore.ieee.org/abstract/document/9041020) | Software architecture for a vehicle with ROS2
| Real-time configuration with ROS2 | [here](https://answers.ros.org/question/382893/best-practices-for-real-time-capabilites-in-ros2/) | Details on configuration of ROS2 and OS for RT performance
| Latency Analysis of ROS2 Multi-Node Systems | [here](https://arxiv.org/abs/2101.02074) | Detailed latency analysis for different ROS2 middlewares
