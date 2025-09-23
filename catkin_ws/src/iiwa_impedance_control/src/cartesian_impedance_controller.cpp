//
// Created by Nicky Mol on 10/08/2022.
// Email: nicky.mol@tudelft.nl
//

#include "iiwa_impedance_control/cartesian_impedance_controller.h"
#include "iiwa_impedance_control/pseudo_inversion.h"


namespace cartesian_impedance_controller {

    CartesianImpedanceController::CartesianImpedanceController() {}
    CartesianImpedanceController::~CartesianImpedanceController() {}

    bool CartesianImpedanceController::init(hardware_interface::EffortJointInterface* effort_joint_interface, ros::NodeHandle& node_handle) {
        nh = node_handle;
        // Init stiffness matrix
        K.setIdentity(6, 6);
        // Init damping matrix
        D.setIdentity(6, 6);
        // Init offset stiffness matrix
        K_ext.setIdentity(6, 6);

        // Init nullspace stiffness matrix
        K_nullspace.setIdentity(7, 7);
        // Init nullspace damping matrix
        D_nullspace.setIdentity(7, 7);

        latched = false;

        // Get the URDF XML from the parameter server
        std::string urdf_string, full_param;
        std::string robot_description = "robot_description";
        std::string base = "iiwa_link_0";
        std::string end_effector = "iiwa_link_ee";

        // Gets the location of the robot description on the parameter server
        if (!nh.searchParam(robot_description, full_param)) {
            ROS_ERROR("Could not find parameter %s on parameter server", robot_description.c_str());
            return false;
        }

        // Search and wait for robot_description on param server
        while (urdf_string.empty()) {
            ROS_INFO("CartesianImpedanceController is waiting for model URDF in parameter [%s] on the ROS param server.",
                                robot_description.c_str());

            nh.getParam(full_param, urdf_string);

            usleep(100000);
        }
        ROS_INFO_STREAM("Received urdf from param server, parsing...");

        // Get the end-effector
        nh.param<std::string>("/CartesianImpedanceController/end_effector", end_effector, "iiwa_link_ee");
        ROS_INFO_STREAM("End-effector: " << end_effector);

        // Get URDF
        urdf::Model robot;
        if (not robot.initParam(robot_description)) {
            ROS_ERROR("Failed to parse urdf file");
            return false;
        }

        // Get KDL Tree
        KDL::Tree tree;
        if (not kdl_parser::treeFromUrdfModel(robot, tree)) {
            ROS_ERROR("Failed to extract kdl tree from xml robot description");
            return 1;
        }

        // Get KDL Chain from base to end effector
        if (not tree.getChain(base, end_effector, eef_chain)) {
            ROS_ERROR("Cannot find chain within URDF tree from base: %s to end effector: %s.", base.c_str(), end_effector.c_str());
        }

        // Get KDL Chain from base to iiwa_link_ee
        if (not tree.getChain(base, end_effector, iiwa_chain)) {
            ROS_ERROR("Cannot find chain within URDF tree from base: %s to end effector: %s.", base.c_str(), "iiwa_link_ee");
        }

        if (nh.getParam("joints", joint_names)) {
            n_joints = joint_names.size();
            if (effort_joint_interface) {
                // get the joint objects to use in the realtime loop
                for (unsigned int i = 0; i < n_joints; i++) {
                    try {
                        jointsHandles.push_back(effort_joint_interface->getHandle(joint_names[i]));
                    }
                    catch (const hardware_interface::HardwareInterfaceException& e) {
                        ROS_ERROR_STREAM("Exception thrown: " << e.what());
                        return false;
                    }
                    urdf::JointConstSharedPtr joint_urdf = robot.getJoint(joint_names[i]);
                    if (!joint_urdf) {
                        ROS_ERROR("Could not find joint '%s' in urdf", joint_names[i].c_str());
                        return false;
                    }
                    joint_urdfs.push_back(joint_urdf);
                }
            } else {
                ROS_ERROR("Effort Joint Interface is empty in hardware interace.");
                return false;
            }
        } else {
            ROS_ERROR("No joints in given namespace: %s", nh.getNamespace().c_str());
            return false;
        }

        // Optional: publishers
        joint_state_pub = nh.advertise<sensor_msgs::JointState>("/CartesianImpedanceController/joint_states", 500);
        cartesian_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/CartesianImpedanceController/cartesian_pose", 500);
        trajectory_pub = nh.advertise<geometry_msgs::PoseStamped>("/CartesianImpedanceController/trajectory", 500);
        commanded_torque_pub = nh.advertise<std_msgs::Float64MultiArray>("/CartesianImpedanceController/commanded_torque", 500);
        cartesian_wrench_pub = nh.advertise<geometry_msgs::WrenchStamped>("/CartesianImpedanceController/cartesian_wrench", 500);

        // Dynamic reconfigure
        dynamic_reconfigure_server_node = ros::NodeHandle(nh.getNamespace() + "/dynamic_reconfigure_server_node");
        dynamic_reconfigure_server = std::make_unique<dynamic_reconfigure::Server<iiwa_impedance_control::CartesianImpedanceControllerConfig>>(dynamic_reconfigure_server_node);
        dynamic_reconfigure_server->setCallback(boost::bind(&CartesianImpedanceController::dynamicReconfigureCallback, this, _1, _2));

        // Action server
        action_server = std::unique_ptr<actionlib::SimpleActionServer<iiwa_impedance_control::CartesianTrajectoryExecutionAction>>(
                new actionlib::SimpleActionServer<iiwa_impedance_control::CartesianTrajectoryExecutionAction>(
                        nh, std::string("cartesian_trajectory_execution_action"), false));
        action_server->registerGoalCallback(boost::bind(&CartesianImpedanceController::goalCallback, this));
        action_server->registerPreemptCallback(boost::bind(&CartesianImpedanceController::preemptCallback, this));
        action_server->start();
        
        // Add subscriber for real-time reference pose updates
        reference_pose_sub = nh.subscribe("reference_pose", 10, &CartesianImpedanceController::refPoseCallback, this);
        ROS_INFO("CartesianImpedanceController: Subscribed to reference_pose topic for real-time updates");
        
        nh.param<double>("/CartesianTrajectoryGenerator/time_limit", time_limit, 2.0);
        nh.param<double>("/CartesianTrajectoryGenerator/goal_translation_tolerance", goal_translation_tolerance, 0.01);
        nh.param<double>("/CartesianTrajectoryGenerator/goal_rotation_tolerance", goal_rotation_tolerance, 0.5);

        return true;
    }

    void CartesianImpedanceController::starting(const ros::Time& time) {
        setReferenceToCurrentPose();
        setHomeToCurrentReference();
    }

    void CartesianImpedanceController::update(const ros::Time& time, const ros::Duration& period) {
        Eigen::Matrix<double, 6, 1> ee_pose;
        Eigen::Matrix<double, 6, 1> ee_pose_error;
        Eigen::VectorXd tau_task;
        Eigen::VectorXd tau_nullspace;
        Eigen::VectorXd tau_ext;
        Eigen::VectorXd tau;

        RobotState robot_state = updateRobotState();

        // Get geometrical Jacobian
        Eigen::MatrixXd J_eef = getJacobian(robot_state.position, eef_chain).data;
        Eigen::MatrixXd J_iiwa = getJacobian(robot_state.position, iiwa_chain).data;

        // Get end effector pose
        Eigen::Affine3d T = getForwardKinematics(robot_state.position, eef_chain);
        Eigen::Vector3d ee_position(T.translation());
        Eigen::Quaterniond ee_orientation(T.linear());

        ee_pose.head(3) << ee_position;
        ee_pose.tail(3) << ee_orientation.x(), ee_orientation.y(), ee_orientation.z();

        // If trajectory controller is active, update reference pose
        if (action_server->isActive() && ee_position_goal != Eigen::Vector3d::Zero()) {
            if (std::isnan(ee_position_goal_enabled(0))) {
                ee_position_goal(0) = ee_position.x();
            }
            if (std::isnan(ee_position_goal_enabled(1))) {
                ee_position_goal(1) = ee_position.y();
            }
            if (std::isnan(ee_position_goal_enabled(2))) {
                ee_position_goal(2) = ee_position.z();
            }
            if (std::isnan(ee_orientation_goal_enabled.w())) {
                ee_orientation_goal.w() = ee_orientation.w();
            }
            if (std::isnan(ee_orientation_goal_enabled.x())) {
                ee_orientation_goal.x() = ee_orientation.x();
            }
            if (std::isnan(ee_orientation_goal_enabled.y())) {
                ee_orientation_goal.y() = ee_orientation.y();
            }
            if (std::isnan(ee_orientation_goal_enabled.z())) {
                ee_orientation_goal.z() = ee_orientation.z();
            }

            // Translation and rotational distance
            double ee_translation_distance = (ee_position_goal - ee_position).norm();
            double ee_rotational_distance = ee_orientation.normalized().angularDistance(ee_orientation_goal.normalized());

            // Check if goal is reached (within bounds)
            if (isGoalReached(ee_translation_distance, ee_rotational_distance)) {
                ee_position_ref = ee_position_goal;
                ee_orientation_ref = ee_orientation_goal;
                action_result.success = true;
                action_server->setSucceeded(action_result);
                trajectory_start_time = ros::Time(0);
                ee_position_goal = Eigen::Vector3d::Zero();
            } else {
                // Goal not reached, check if new trajectory, otherwise calculate new reference pose based on the time elapsed
                if (trajectory_start_time.isZero()) {
					if (std::isnan(ee_position_start_enabled(0)) && std::isnan(ee_position_start_enabled(1)) && std::isnan(ee_position_start_enabled(2)) && std::isnan(ee_orientation_goal_enabled.w()) && std::isnan(ee_orientation_goal_enabled.x()) && std::isnan(ee_orientation_goal_enabled.y()) && std::isnan(ee_orientation_goal_enabled.z())) {
                    	initializeTrajectoryParameters(ee_position, ee_orientation, ee_translation_distance, ee_rotational_distance);
					} else {
						if (std::isnan(ee_position_start_enabled(0))) {
							ee_position_start(0) = ee_position.x();
						}
						if (std::isnan(ee_position_start_enabled(1))) {
							ee_position_start(1) = ee_position.y();
						}
						if (std::isnan(ee_position_start_enabled(2))) {
							ee_position_start(2) = ee_position.z();
						}
						if (std::isnan(ee_orientation_start_enabled.w())) {
							ee_orientation_start.w() = ee_orientation.w();
						}
						if (std::isnan(ee_orientation_start_enabled.x())) {
							ee_orientation_start.x() = ee_orientation.x();
						}
						if (std::isnan(ee_orientation_start_enabled.y())) {
							ee_orientation_start.y() = ee_orientation.y();
						}
						if (std::isnan(ee_orientation_start_enabled.z())) {
							ee_orientation_start.z() = ee_orientation.z();
						}
                        ee_translation_distance = (ee_position_goal - ee_position_start).norm();
                    	initializeTrajectoryParameters(ee_position_start, ee_orientation_start, ee_translation_distance, ee_rotational_distance);
					}
                }
                updateReferencePose(ee_translation_distance, ee_rotational_distance);
            }
        }

        Eigen::Matrix<double, 6, 1> ee_offset_cart;
//        ee_offset_cart.head(3) << T.linear() * ee_offset_ee; // Offset in end-effector frame
        ee_offset_cart.head(3) << ee_offset_ee; // Pure z-axis offset (experimental conditions)

        // Position error
        ee_pose_error.head(3) << ee_position - ee_position_ref;

        // Orientation error
        if (ee_orientation_ref.coeffs().dot(ee_orientation.coeffs()) < 0.0) {
            ee_orientation.coeffs() << -ee_orientation.coeffs();
        }
        // "Difference" quaternion
        Eigen::Quaterniond error_quaternion(ee_orientation.inverse() * ee_orientation_ref);
        ee_pose_error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

        // Transform to base frame
        ee_pose_error.tail(3) << -T.linear() * ee_pose_error.tail(3);

        publish_trajectory(ee_position_ref, ee_orientation_ref);
        
        // Calculate torques for cartesian impedance control
        tau_task = J_eef.transpose() * (-K * ee_pose_error - D * (J_eef * robot_state.velocity));

        // Calculate torques for null space control
        Eigen::MatrixXd jacobian_eef_transpose_pinv;
        pseudoInverse(J_eef.transpose(), &jacobian_eef_transpose_pinv);
//        tau_nullspace = (Eigen::MatrixXd::Identity(7, 7) - J_eef.transpose() * jacobian_eef_transpose_pinv) * (K_nullspace * (q_nullspace - robot_state.position) - D_nullspace * robot_state.velocity);
        tau_nullspace = (Eigen::MatrixXd::Identity(7, 7) - J_eef.transpose() * jacobian_eef_transpose_pinv) * (K_nullspace * (q_nullspace - robot_state.position) - D_nullspace * (Eigen::MatrixXd::Identity(7, 7) - J_eef.transpose() * jacobian_eef_transpose_pinv) * robot_state.velocity);

        // Calculate torques for impedance based assistance
        if (z_margin_latch != 0.0) {
            if (latched) {
                if ((ee_position(2) < (z_pose - z_margin_unlatch)) || (ee_position(2) > (z_pose + z_margin_unlatch))) {
                    tau_ext = Eigen::VectorXd::Zero(7);
                    latched = false;
                } else {
                    tau_ext = J_eef.transpose() * (K_ext * ee_offset_cart);
                }
            } else {
                if ((ee_position(2) < (z_pose + z_margin_latch)) && (ee_position(2) > (z_pose - z_margin_latch))) {
                    tau_ext = J_eef.transpose() * (K_ext * ee_offset_cart);
                    latched = true;
                } else {
                    tau_ext = Eigen::VectorXd::Zero(7);
                }
            }
        } else {
            tau_ext = J_eef.transpose() * (K_ext * ee_offset_cart);
        }

        // Calculate torques for joint damping
//        tau_damping = J_eef.transpose() * (-D_extra * (J_eef * robot_state.velocity));

        // Get commanded torques
        tau = tau_task + tau_ext + tau_nullspace;

        // Enforce joint limits and command to joint handles
        for (unsigned int i = 0; i < n_joints; ++i) {
            enforceJointLimits(tau[i], i);
            jointsHandles[i].setCommand(tau[i]);
        }

        // Publish data
        publish_joint_state(robot_state);
        publish_cartesian_pose(ee_position, ee_orientation);
        publish_commanded_torque(tau);
        Eigen::MatrixXd jacobian_iiwa_transpose_pinv;
        pseudoInverse(J_eef.transpose(), &jacobian_iiwa_transpose_pinv);
        Eigen::VectorXd cartesian_wrench = jacobian_iiwa_transpose_pinv * robot_state.effort;
        publish_cartesian_wrench(cartesian_wrench);
    }

//    void CartesianImpedanceController::stopping(const ros::Time& time) {
//
//    }

    void CartesianImpedanceController::refPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        ROS_INFO_STREAM("Received reference pose message: " << msg->pose.position.x << " " << msg->pose.position.y << " " << msg->pose.position.z << " " << msg->pose.orientation.w << " " << msg->pose.orientation.x << " " << msg->pose.orientation.y << " " << msg->pose.orientation.z);
        ee_position_ref =  Eigen::Vector3d(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
        ee_orientation_ref = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    }

    void CartesianImpedanceController::dynamicReconfigureCallback(iiwa_impedance_control::CartesianImpedanceControllerConfig &config, uint32_t level) {
        ROS_INFO("Reconfigure Request");
        if (config.pause) {
            ROS_INFO("Abort: pause");
            action_result.success = false;
            if (action_server->isActive()) {
                action_server->setAborted(action_result);
            }
            trajectory_start_time = ros::Time(0);
            ee_position_goal = Eigen::Vector3d::Zero();
            setReferenceToCurrentPose();
            config.translational_stiffness = 300.0;
            config.translational_damping_ratio = 0.7;
            config.rotational_stiffness = 10.0;
            config.rotational_damping_ratio = 0.7;
            config.seperate_axis = false;
            config.pause = false;
            config.sticky_assistance = false;
            config.assistance = false;
        }

        if (!config.seperate_axis) {
            if (K.topLeftCorner(3, 3) != config.translational_stiffness * Eigen::Matrix3d::Identity()) {
                K.topLeftCorner(3, 3) << config.translational_stiffness * Eigen::Matrix3d::Identity(3, 3);
                D.topLeftCorner(3, 3)
                        << 2.0 * config.translational_damping_ratio * sqrt(config.translational_stiffness) *
                           Eigen::Matrix3d::Identity();
            }
            if (K.bottomRightCorner(3, 3) != config.rotational_stiffness * Eigen::Matrix3d::Identity()) {
                K.bottomRightCorner(3, 3) << config.rotational_stiffness * Eigen::Matrix3d::Identity();
                D.bottomRightCorner(3, 3) << 2.0 * config.rotational_damping_ratio * sqrt(config.rotational_stiffness) *
                                             Eigen::Matrix3d::Identity();
            }
            if (translational_damping_ratio != config.translational_damping_ratio) {
                translational_damping_ratio = config.translational_damping_ratio;
                D.topLeftCorner(3, 3) << 2.0 * translational_damping_ratio * sqrt(config.translational_stiffness) *
                                         Eigen::Matrix3d::Identity();
            }
            if (rotational_damping_ratio != config.rotational_damping_ratio) {
                rotational_damping_ratio = config.rotational_damping_ratio;
                D.bottomRightCorner(3, 3) << 2.0 * rotational_damping_ratio * sqrt(config.rotational_stiffness) *
                                             Eigen::Matrix3d::Identity();
            }
        }

        if (config.seperate_axis) {
            if (K(0, 0) != config.translational_stiffness_x) {
                K(0, 0) = config.translational_stiffness_x;
                D(0, 0) = 2.0 * config.translational_damping_ratio * sqrt(config.translational_stiffness_x);
            }
            if (K(1, 1) != config.translational_stiffness_y) {
                K(1, 1) = config.translational_stiffness_y;
                D(1, 1) = 2.0 * config.translational_damping_ratio * sqrt(config.translational_stiffness_y);
            }
            if (K(2, 2) != config.translational_stiffness_z) {
                K(2, 2) = config.translational_stiffness_z;
                D(2, 2) = 2.0 * config.translational_damping_ratio * sqrt(config.translational_stiffness_z);
            }
            if (K(3, 3) != config.rotational_stiffness_alpha) {
                K(3, 3) = config.rotational_stiffness_alpha;
                D(3, 3) = 2.0 * config.rotational_damping_ratio * sqrt(config.rotational_stiffness_alpha);
            }
            if (K(4, 4) != config.rotational_stiffness_theta) {
                K(4, 4) = config.rotational_stiffness_theta;
                D(4, 4) = 2.0 * config.rotational_damping_ratio * sqrt(config.rotational_stiffness_theta);
            }
            if (K(5, 5) != config.rotational_stiffness_phi) {
                K(5, 5) = config.rotational_stiffness_phi;
                D(5, 5) = 2.0 * config.rotational_damping_ratio * sqrt(config.rotational_stiffness_phi);
            }
            if (translational_damping_ratio != config.translational_damping_ratio) {
                translational_damping_ratio = config.translational_damping_ratio;
                Eigen::Matrix3d D_temp = Eigen::Matrix3d::Identity();
                D(0, 0) = 2.0 * translational_damping_ratio * sqrt(config.translational_stiffness_x);
                D(1, 1) = 2.0 * translational_damping_ratio * sqrt(config.translational_stiffness_y);
                D(2, 2) = 2.0 * translational_damping_ratio * sqrt(config.translational_stiffness_z);
                D.topLeftCorner(3, 3) << D_temp;
            }
            if (rotational_damping_ratio != config.rotational_damping_ratio) {
                rotational_damping_ratio = config.rotational_damping_ratio;
                Eigen::Matrix3d D_temp = Eigen::Matrix3d::Identity();
                D(3, 3) = 2.0 * rotational_damping_ratio * sqrt(config.rotational_stiffness_alpha);
                D(4, 4) = 2.0 * rotational_damping_ratio * sqrt(config.rotational_stiffness_theta);
                D(5, 5) = 2.0 * rotational_damping_ratio * sqrt(config.rotational_stiffness_phi);
                D.bottomRightCorner(3, 3) << D_temp;
            }
        }

        if (config.nullspace_control) {
            if (q_nullspace(0) != config.q_nullspace_joint_1) {
                q_nullspace(0) = config.q_nullspace_joint_1;
            }
            if (q_nullspace(1) != config.q_nullspace_joint_2) {
                q_nullspace(1) = config.q_nullspace_joint_2;
            }
            if (q_nullspace(2) != config.q_nullspace_joint_3) {
                q_nullspace(2) = config.q_nullspace_joint_3;
            }
            if (q_nullspace(3) != config.q_nullspace_joint_4) {
                q_nullspace(3) = config.q_nullspace_joint_4;
            }
            if (q_nullspace(4) != config.q_nullspace_joint_5) {
                q_nullspace(4) = config.q_nullspace_joint_5;
            }
            if (q_nullspace(5) != config.q_nullspace_joint_6) {
                q_nullspace(5) = config.q_nullspace_joint_6;
            }
            if (q_nullspace(6) != config.q_nullspace_joint_7) {
                q_nullspace(6) = config.q_nullspace_joint_7;
            }

            if (config.nullspace_stiffness_joint_1 != K_nullspace(0, 0)) {
                K_nullspace(0, 0) = config.nullspace_stiffness_joint_1;
                D_nullspace(0, 0) =
                        2.0 * config.nullspace_damping_ratio_joint_1 * sqrt(config.nullspace_stiffness_joint_1);
            }
            if (config.nullspace_damping_ratio_joint_1 != nullspace_damping_ratio_joint_1) {
                nullspace_damping_ratio_joint_1 = config.nullspace_damping_ratio_joint_1;
                D_nullspace(0, 0) = 2.0 * nullspace_damping_ratio_joint_1 * sqrt(config.nullspace_stiffness_joint_1);
            }
            if (config.nullspace_stiffness_joint_2 != K_nullspace(1, 1)) {
                K_nullspace(1, 1) = config.nullspace_stiffness_joint_2;
                D_nullspace(1, 1) =
                        2.0 * config.nullspace_damping_ratio_joint_2 * sqrt(config.nullspace_stiffness_joint_2);
            }
            if (config.nullspace_damping_ratio_joint_2 != nullspace_damping_ratio_joint_2) {
                nullspace_damping_ratio_joint_2 = config.nullspace_damping_ratio_joint_2;
                D_nullspace(1, 1) = 2.0 * nullspace_damping_ratio_joint_2 * sqrt(config.nullspace_stiffness_joint_2);
            }
            if (config.nullspace_stiffness_joint_3 != K_nullspace(2, 2)) {
                K_nullspace(2, 2) = config.nullspace_stiffness_joint_3;
                D_nullspace(2, 2) =
                        2.0 * config.nullspace_damping_ratio_joint_3 * sqrt(config.nullspace_stiffness_joint_3);
            }
            if (config.nullspace_damping_ratio_joint_3 != nullspace_damping_ratio_joint_3) {
                nullspace_damping_ratio_joint_3 = config.nullspace_damping_ratio_joint_3;
                D_nullspace(2, 2) = 2.0 * nullspace_damping_ratio_joint_3 * sqrt(config.nullspace_stiffness_joint_3);
            }
            if (config.nullspace_stiffness_joint_4 != K_nullspace(3, 3)) {
                K_nullspace(3, 3) = config.nullspace_stiffness_joint_4;
                D_nullspace(3, 3) =
                        2.0 * config.nullspace_damping_ratio_joint_4 * sqrt(config.nullspace_stiffness_joint_4);
            }
            if (config.nullspace_damping_ratio_joint_4 != nullspace_damping_ratio_joint_4) {
                nullspace_damping_ratio_joint_4 = config.nullspace_damping_ratio_joint_4;
                D_nullspace(3, 3) = 2.0 * nullspace_damping_ratio_joint_4 * sqrt(config.nullspace_stiffness_joint_4);
            }
            if (config.nullspace_stiffness_joint_5 != K_nullspace(4, 4)) {
                K_nullspace(4, 4) = config.nullspace_stiffness_joint_5;
                D_nullspace(4, 4) =
                        2.0 * config.nullspace_damping_ratio_joint_5 * sqrt(config.nullspace_stiffness_joint_5);
            }
            if (config.nullspace_damping_ratio_joint_5 != nullspace_damping_ratio_joint_5) {
                nullspace_damping_ratio_joint_5 = config.nullspace_damping_ratio_joint_5;
                D_nullspace(4, 4) = 2.0 * nullspace_damping_ratio_joint_5 * sqrt(config.nullspace_stiffness_joint_5);
            }
            if (config.nullspace_stiffness_joint_6 != K_nullspace(5, 5)) {
                K_nullspace(5, 5) = config.nullspace_stiffness_joint_6;
                D_nullspace(5, 5) =
                        2.0 * config.nullspace_damping_ratio_joint_6 * sqrt(config.nullspace_stiffness_joint_6);
            }
            if (config.nullspace_damping_ratio_joint_6 != nullspace_damping_ratio_joint_6) {
                nullspace_damping_ratio_joint_6 = config.nullspace_damping_ratio_joint_6;
                D_nullspace(5, 5) = 2.0 * nullspace_damping_ratio_joint_6 * sqrt(config.nullspace_stiffness_joint_6);
            }
            if (config.nullspace_stiffness_joint_7 != K_nullspace(6, 6)) {
                K_nullspace(6, 6) = config.nullspace_stiffness_joint_7;
                D_nullspace(6, 6) =
                        2.0 * config.nullspace_damping_ratio_joint_7 * sqrt(config.nullspace_stiffness_joint_7);
            }
            if (config.nullspace_damping_ratio_joint_7 != nullspace_damping_ratio_joint_7) {
                nullspace_damping_ratio_joint_7 = config.nullspace_damping_ratio_joint_7;
                D_nullspace(6, 6) = 2.0 * nullspace_damping_ratio_joint_7 * sqrt(config.nullspace_stiffness_joint_7);
            }
        } else {
            K_nullspace.setZero(7, 7);
            D_nullspace.setZero(7, 7);
        }


        if (config.set_new_reference_pose && ((int) config.set_new_reference_pose != previous_set_new_reference_pose)) {
            config.set_new_reference_pose = false;
            ROS_INFO("New reference pose requested:");
            setReferenceToCurrentPose();
        }
        if (config.set_new_home_pose && ((int) config.set_new_home_pose != previous_set_new_home_pose)) {
            config.set_new_home_pose = false;
            ROS_INFO("New home pose requested:");
            setHomeToCurrentReference();
        }
        if (config.set_reference_to_home_pose &&
            ((int) config.set_reference_to_home_pose != previous_set_reference_to_home_pose)) {
            config.set_reference_to_home_pose = false;
            ROS_INFO("New reference to home pose requested:");
            setReferenceToHome();
        }

        previous_set_new_reference_pose = (int) config.set_new_reference_pose;
        previous_set_new_home_pose = (int) config.set_new_home_pose;
        previous_set_reference_to_home_pose = (int) config.set_reference_to_home_pose;

        if (config.sticky_assistance) {
            if (!previous_assistance) {
                config.assistance = true;
            }
            z_pose = config.sticky_assistance_z_pose;
            z_margin_latch = config.sticky_assistance_z_margin_latch;
            z_margin_unlatch = config.sticky_assistance_z_margin_unlatch;
        } else {
            z_pose = 0.0;
            z_margin_latch = 0.0;
            z_margin_unlatch = 0.0;
            latched = false;
        }

        if (config.assistance) {
            ee_offset_ee[0] = 0; // x
            ee_offset_ee[1] = 0; // y
//            ee_offset_ee[2] = config.assistance_z_offset; // for z-axis offset in end-effector frame
            ee_offset_ee[2] = -config.assistance_z_offset; // for pure z-axis offset (experimental conditions)
            if (K_ext.topLeftCorner(3, 3) != config.assistance_stiffness * Eigen::Matrix3d::Identity()) {
                K_ext.topLeftCorner(3, 3) << config.assistance_stiffness * Eigen::Matrix3d::Identity(3, 3);
            }
            K_ext.bottomRightCorner(3, 3) << Eigen::Matrix3d::Zero();
        } else {
            ee_offset_ee[0] = 0; // x
            ee_offset_ee[1] = 0; // y
            ee_offset_ee[2] = 0; // z
            K_ext.topLeftCorner(3, 3) << Eigen::Matrix3d::Zero();
            K_ext.bottomRightCorner(3, 3) << Eigen::Matrix3d::Zero();
            config.sticky_assistance = false;
            z_pose = 0.0;
            z_margin_latch = 0.0;
            z_margin_unlatch = 0.0;
            latched = false;
        }

        previous_assistance = (bool) config.assistance;

//        ROS_INFO("K:");
//        ROS_INFO(
//                "\n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f",
//                K(0, 0), K(0, 1), K(0, 2), K(0, 3), K(0, 4), K(0, 5),
//                K(1, 0), K(1, 1), K(1, 2), K(1, 3), K(1, 4), K(1, 5),
//                K(2, 0), K(2, 1), K(2, 2), K(2, 3), K(2, 4), K(2, 5),
//                K(3, 0), K(3, 1), K(3, 2), K(3, 3), K(3, 4), K(3, 5),
//                K(4, 0), K(4, 1), K(4, 2), K(4, 3), K(4, 4), K(4, 5),
//                K(5, 0), K(5, 1), K(5, 2), K(5, 3), K(5, 4), K(5, 5));
//        ROS_INFO("D:");
//        ROS_INFO(
//                "\n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f",
//                D(0, 0), D(0, 1), D(0, 2), D(0, 3), D(0, 4), D(0, 5),
//                D(1, 0), D(1, 1), D(1, 2), D(1, 3), D(1, 4), D(1, 5),
//                D(2, 0), D(2, 1), D(2, 2), D(2, 3), D(2, 4), D(2, 5),
//                D(3, 0), D(3, 1), D(3, 2), D(3, 3), D(3, 4), D(3, 5),
//                D(4, 0), D(4, 1), D(4, 2), D(4, 3), D(4, 4), D(4, 5),
//                D(5, 0), D(5, 1), D(5, 2), D(5, 3), D(5, 4), D(5, 5));
//        ROS_INFO("K_nullspace:");
//        ROS_INFO(
//                "\n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f",
//                K_nullspace(0, 0), K_nullspace(0, 1), K_nullspace(0, 2), K_nullspace(0, 3), K_nullspace(0, 4),
//                K_nullspace(0, 5),
//                K_nullspace(1, 0), K_nullspace(1, 1), K_nullspace(1, 2), K_nullspace(1, 3), K_nullspace(1, 4),
//                K_nullspace(1, 5),
//                K_nullspace(2, 0), K_nullspace(2, 1), K_nullspace(2, 2), K_nullspace(2, 3), K_nullspace(2, 4),
//                K_nullspace(2, 5),
//                K_nullspace(3, 0), K_nullspace(3, 1), K_nullspace(3, 2), K_nullspace(3, 3), K_nullspace(3, 4),
//                K_nullspace(3, 5),
//                K_nullspace(4, 0), K_nullspace(4, 1), K_nullspace(4, 2), K_nullspace(4, 3), K_nullspace(4, 4),
//                K_nullspace(4, 5),
//                K_nullspace(5, 0), K_nullspace(5, 1), K_nullspace(5, 2), K_nullspace(5, 3), K_nullspace(5, 4),
//                K_nullspace(5, 5));
//        ROS_INFO("D_nullspace:");
//        ROS_INFO(
//                "\n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f \n %f %f %f %f %f %f",
//                D_nullspace(0, 0), D_nullspace(0, 1), D_nullspace(0, 2), D_nullspace(0, 3), D_nullspace(0, 4),
//                D_nullspace(0, 5),
//                D_nullspace(1, 0), D_nullspace(1, 1), D_nullspace(1, 2), D_nullspace(1, 3), D_nullspace(1, 4),
//                D_nullspace(1, 5),
//                D_nullspace(2, 0), D_nullspace(2, 1), D_nullspace(2, 2), D_nullspace(2, 3), D_nullspace(2, 4),
//                D_nullspace(2, 5),
//                D_nullspace(3, 0), D_nullspace(3, 1), D_nullspace(3, 2), D_nullspace(3, 3), D_nullspace(3, 4),
//                D_nullspace(3, 5),
//                D_nullspace(4, 0), D_nullspace(4, 1), D_nullspace(4, 2), D_nullspace(4, 3), D_nullspace(4, 4),
//                D_nullspace(4, 5),
//                D_nullspace(5, 0), D_nullspace(5, 1), D_nullspace(5, 2), D_nullspace(5, 3), D_nullspace(5, 4),
//                D_nullspace(5, 5));
//        ROS_INFO("q_nullspace:");
//        ROS_INFO("\n %f %f %f %f %f %f %f",
//                 q_nullspace(0), q_nullspace(1), q_nullspace(2), q_nullspace(3), q_nullspace(4), q_nullspace(5), q_nullspace(6));
    }

    void CartesianImpedanceController::goalCallback() {
        ROS_INFO("New goal received");
        // Initialize goal and start values to NaN
        double ee_position_goal_x = std::numeric_limits<double>::quiet_NaN();
        double ee_position_goal_y = std::numeric_limits<double>::quiet_NaN();
        double ee_position_goal_z = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_goal_w = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_goal_x = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_goal_y = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_goal_z = std::numeric_limits<double>::quiet_NaN();
        double ee_position_start_x = std::numeric_limits<double>::quiet_NaN();
        double ee_position_start_y = std::numeric_limits<double>::quiet_NaN();
        double ee_position_start_z = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_start_w = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_start_x = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_start_y = std::numeric_limits<double>::quiet_NaN();
        double ee_orientation_start_z = std::numeric_limits<double>::quiet_NaN();
        boost::shared_ptr<const actionlib::SimpleActionServer<iiwa_impedance_control::CartesianTrajectoryExecutionAction>::Goal> newGoal = action_server->acceptNewGoal();
        if (newGoal) {
            geometry_msgs::PoseStamped::ConstPtr pose_goal = boost::make_shared<const geometry_msgs::PoseStamped>(newGoal->pose_goal);
            geometry_msgs::PoseStamped::ConstPtr pose_start = boost::make_shared<const geometry_msgs::PoseStamped>(newGoal->pose_start);
            boost::shared_ptr<const double> translational_velocity_goal = boost::make_shared<const double>(newGoal->translational_velocity_goal);
            boost::shared_ptr<const double> rotational_velocity_goal = boost::make_shared<const double>(newGoal->rotational_velocity_goal);
            // Check if pose_goal values are not None before assigning
            if (pose_goal->pose.position.x != 99) {
                ee_position_goal_x = pose_goal->pose.position.x;
            }
            if (pose_goal->pose.position.y != 99) {
                ee_position_goal_y = pose_goal->pose.position.y;
            }
            if (pose_goal->pose.position.z != 99) {
                ee_position_goal_z = pose_goal->pose.position.z;
            }
            if (pose_goal->pose.orientation.w != 99) {
                ee_orientation_goal_w = pose_goal->pose.orientation.w;
            }
            if (pose_goal->pose.orientation.x != 99) {
                ee_orientation_goal_x = pose_goal->pose.orientation.x;
            }
            if (pose_goal->pose.orientation.y != 99) {
                ee_orientation_goal_y = pose_goal->pose.orientation.y;
            }
            if (pose_goal->pose.orientation.z != 99) {
                ee_orientation_goal_z = pose_goal->pose.orientation.z;
            }
            // Check if pose_start values are not None before assigning
            if (pose_start->pose.position.x != 99) {
                ee_position_start_x = pose_start->pose.position.x;
            }
            if (pose_start->pose.position.y != 99) {
                ee_position_start_y = pose_start->pose.position.y;
            }
            if (pose_start->pose.position.z != 99) {
                ee_position_start_z = pose_start->pose.position.z;
            }
            if (pose_start->pose.orientation.w != 99) {
                ee_orientation_start_w = pose_start->pose.orientation.w;
            }
            if (pose_start->pose.orientation.x != 99) {
                ee_orientation_start_x = pose_start->pose.orientation.x;
            }
            if (pose_start->pose.orientation.y != 99) {
                ee_orientation_start_y = pose_start->pose.orientation.y;
            }
            if (pose_start->pose.orientation.z != 99) {
                ee_orientation_start_z = pose_start->pose.orientation.z;
            }
            ee_position_goal = Eigen::Vector3d(ee_position_goal_x, ee_position_goal_y, ee_position_goal_z);
            ee_orientation_goal = Eigen::Quaterniond(ee_orientation_goal_w, ee_orientation_goal_x, ee_orientation_goal_y, ee_orientation_goal_z);
            ee_position_goal_enabled = Eigen::Vector3d(ee_position_goal_x, ee_position_goal_y, ee_position_goal_z);
            ee_orientation_goal_enabled = Eigen::Quaterniond(ee_orientation_goal_w, ee_orientation_goal_x, ee_orientation_goal_y, ee_orientation_goal_z);

			ee_position_start = Eigen::Vector3d(ee_position_start_x, ee_position_start_y, ee_position_start_z);
            ee_orientation_start = Eigen::Quaterniond(ee_orientation_start_w, ee_orientation_start_x, ee_orientation_start_y, ee_orientation_start_z);
            ee_position_start_enabled = Eigen::Vector3d(ee_position_start_x, ee_position_start_y, ee_position_start_z);
            ee_orientation_start_enabled = Eigen::Quaterniond(ee_orientation_start_w, ee_orientation_start_x, ee_orientation_start_y, ee_orientation_start_z);
            v_ref_translation = *translational_velocity_goal;
            v_ref_rotation = *rotational_velocity_goal;
            ROS_INFO_STREAM("Received action goal: " << pose_goal->pose.position.x << " " << pose_goal->pose.position.y << " " << pose_goal->pose.position.z << " " << pose_goal->pose.orientation.w << " " << pose_goal->pose.orientation.x << " " << pose_goal->pose.orientation.y << " " << pose_goal->pose.orientation.z << " with translational velocity: " << v_ref_translation << " and rotational velocity: " << v_ref_rotation);
            ROS_INFO_STREAM("Received action start: " << pose_start->pose.position.x << " " << pose_start->pose.position.y << " " << pose_start->pose.position.z << " " << pose_start->pose.orientation.w << " " << pose_start->pose.orientation.x << " " << pose_start->pose.orientation.y << " " << pose_start->pose.orientation.z);
        } else {
            ROS_ERROR("No new goal available to accept");
        }
    }

    void CartesianImpedanceController::preemptCallback() {
        ROS_INFO("Preempt received");
        action_server->setPreempted();
        trajectory_start_time = ros::Time(0);
        ee_position_goal = Eigen::Vector3d::Zero();
		ee_position_start = Eigen::Vector3d::Zero();
    }

    bool CartesianImpedanceController::isGoalReached(double translation_distance, double rotational_distance) {
        return (translation_distance < goal_translation_tolerance && rotational_distance < goal_rotation_tolerance);
    }

    void CartesianImpedanceController::updateReferencePose(double ee_translation_distance, double ee_rotational_distance) {
        double elapsed_time = (ros::Time::now() - trajectory_start_time).toSec();
        // Provide action feedback
        action_feedback.time_progress = (elapsed_time / total_time);
        action_feedback.distance_progress = 1 - std::min(ee_translation_distance / total_distance_translation, ee_rotational_distance / total_distance_rotation);
        action_server->publishFeedback(action_feedback);
        if (action_feedback.time_progress > time_limit) {
            ROS_INFO("Abort > time_limit");
            action_result.success = false;
            action_server->setAborted(action_result);
            trajectory_start_time = ros::Time(0);
            ee_position_goal = Eigen::Vector3d::Zero();
            return;
        }
        if (elapsed_time < total_time) {
            // Update reference pose
            ee_position_ref = start_position + translation_vector * v_ref_translation * time_factor_translation * elapsed_time;
            ee_orientation_ref = start_orientation.slerp(v_ref_rotation * time_factor_rotation * elapsed_time, ee_orientation_goal);
        } else {
            // Trajectory should be finished, set reference pose to goal pose
            ee_position_ref = ee_position_goal;
            ee_orientation_ref = ee_orientation_goal;
        }
        publish_trajectory(ee_position_ref, ee_orientation_ref);

    }

    void CartesianImpedanceController::initializeTrajectoryParameters(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation, double ee_translation_distance, double ee_rotational_distance) {
        //TODO: move to (async) callback instead of main loop
        ROS_INFO("Setting new trajectory parameters");
        // New trajectory, determine trajectory parameters
        trajectory_start_time = ros::Time::now();
        start_position = ee_position;
        translation_vector = (ee_position_goal - start_position).normalized();
        start_orientation = ee_orientation;
        end_orientation = ee_orientation_goal.normalized();

        total_distance_translation = ee_translation_distance;
        total_distance_rotation = ee_rotational_distance;
        total_time_translation = ee_translation_distance / v_ref_translation;
        total_time_rotation = ee_rotational_distance / v_ref_rotation;
        total_time = std::max(total_time_translation, total_time_rotation);

        time_factor_translation = 1;
        time_factor_rotation = 1;
        if (total_time_translation > total_time_rotation) {
            time_factor_rotation = total_time_rotation / total_time_translation;
        } else if (total_time_rotation > total_time_translation) {
            time_factor_translation = total_time_translation / total_time_rotation;
        }
    }

    RobotState CartesianImpedanceController::updateRobotState() {
        RobotState robot_state;
        robot_state.position.resize(n_joints);
        robot_state.velocity.resize(n_joints);
        robot_state.effort.resize(n_joints);

        for(unsigned int i = 0; i < n_joints; ++i) {
            robot_state.position[i] = jointsHandles[i].getPosition();
            robot_state.velocity[i] = jointsHandles[i].getVelocity();
            robot_state.effort[i] = jointsHandles[i].getEffort();
        }

        return robot_state;
    }

    void CartesianImpedanceController::enforceJointLimits(double& command, unsigned int index) {
        if (command > joint_urdfs[index]->limits->effort) { // above upper limit
            command = joint_urdfs[index]->limits->effort;
        } else if (command < -joint_urdfs[index]->limits->effort) { // below lower limit
            command = -joint_urdfs[index]->limits->effort;
        }
    }

    void CartesianImpedanceController::setReferenceToCurrentPose() {
        RobotState robot_state = updateRobotState();

        // Get end effector pose in cartesian space (base_frame)
        Eigen::Affine3d T = getForwardKinematics(robot_state.position, eef_chain);
        ee_position_ref = T.translation();
        ee_orientation_ref = T.linear();
        ROS_INFO("Reference position and orientation:");
        ROS_INFO("%f %f %f", ee_position_ref(0), ee_position_ref(1), ee_position_ref(2));
        ROS_INFO("%f %f %f %f", ee_orientation_ref.coeffs()(0), ee_orientation_ref.coeffs()(1), ee_orientation_ref.coeffs()(2), ee_orientation_ref.coeffs()(3));
    }

    void CartesianImpedanceController::setReferenceToHome() {
        ee_position_ref = ee_position_home;
        ee_orientation_ref = ee_orientation_home;
        ROS_INFO("Reference position and orientation:");
        ROS_INFO("%f %f %f", ee_position_ref(0), ee_position_ref(1), ee_position_ref(2));
        ROS_INFO("%f %f %f %f", ee_orientation_ref.coeffs()(0), ee_orientation_ref.coeffs()(1), ee_orientation_ref.coeffs()(2), ee_orientation_ref.coeffs()(3));
    }

    void CartesianImpedanceController::setHomeToCurrentReference() {
        ee_position_home = ee_position_ref;
        ee_orientation_home = ee_orientation_ref;
        ROS_INFO("Home position and orientation:");
        ROS_INFO("%f %f %f", ee_position_home(0), ee_position_home(1), ee_position_home(2));
        ROS_INFO("%f %f %f %f", ee_orientation_home.coeffs()(0), ee_orientation_home.coeffs()(1), ee_orientation_home.coeffs()(2), ee_orientation_ref.coeffs()(3));
    }

    KDL::Jacobian CartesianImpedanceController::getJacobian(Eigen::VectorXd position, KDL::Chain chain) {
        KDL::JntArray q;
        KDL::Jacobian J(7);
        KDL::ChainJntToJacSolver solver(chain);

        q.data = position;
        int error = solver.JntToJac(q, J);
        if (error != KDL::SolverI::E_NOERROR) {
            ROS_ERROR("KDL zero jacobian calculation failed with error: %s", std::strerror(error));
        }
        return J;
    }

    Eigen::Affine3d CartesianImpedanceController::getForwardKinematics(Eigen::VectorXd position, KDL::Chain chain) {
        KDL::JntArray q;
        KDL::Frame frame;
        Eigen::Affine3d T;
        KDL::ChainFkSolverPos_recursive solver(chain);

        q.data = position;
        int error = solver.JntToCart(q, frame);
        if (error != KDL::SolverI::E_NOERROR) {
            ROS_ERROR("KDL forward kinematics pose calculation failed with error: %s", std::strerror(error));
        }

        tf::transformKDLToEigen(frame, T);
        return T;
    }

    void CartesianImpedanceController::publish_joint_state(RobotState robot_state) {
        sensor_msgs::JointState msg;
        std::vector<double> position(robot_state.position.data(), robot_state.position.data() + robot_state.position.size());
        std::vector<double> velocity(robot_state.velocity.data(), robot_state.velocity.data() + robot_state.velocity.size());
        std::vector<double> effort(robot_state.effort.data(), robot_state.effort.data() + robot_state.effort.size());

        msg.header.stamp = ros::Time::now();
        msg.position = position;
        msg.velocity = velocity;
        msg.effort = effort;

        joint_state_pub.publish(msg);
    }

    void CartesianImpedanceController::publish_cartesian_pose(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation) {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.pose.position.x = ee_position.x();
        msg.pose.position.y = ee_position.y();
        msg.pose.position.z = ee_position.z();
        msg.pose.orientation.x = ee_orientation.x();
        msg.pose.orientation.y = ee_orientation.y();
        msg.pose.orientation.z = ee_orientation.z();
        msg.pose.orientation.w = ee_orientation.w();

        cartesian_pose_pub.publish(msg);
    }

    void CartesianImpedanceController::publish_trajectory(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation) {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.pose.position.x = ee_position.x();
        msg.pose.position.y = ee_position.y();
        msg.pose.position.z = ee_position.z();
        msg.pose.orientation.x = ee_orientation.x();
        msg.pose.orientation.y = ee_orientation.y();
        msg.pose.orientation.z = ee_orientation.z();
        msg.pose.orientation.w = ee_orientation.w();

        trajectory_pub.publish(msg);
    }

    void CartesianImpedanceController::publish_commanded_torque(Eigen::MatrixXd commanded_torque) {
        std_msgs::Float64MultiArray msg;
        std::vector<double> commanded_torque_data(7, 0.);
        Eigen::VectorXd::Map(commanded_torque_data.data(), commanded_torque_data.size()) = commanded_torque;
        msg.data = commanded_torque_data;
        commanded_torque_pub.publish(msg);
    }

    void CartesianImpedanceController::publish_cartesian_wrench(Eigen::VectorXd cartesian_wrench) {
        geometry_msgs::WrenchStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.wrench.force.x = cartesian_wrench(0);
        msg.wrench.force.y = cartesian_wrench(1);
        msg.wrench.force.z = cartesian_wrench(2);
        msg.wrench.torque.x = cartesian_wrench(3);
        msg.wrench.torque.y = cartesian_wrench(4);
        msg.wrench.torque.z = cartesian_wrench(5);

        cartesian_wrench_pub.publish(msg);
    }

}//namespace effort_controllers

PLUGINLIB_EXPORT_CLASS(cartesian_impedance_controller::CartesianImpedanceController, controller_interface::ControllerBase);