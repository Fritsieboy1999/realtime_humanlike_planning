//
// Created by Nicky Mol on 10/08/2022.
// Email: nicky.mol@tudelft.nl
//

#include "iiwa_impedance_control/joint_impedance_controller.h"


namespace joint_impedance_controller {

    JointImpedanceController::JointImpedanceController() {}
    JointImpedanceController::~JointImpedanceController() {}

    bool JointImpedanceController::init(hardware_interface::EffortJointInterface* effort_joint_interface, ros::NodeHandle& node_handle) {
        nh = node_handle;
        // Init stiffness matrix
        K.setIdentity(7, 7);
        // Init damping matrix
        D.setIdentity(7, 7);
        // Init q_d
        q_d = Eigen::VectorXd::Zero(7);

        // Get the URDF XML from the parameter server
        std::string urdf_string, full_param;
        std::string robot_description = "robot_description";

        // Gets the location of the robot description on the parameter server
        if (!nh.searchParam(robot_description, full_param)) {
            ROS_ERROR("Could not find parameter %s on parameter server", robot_description.c_str());
            return false;
        }

        // Search and wait for robot_description on param server
        while (urdf_string.empty()) {
            ROS_INFO("JointImpedanceController is waiting for model URDF in parameter [%s] on the ROS param server.",
                     robot_description.c_str());

            nh.getParam(full_param, urdf_string);

            usleep(100000);
        }
        ROS_INFO_STREAM("Received urdf from param server, parsing...");

        // Get URDF
        urdf::Model robot;
        if (not robot.initParam(robot_description)) {
            ROS_ERROR("Failed to parse urdf file");
            return false;
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
        joint_state_pub = nh.advertise<sensor_msgs::JointState>("/JointImpedanceController/joint_states", 100);
        trajectory_pub = nh.advertise<std_msgs::Float64MultiArray>("/JointImpedanceController/trajectory", 500);
        commanded_torque_pub = nh.advertise<std_msgs::Float64MultiArray>("/JointImpedanceController/commanded_torque", 500);

        // Dynamic reconfigure
        dynamic_reconfigure_server_node = ros::NodeHandle(nh.getNamespace() + "/dynamic_reconfigure_server_node");
        dynamic_reconfigure_server = std::make_unique<dynamic_reconfigure::Server<iiwa_impedance_control::JointImpedanceControllerConfig>>(dynamic_reconfigure_server_node);
        dynamic_reconfigure_server->setCallback(boost::bind(&JointImpedanceController::dynamicReconfigureCallback, this, _1, _2));

        // Action server
        action_server = std::unique_ptr<actionlib::SimpleActionServer<iiwa_impedance_control::JointTrajectoryExecutionAction>>(
                new actionlib::SimpleActionServer<iiwa_impedance_control::JointTrajectoryExecutionAction>(
                        nh, std::string("joint_trajectory_execution_action"), false));
        action_server->registerGoalCallback(boost::bind(&JointImpedanceController::goalCallback, this));
        action_server->registerPreemptCallback(boost::bind(&JointImpedanceController::preemptCallback, this));
        action_server->start();
        nh.param<double>("/JointTrajectoryGenerator/time_limit", time_limit, 2.0);
        std::vector<double> goal_tolerance_vector;
        nh.param<std::vector<double>>("/JointImpedanceController/goal_tolerance", goal_tolerance_vector, {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
        goal_tolerances = Eigen::VectorXd::Map(goal_tolerance_vector.data(), goal_tolerance_vector.size());
        nh.param<double>("/JointTrajectoryGenerator/time_limit", time_limit, 2.0);

        return true;
    }

    void JointImpedanceController::starting(const ros::Time& time) {
        setReferenceToCurrentPose();
        setHomeToCurrentReference();
    }

    void JointImpedanceController::update(const ros::Time& time, const ros::Duration& period) {
        Eigen::VectorXd tau;

        RobotState robot_state = updateRobotState();

        // If trajectory controller is active, update reference joint positions
        if (action_server->isActive() && v_refs != Eigen::VectorXd::Zero(7)) {
            // Distance
            Eigen::VectorXd distance = joint_position_goals - robot_state.position;

            // Check if goal is reached (within bounds)
            if (isGoalReached(distance)) {
                q_d = joint_position_goals;
                action_result.success = true;
                action_server->setSucceeded(action_result);
                trajectory_start_time = ros::Time(0);
                joint_position_goals = Eigen::VectorXd::Zero(7);
                v_refs = Eigen::VectorXd::Zero(7);
                updateDynamicReconfigureConfig(q_d);
            } else {
                // Goal not reached, check if new trajectory, otherwise calculate new reference joint positions based on the time elapsed
                if (trajectory_start_time.isZero()) {
                    initializeTrajectoryParameters(robot_state.position, distance);
                }
                updateReferencePose(distance);
            }
        }

        // Get commanded torques
        tau = K * (q_d - robot_state.position) - D * robot_state.velocity;

        // Enforce joint limits and command to joint handles
        for (unsigned int i = 0; i < n_joints; ++i) {
            enforceJointLimits(tau[i], i);
            jointsHandles[i].setCommand(tau[i]);
        }

        // Publish data
        publish_joint_state(robot_state);
        publish_commanded_torque(tau);
    }

//    void JointImpedanceController::stopping(const ros::Time& time) {
//
//    }

    void JointImpedanceController::dynamicReconfigureCallback(iiwa_impedance_control::JointImpedanceControllerConfig &config, uint32_t level) {
        ROS_INFO("Reconfigure Request");

        if (!config.seperate_joints) {
            if (K != config.stiffness * Eigen::MatrixXd::Identity(7, 7)) {
                K << config.stiffness * Eigen::MatrixXd::Identity(7, 7);
                D << 2.0 * config.damping_ratio * sqrt(config.stiffness) * Eigen::MatrixXd::Identity(7, 7);
            }
            if (damping_ratio != config.damping_ratio) {
                damping_ratio = config.damping_ratio;
                D << 2.0 * config.damping_ratio * sqrt(config.stiffness) * Eigen::MatrixXd::Identity(7, 7);
            }
        }

        if (config.seperate_joints) {
            if ((K(0, 0) != config.stiffness_joint_1) || (D(0, 0) != 2.0 * config.damping_ratio_joint_1 * sqrt(config.stiffness_joint_1))) {
                K(0, 0) = config.stiffness_joint_1;
                D(0, 0) = 2.0 * config.damping_ratio_joint_1 * sqrt(config.stiffness_joint_1);
            }
            if ((K(1, 1) != config.stiffness_joint_2) || (D(1, 1) != 2.0 * config.damping_ratio_joint_2 * sqrt(config.stiffness_joint_2))) {
                K(1, 1) = config.stiffness_joint_2;
                D(1, 1) = 2.0 * config.damping_ratio_joint_2 * sqrt(config.stiffness_joint_2);
            }
            if ((K(2, 2) != config.stiffness_joint_3) || (D(2, 2) != 2.0 * config.damping_ratio_joint_3 * sqrt(config.stiffness_joint_3))) {
                K(2, 2) = config.stiffness_joint_3;
                D(2, 2) = 2.0 * config.damping_ratio_joint_3 * sqrt(config.stiffness_joint_3);
            }
            if ((K(3, 3) != config.stiffness_joint_4) || (D(3, 3) != 2.0 * config.damping_ratio_joint_4 * sqrt(config.stiffness_joint_4))) {
                K(3, 3) = config.stiffness_joint_4;
                D(3, 3) = 2.0 * config.damping_ratio_joint_4 * sqrt(config.stiffness_joint_4);
            }
            if ((K(4, 4) != config.stiffness_joint_5) || (D(4, 4) != 2.0 * config.damping_ratio_joint_5 * sqrt(config.stiffness_joint_5))) {
                K(4, 4) = config.stiffness_joint_5;
                D(4, 4) = 2.0 * config.damping_ratio_joint_5 * sqrt(config.stiffness_joint_5);
            }
            if ((K(5, 5) != config.stiffness_joint_6) || (D(5, 5) != 2.0 * config.damping_ratio_joint_6 * sqrt(config.stiffness_joint_6))) {
                K(5, 5) = config.stiffness_joint_6;
                D(5, 5) = 2.0 * config.damping_ratio_joint_6 * sqrt(config.stiffness_joint_6);
            }
            if ((K(6, 6) != config.stiffness_joint_7) || (D(6, 6) != 2.0 * config.damping_ratio_joint_7 * sqrt(config.stiffness_joint_7))) {
                K(6, 6) = config.stiffness_joint_7;
                D(6, 6) = 2.0 * config.damping_ratio_joint_7 * sqrt(config.stiffness_joint_7);
            }
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

        if (config.q_d_joint_1 != q_d(0)) {
            q_d(0) = config.q_d_joint_1;
        }
        if (config.q_d_joint_2 != q_d(1)) {
            q_d(1) = config.q_d_joint_2;
        }
        if (config.q_d_joint_3 != q_d(2)) {
            q_d(2) = config.q_d_joint_3;
        }
        if (config.q_d_joint_4 != q_d(3)) {
            q_d(3) = config.q_d_joint_4;
        }
        if (config.q_d_joint_5 != q_d(4)) {
            q_d(4) = config.q_d_joint_5;
        }
        if (config.q_d_joint_6 != q_d(5)) {
            q_d(5) = config.q_d_joint_6;
        }
        if (config.q_d_joint_7 != q_d(6)) {
            q_d(6) = config.q_d_joint_7;
        }

        previous_set_new_reference_pose = (int) config.set_new_reference_pose;
        previous_set_new_home_pose = (int) config.set_new_home_pose;
        previous_set_reference_to_home_pose = (int) config.set_reference_to_home_pose;

        ROS_INFO("\n %f %f %f %f %f %f %f",
                 K(0, 0), K(1, 1), K(2, 2), K(3, 3), K(4, 4), K(5, 5), K(6, 6));
        ROS_INFO("\n %f %f %f %f %f %f %f",
                 D(0, 0), D(1, 1), D(2, 2), D(3, 3), D(4, 4), D(5, 5), D(6, 6));
    }

    void JointImpedanceController::goalCallback() {
        ROS_INFO("Goal received");
        //TODO: first check if goal is valid and if so init trajectory parameters, then accept goal
        boost::shared_ptr<const actionlib::SimpleActionServer<iiwa_impedance_control::JointTrajectoryExecutionAction>::Goal> newGoal = action_server->acceptNewGoal();
        if (newGoal) {
            std_msgs::Float64MultiArray::ConstPtr joint_positions_goal = boost::make_shared<const std_msgs::Float64MultiArray>(newGoal->joint_positions_goal);
            std_msgs::Float64MultiArray::ConstPtr joint_velocities_goal = boost::make_shared<const std_msgs::Float64MultiArray>(newGoal->joint_velocities_goal);
            joint_position_goals << joint_positions_goal->data[0], joint_positions_goal->data[1], joint_positions_goal->data[2], joint_positions_goal->data[3], joint_positions_goal->data[4], joint_positions_goal->data[5], joint_positions_goal->data[6];
            v_refs << joint_velocities_goal->data[0], joint_velocities_goal->data[1], joint_velocities_goal->data[2], joint_velocities_goal->data[3], joint_velocities_goal->data[4], joint_velocities_goal->data[5], joint_velocities_goal->data[6];
            ROS_INFO_STREAM("Received action goal: " << joint_position_goals.transpose() << " with velocities: " << v_refs.transpose());
        } else {
            ROS_ERROR("No new goal available to accept");
        }
    }

    void JointImpedanceController::preemptCallback() {
        ROS_INFO("Preempt received");
        action_server->setPreempted();
        trajectory_start_time = ros::Time(0);
        joint_position_goals = Eigen::VectorXd::Zero(7);
    }

    bool JointImpedanceController::isGoalReached(Eigen::VectorXd distance) {
        for (unsigned int i = 0; i < n_joints; ++i) {
            if (std::abs(distance(i)) > goal_tolerances(i)) {
                return false;
            }
        }
        return true;
    }

    void JointImpedanceController::updateReferencePose(Eigen::VectorXd distance) {
        double elapsed_time = (ros::Time::now() - trajectory_start_time).toSec();
        // Provide action feedback
        action_feedback.time_progress = (elapsed_time / max_time);
        // TODO: distance progress not correct for negative goal positions
        action_feedback.distance_progress = 1 - (distance.array().abs().maxCoeff() / max_distance);
        action_server->publishFeedback(action_feedback);
        if (action_feedback.time_progress > time_limit) {
            action_result.success = false;
            action_server->setAborted(action_result);
            trajectory_start_time = ros::Time(0);
            joint_position_goals = Eigen::VectorXd::Zero(7);
            v_refs = Eigen::VectorXd::Zero(7);
            updateDynamicReconfigureConfig(q_d);
            return;
        }

        if (elapsed_time < max_time) {
            // Update reference joint positions
            q_d = start_joint_positions + (distance.array().sign() * v_refs.array() * time_factors.array() * elapsed_time).matrix();
        } else {
            // Trajectory should be finished, set reference joint positions to goal joint positions
            q_d = joint_position_goals;
        }
        publish_trajectory(q_d);

    }

    void JointImpedanceController::initializeTrajectoryParameters(Eigen::VectorXd joint_positions, Eigen::VectorXd distance) {
        //TODO: move to (async) callback instead of main loop
        ROS_INFO("Setting new trajectory parameters");
        // New trajectory, determine trajectory parameters
        trajectory_start_time = ros::Time::now();
        start_joint_positions = joint_positions;
        distances = distance;
        max_distance = distances.maxCoeff();

        total_times = distances.array().abs() / v_refs.array();
        max_time = total_times.maxCoeff();
        time_factors = total_times.array() / max_time;
    }

    RobotState JointImpedanceController::updateRobotState() {
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

    void JointImpedanceController::enforceJointLimits(double& command, unsigned int index) {
        if (command > joint_urdfs[index]->limits->effort) { // above upper limit
            command = joint_urdfs[index]->limits->effort;
        } else if (command < -joint_urdfs[index]->limits->effort) { // below lower limit
            command = -joint_urdfs[index]->limits->effort;
        }
    }

    void JointImpedanceController::setReferenceToCurrentPose() {
        RobotState robot_state = updateRobotState();
        q_d = robot_state.position;
        updateDynamicReconfigureConfig(q_d);

        ROS_INFO("Reference pose:");
        ROS_INFO("%f %f %f %f %f %f %f", q_d(0), q_d(1), q_d(2), q_d(3), q_d(4), q_d(5), q_d(6));
    }

    void JointImpedanceController::setReferenceToHome() {
        q_d = q_home;
        updateDynamicReconfigureConfig(q_d);

        ROS_INFO("Reference pose:");
        ROS_INFO("%f %f %f %f %f %f %f", q_d(0), q_d(1), q_d(2), q_d(3), q_d(4), q_d(5), q_d(6));
    }

    void JointImpedanceController::setHomeToCurrentReference() {
        q_home = q_d;
        ROS_INFO("Home pose:");
        ROS_INFO("%f %f %f %f %f %f %f", q_home(0), q_home(1), q_home(2), q_home(3), q_home(4), q_home(5), q_home(6));
    }

    void JointImpedanceController::updateDynamicReconfigureConfig(Eigen::VectorXd q_d) {
        dynamic_reconfigure::Config conf;
        dynamic_reconfigure::DoubleParameter double_param;
        dynamic_reconfigure::ReconfigureRequest srv_req;
        dynamic_reconfigure::ReconfigureResponse srv_resp;

        double_param.name = "q_d_joint_1";
        double_param.value = q_d(0);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_2";
        double_param.value = q_d(1);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_3";
        double_param.value = q_d(2);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_4";
        double_param.value = q_d(3);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_5";
        double_param.value = q_d(4);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_6";
        double_param.value = q_d(5);
        conf.doubles.push_back(double_param);
        double_param.name = "q_d_joint_7";
        double_param.value = q_d(6);
        conf.doubles.push_back(double_param);
        srv_req.config = conf;
//        ros::service::call("/JointImpedanceController/dynamic_reconfigure_server_node/set_parameters", srv_req, srv_resp);
    }

    void JointImpedanceController::publish_joint_state(RobotState robot_state) {
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

    void JointImpedanceController::publish_trajectory(Eigen::VectorXd q_d) {
        std_msgs::Float64MultiArray msg;
        std::vector<double> q_d_data(7, 0.);
        Eigen::VectorXd::Map(q_d_data.data(), q_d_data.size()) = q_d;
        msg.data = q_d_data;
        trajectory_pub.publish(msg);
    }

    void JointImpedanceController::publish_commanded_torque(Eigen::MatrixXd commanded_torque) {
        std_msgs::Float64MultiArray msg;
        std::vector<double> commanded_torque_data(7, 0.);
        Eigen::VectorXd::Map(commanded_torque_data.data(), commanded_torque_data.size()) = commanded_torque;
        msg.data = commanded_torque_data;
        commanded_torque_pub.publish(msg);
    }

}//namespace effort_controllers

PLUGINLIB_EXPORT_CLASS(joint_impedance_controller::JointImpedanceController, controller_interface::ControllerBase);