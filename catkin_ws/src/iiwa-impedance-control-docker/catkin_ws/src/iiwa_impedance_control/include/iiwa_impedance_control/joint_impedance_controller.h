//
// Created by Nicky Mol on 10/08/2022.
//
#ifndef IIWA_IMPEDANCE_CONTROL_JOINT_IMPEDANCE_CONTROLLER_H
#define IIWA_IMPEDANCE_CONTROL_JOINT_IMPEDANCE_CONTROLLER_H

// ROS headers
#include <ros/node_handle.h>

// ROS control
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <pluginlib/class_list_macros.h>

//Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <iiwa_impedance_control/JointImpedanceControllerConfig.h>
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>

// Actionlib
#include <actionlib/server/simple_action_server.h>
#include <iiwa_impedance_control/JointTrajectoryExecutionAction.h>

// msgs
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/JointState.h>

// URDF
#include <urdf/model.h>

//Eigen
#include <Eigen/Eigenvalues>

#include <vector>


namespace joint_impedance_controller {
    struct RobotState {
        Eigen::VectorXd position,
                velocity,
                effort;
    };
    class JointImpedanceController : public controller_interface::Controller<hardware_interface::EffortJointInterface> {
        public:
            JointImpedanceController();
            ~JointImpedanceController();
            bool init(hardware_interface::EffortJointInterface* robot_hw, ros::NodeHandle& node_handle);

            void starting(const ros::Time& time);

            void update(const ros::Time& time, const ros::Duration& period);

//            void stopping(const ros::Time& time);

            bool isGoalReached(Eigen::VectorXd distance);
            void updateReferencePose(Eigen::VectorXd distance);
            void initializeTrajectoryParameters(Eigen::VectorXd joint_positions, Eigen::VectorXd distance);
            void goalCallback();
            void preemptCallback();

        private:
            ros::NodeHandle nh;
            std::vector<hardware_interface::JointHandle> jointsHandles;
            unsigned int n_joints;
            std::vector<std::string> joint_names;
            // URDF
            std::vector<urdf::JointConstSharedPtr> joint_urdfs;

            RobotState updateRobotState();

            Eigen::VectorXd q_d;
            Eigen::VectorXd q_home;
            Eigen::MatrixXd K;
            Eigen::MatrixXd D;
            double damping_ratio;

            ros::Publisher joint_state_pub;
            ros::Publisher trajectory_pub;
            ros::Publisher commanded_torque_pub;

            unsigned int previous_set_new_reference_pose;
            unsigned int previous_set_new_home_pose;
            unsigned int previous_set_reference_to_home_pose;

            Eigen::VectorXd joint_position_goals = Eigen::VectorXd::Zero(7);
            ros::Time trajectory_start_time;
            Eigen::VectorXd start_joint_positions;
            double time_limit;
            Eigen::VectorXd goal_tolerances;
            Eigen::VectorXd v_refs = Eigen::VectorXd::Zero(7);
            Eigen::VectorXd total_times;
            double max_time;
            double max_distance;
            Eigen::VectorXd distances;
            Eigen::VectorXd translation_vector;
            Eigen::VectorXd time_factors;

            // Enforce effort limits
            void enforceJointLimits(double& command, unsigned int index);

            void setReferenceToCurrentPose();
            void setHomeToCurrentReference();
            void setReferenceToHome();

            // Publish messages
            void publish_joint_state(RobotState robot_state);
            void publish_trajectory(Eigen::VectorXd q_d);
            void publish_commanded_torque(Eigen::MatrixXd commanded_torque);

            // Dynamic reconfigure
            std::unique_ptr<dynamic_reconfigure::Server<iiwa_impedance_control::JointImpedanceControllerConfig>> dynamic_reconfigure_server;
            ros::NodeHandle dynamic_reconfigure_server_node;
            void dynamicReconfigureCallback(iiwa_impedance_control::JointImpedanceControllerConfig& config, uint32_t level);
            void updateDynamicReconfigureConfig(Eigen::VectorXd q_d);

            // Action
            std::unique_ptr<actionlib::SimpleActionServer<iiwa_impedance_control::JointTrajectoryExecutionAction>> action_server;
            iiwa_impedance_control::JointTrajectoryExecutionFeedback action_feedback;
            iiwa_impedance_control::JointTrajectoryExecutionResult action_result;

    };
} // namespace joint_impedance_controller

#endif