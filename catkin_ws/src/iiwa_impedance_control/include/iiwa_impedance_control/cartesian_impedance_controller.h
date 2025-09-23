//
// Created by Nicky Mol on 10/08/2022.
//
#ifndef IIWA_IMPEDANCE_CONTROL_CARTESIAN_IMPEDANCE_CONTROLLER_H
#define IIWA_IMPEDANCE_CONTROL_CARTESIAN_IMPEDANCE_CONTROLLER_H

// ROS headers
#include <ros/node_handle.h>

// ROS control
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <pluginlib/class_list_macros.h>

//Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <iiwa_impedance_control/CartesianImpedanceControllerConfig.h>

// Actionlib
#include <actionlib/server/simple_action_server.h>
#include <iiwa_impedance_control/CartesianTrajectoryExecutionAction.h>

// msgs
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/JointState.h>

// URDF
#include <urdf/model.h>

//Eigen
#include <Eigen/Eigenvalues>
#include <eigen_conversions/eigen_kdl.h>

// KDL
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <vector>

#include <random>


namespace cartesian_impedance_controller {
    struct RobotState {
        Eigen::VectorXd position,
                velocity,
                effort;
    };

    class CartesianImpedanceController : public controller_interface::Controller<hardware_interface::EffortJointInterface> {
    public:
        CartesianImpedanceController();
        ~CartesianImpedanceController();
        bool init(hardware_interface::EffortJointInterface* robot_hw, ros::NodeHandle& node_handle);

        void starting(const ros::Time& time);

        void update(const ros::Time& time, const ros::Duration& period);

//        void stopping(const ros::Time& time);

        void refPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
        bool isGoalReached(double translation_distance, double rotational_distance);
        void updateReferencePose(double ee_translation_distance, double ee_rotational_distance);
        void initializeTrajectoryParameters(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation, double ee_translation_distance, double ee_rotational_distance);
        void goalCallback();
        void preemptCallback();

    private:
        ros::NodeHandle nh;
        std::vector<hardware_interface::JointHandle> jointsHandles;
        unsigned int n_joints;
        std::vector<std::string> joint_names;

        Eigen::MatrixXd K;
        Eigen::MatrixXd D;
        Eigen::MatrixXd K_nullspace;
        Eigen::MatrixXd D_nullspace;
        Eigen::MatrixXd K_ext;
        Eigen::Matrix<double, 7, 1> q_nullspace;
        double translational_damping_ratio;
        double rotational_damping_ratio;
        double nullspace_damping_ratio_joint_1;
        double nullspace_damping_ratio_joint_2;
        double nullspace_damping_ratio_joint_3;
        double nullspace_damping_ratio_joint_4;
        double nullspace_damping_ratio_joint_5;
        double nullspace_damping_ratio_joint_6;
        double nullspace_damping_ratio_joint_7;
        ros::Publisher joint_state_pub;
        ros::Publisher cartesian_pose_pub;
        ros::Publisher trajectory_pub;
        ros::Publisher commanded_torque_pub;
        ros::Publisher cartesian_wrench_pub;
        ros::Subscriber reference_pose_sub;

        KDL::Chain iiwa_chain;
        KDL::Chain eef_chain;
        Eigen::Vector3d ee_position_home;
        Eigen::Quaterniond ee_orientation_home;
        Eigen::Vector3d ee_position_ref;
        Eigen::Quaterniond ee_orientation_ref;
        Eigen::Vector3d ee_position_goal = Eigen::Vector3d::Zero();
        Eigen::Quaterniond ee_orientation_goal = Eigen::Quaterniond::Identity();
        Eigen::Vector3d ee_position_goal_enabled;
        Eigen::Quaterniond ee_orientation_goal_enabled;
        Eigen::Vector3d ee_position_start = Eigen::Vector3d::Zero();
        Eigen::Quaterniond ee_orientation_start = Eigen::Quaterniond::Identity();
        Eigen::Vector3d ee_position_start_enabled;
        Eigen::Quaterniond ee_orientation_start_enabled;
        Eigen::Vector3d ee_offset_ee;
        unsigned int previous_set_new_reference_pose;
        unsigned int previous_set_new_home_pose;
        unsigned int previous_set_reference_to_home_pose;
        double z_pose;
        double z_margin_latch;
        double z_margin_unlatch;
        bool latched;
        unsigned int previous_assistance;

        ros::Time trajectory_start_time;
        Eigen::Vector3d start_position;
        Eigen::Vector3d translation_vector;
        Eigen::Quaterniond start_orientation;
        Eigen::Quaterniond end_orientation;

        double time_limit;
        double goal_translation_tolerance;
        double v_ref_translation;
        double total_time_translation;
        double total_distance_translation;
        double time_factor_translation;

        double goal_rotation_tolerance;
        double v_ref_rotation;
        double total_time_rotation;
        double total_distance_rotation;
        double time_factor_rotation;

        double total_time;

        // URDF
        std::vector<urdf::JointConstSharedPtr> joint_urdfs;

        RobotState updateRobotState();

        // Enforce effort limits
        void enforceJointLimits(double& command, unsigned int index);

        void setReferenceToCurrentPose();
        void setHomeToCurrentReference();
        void setReferenceToHome();

        KDL::Jacobian getJacobian(Eigen::VectorXd q, KDL::Chain chain);
        Eigen::Affine3d getForwardKinematics(Eigen::VectorXd q, KDL::Chain chain);

        // Publish messages
        void publish_joint_state(RobotState robot_state);
        void publish_cartesian_pose(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation);
        void publish_trajectory(Eigen::Vector3d ee_position, Eigen::Quaterniond ee_orientation);
        void publish_commanded_torque(Eigen::MatrixXd commanded_torque);
        void publish_cartesian_wrench(Eigen::VectorXd cartesian_wrench);

        // Dynamic reconfigure
        std::unique_ptr<dynamic_reconfigure::Server<iiwa_impedance_control::CartesianImpedanceControllerConfig>> dynamic_reconfigure_server;
        ros::NodeHandle dynamic_reconfigure_server_node;
        void dynamicReconfigureCallback(iiwa_impedance_control::CartesianImpedanceControllerConfig& config, uint32_t level);

        // Action
        std::unique_ptr<actionlib::SimpleActionServer<iiwa_impedance_control::CartesianTrajectoryExecutionAction>> action_server;
        iiwa_impedance_control::CartesianTrajectoryExecutionFeedback action_feedback;
        iiwa_impedance_control::CartesianTrajectoryExecutionResult action_result;

    };
} // namespace cartesian_impedance_controller

#endif