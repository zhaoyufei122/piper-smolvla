"""MoveIt 2 for Piper: robot_state_publisher + piper_bridge + move_group + RViz.

  ros2 launch piper_moveit_config piper_moveit.launch.py                  # simulated arm
  ros2 launch piper_moveit_config piper_moveit.launch.py use_sim:=false   # real arm on can0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("piper", package_name="piper_moveit_config")
        .robot_description(file_path="config/piper.urdf.xacro")
        .robot_description_semantic(file_path="config/piper.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .to_moveit_configs()
    )
    rviz_config = str(moveit_config.package_path / "config" / "moveit.rviz")

    return LaunchDescription([
        # MoveIt/RViz parse doubles with the process locale; comma-decimal locales break them.
        SetEnvironmentVariable("LC_NUMERIC", "en_US.UTF-8"),
        DeclareLaunchArgument("use_sim", default_value="true",
                              description="true: simulated arm, false: real arm over CAN"),
        DeclareLaunchArgument("can_port", default_value="can0"),
        DeclareLaunchArgument("auto_enable", default_value="true",
                              description="enable the motors when the bridge starts"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="log",
            parameters=[moveit_config.robot_description],
        ),
        Node(
            package="piper_bridge",
            executable="piper_bridge",
            output="screen",
            parameters=[{
                "use_sim": LaunchConfiguration("use_sim"),
                "can_port": LaunchConfiguration("can_port"),
                "auto_enable": LaunchConfiguration("auto_enable"),
                "finger_sign": -1.0,  # Pika finger joints open negative; 1.0 for the standard gripper
            }],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[moveit_config.to_dict(), {"publish_robot_description_semantic": True}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            arguments=["-d", rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
            ],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
