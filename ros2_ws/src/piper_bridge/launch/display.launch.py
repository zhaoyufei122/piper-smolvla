"""Piper model in RViz, driven by piper_bridge (real arm, or use_sim:=true). No MoveIt.

Use it to confirm that the real joint angles match the URDF before running MoveIt.
Motors stay disabled unless auto_enable:=true.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description = FindPackageShare("piper_description")
    tool = LaunchConfiguration("tool")
    # The Pika gripper's finger joints run the other way, hence the sign.
    urdf_file = PythonExpression(
        ["'piper_pika_gripper.urdf' if '", tool, "' == 'pika' else 'piper_description.urdf'"])
    finger_sign = PythonExpression(["-1.0 if '", tool, "' == 'pika' else 1.0"])
    robot_description = ParameterValue(
        Command(["xacro ", PathJoinSubstitution([description, "urdf", urdf_file])]),
        value_type=str,
    )
    return LaunchDescription([
        DeclareLaunchArgument("tool", default_value="pika",
                              description="end-effector fitted: pika or piper"),
        DeclareLaunchArgument("use_sim", default_value="false", description="simulated arm instead of CAN"),
        DeclareLaunchArgument("can_port", default_value="can0"),
        DeclareLaunchArgument("auto_enable", default_value="false", description="enable motors on start"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="piper_bridge",
            executable="piper_bridge",
            output="screen",
            parameters=[{
                "use_sim": LaunchConfiguration("use_sim"),
                "can_port": LaunchConfiguration("can_port"),
                "auto_enable": LaunchConfiguration("auto_enable"),
                "finger_sign": finger_sign,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", PathJoinSubstitution([description, "rviz", "piper_ctrl.rviz"])],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
