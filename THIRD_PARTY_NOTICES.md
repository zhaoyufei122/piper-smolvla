# Third-party notices

The robot description and selected configuration/setup assets originate from [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros), `humble` branch, under the MIT license:

- `ros2_ws/src/piper_description/` (with local modifications)
- `ros2_ws/src/piper_moveit_config/config/piper.srdf`
- `ros2_ws/src/piper_moveit_config/config/pilz_cartesian_limits.yaml`
- `ros2_ws/src/piper_moveit_config/config/moveit.rviz`
- `scripts/can_activate.sh`

The upstream copyright and full license text are retained at [ros2_ws/src/piper_description/LICENSE](ros2_ws/src/piper_description/LICENSE). Local adaptations include the Pika gripper model, RViz configuration and the ready pose. Earlier adaptation notes are preserved in [docs/BRINGUP.md](docs/BRINGUP.md).

Runtime dependencies include [Piper SDK](https://github.com/agilexrobotics/piper_sdk), [LeRobot](https://github.com/huggingface/lerobot), [SmolVLA](https://huggingface.co/lerobot/smolvla_base), PyTorch and Intel RealSense libraries. Those projects and any downloaded weights retain their own licenses. No model weights or LeRobot source copy are bundled here.

This publication preserves existing file/package license notices; it does not select a new repository-wide license for otherwise unlicensed local code.
