from glob import glob

from setuptools import find_packages, setup

package_name = "piper_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yufei Zhao",
    maintainer_email="hi25078@bristol.ac.uk",
    description="ROS 2 bridge for the AgileX Piper arm on top of piper_sdk",
    license="MIT",
    entry_points={
        "console_scripts": [
            "piper_bridge = piper_bridge.bridge_node:main",
        ],
    },
)
