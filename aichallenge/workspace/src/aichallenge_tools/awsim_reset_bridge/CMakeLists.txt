cmake_minimum_required(VERSION 3.8)
project(awsim_reset_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install entry point script for the node
install(PROGRAMS
  awsim_reset_bridge/awsim_reset_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, config and checkpoint files to share
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
