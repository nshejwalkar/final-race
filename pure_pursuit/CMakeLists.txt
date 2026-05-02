cmake_minimum_required(VERSION 3.8)
project(pure_pursuit)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  pure_pursuit/pure_pursuit_node.py
  pure_pursuit/gap_follow.py
  pure_pursuit/drive_mux.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config waypoints
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
