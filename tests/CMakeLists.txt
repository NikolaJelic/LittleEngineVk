cmake_minimum_required(VERSION 3.13)

# Hash
add_executable(test-hash hash_test.cpp)
target_link_libraries(test-hash PRIVATE levk-core levk-interface)
add_test(Hash test-hash)

# StaticAny
add_executable(test-any any_test.cpp)
target_link_libraries(test-any PRIVATE levk-core levk-interface)
add_test(StaticAny test-any)

# ECS (Registry)
add_executable(test-ecs ecs_test.cpp)
target_link_libraries(test-ecs PRIVATE levk-core levk-interface)
add_test(ECS test-ecs)
