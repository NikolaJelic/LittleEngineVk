project(vulkan-hpp)

# Vulkan
if(NOT ${PLATFORM} STREQUAL ANDROID)
find_package(Vulkan)
if(NOT Vulkan_FOUND)
	message(FATAL_ERROR 
"Vulkan SDK not found! LittleEngineVk requires Vulkan ICD loader, headers, and validation layers, and SPIR-V cross!
Note: for custom SDK locations, define `VULKAN_SDK` and source `vk-setup-env.sh` into this shell (and runtime environment).
Make sure to delete CMakeCache.txt before re-configuring!"
	)
endif()
endif()

unzip_archive(vulkan_hpp.zip "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})
add_library(${PROJECT_NAME} INTERFACE)
target_sources(${PROJECT_NAME} INTERFACE ${SOURCES})
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME} INTERFACE ${Vulkan_LIBRARIES})

target_compile_definitions(${PROJECT_NAME} INTERFACE)

set(SPIRV_CROSS_CLI OFF)
set(SPIRV_CROSS_ENABLE_CPP OFF)
set(SPIRV_CROSS_ENABLE_C_API OFF)
set(SPIRV_CROSS_ENABLE_GLSL ON)
set(SPIRV_CROSS_ENABLE_HLSL OFF)
set(SPIRV_CROSS_ENABLE_MSL OFF)
set(SPIRV_CROSS_ENABLE_REFLECT OFF)
set(SPIRV_CROSS_ENABLE_TESTS OFF)
set(SPIRV_CROSS_ENABLE_UTIL OFF)
set(SPIRV_CROSS_SHARED OFF)
set(SPIRV_CROSS_STATIC ON)
FetchContent_Declare(
	spirv-cross
	GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross
	GIT_TAG 2020-09-17
)
message(STATUS "== Fetching SPIR-V Cross")
FetchContent_MakeAvailable(spirv-cross)
if(LINUX_CLANG OR LINUX_GCC)
	target_compile_options(spirv-cross-glsl PRIVATE -Wno-sign-conversion)
endif()
set(SPIRV-Cross_SOURCE "${spirv-cross_SOURCE_DIR}")
set(Vulkan_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "" FORCE)
message(STATUS "== Vulkan ==\n\tHeaders : ${Vulkan_INCLUDE_DIRS}\n\tLibraries : ${Vulkan_LIBRARIES}\n\tSPIR-V Cross : ${SPIRV-Cross_SOURCE}")
