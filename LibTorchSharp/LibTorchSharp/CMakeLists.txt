project(LibTorchSharp)

if(APPLE)
 include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
 link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

FILE(GLOB_RECURSE SOURCES *.h *.cpp)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(NOT APPLE)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")

    else()
        SET(CMAKE_MACOSX_RPATH TRUE)
    endif()
endif()

# Add libTorch bindings
include_directories(${TORCH_INCLUDE_DIRS})
#link_directories(${CUDAToolkit_LIBRARY_DIR}/stubs)

add_library(LibTorchSharp SHARED ${SOURCES} ${RESOURCES})

target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES})

#message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
#message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
#message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")

set_property(TARGET LibTorchSharp PROPERTY CXX_STANDARD 14)

if(APPLE)
    set_target_properties(LibTorchSharp PROPERTIES INSTALL_RPATH "@loader_path;@executable_path;")
endif()

#install_library_and_symbols (LibTorchSharp)
