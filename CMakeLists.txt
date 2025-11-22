cmake_minimum_required(VERSION 3.18)
project(QtCudaVis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 1. Find Packages
find_package(Qt6 REQUIRED COMPONENTS Quick Gui Qml OpenGLWidgets)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 2. Create Executable
# Notice we now use "src/filename.cpp"
qt_add_executable(QtCudaVis
    src/main.cpp
    src/VolumeItem.cpp
    src/VolumeRenderer.cu
    src/VolumeTexture.cu
    
    # Headers are technically optional here but good for IDE visibility
    include/VolumeItem.h
    include/VolumeLoader.h
    include/VolumeRenderer.cuh
    include/VolumeTexture.cuh
    include/CudaUtils.cuh
    include/Ray.cuh
)

# 3. search on include folder
target_include_directories(QtCudaVis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 4. Register QML
qt_add_qml_module(QtCudaVis
    URI "MyVolRender"
    VERSION 1.0
    QML_FILES main.qml
)

# 5. Link Libraries
target_link_libraries(QtCudaVis PRIVATE
    Qt6::Quick
    Qt6::Gui
    Qt6::Qml
    Qt6::OpenGLWidgets
    OpenGL::GL
    CUDA::cudart
)

# 6. GPU Architecture
set_property(TARGET QtCudaVis PROPERTY CUDA_ARCHITECTURES 75)

# 7. Auto-Copy Data File
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/maleHead.raw 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})