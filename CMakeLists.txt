cmake_minimum_required(VERSION 3.0)

project(lenet5) #1
set(OpenCV_DIR "D:\\soft\\Opencv\\opencv\\build")  #2
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}\\include) #3
set(OpenCV_LIB_DIRS ${OpenCV_DIR}\\x64\\vc15\\lib) #4
set(OpenCV_Debug_LIBS "opencv_world340d.lib") #5
set(OpenCV_Release_LIBS "opencv_world340.lib") #6
set(Libtorch_Debug_DIR "D:\\Git_Project\\libtorch_train\\libtorch") #7
set(Libtorch_Debug_INCLUDE_DIRS ${Libtorch_Debug_DIR}\\include) #8
set(Libtorch_Debug_LIB_DIRs ${Libtorch_Debug_DIR}\\lib) #9

#setup CUDA  #10
find_package(CUDA REQUIRED)
message(STATUS "      libraries: ${CUDA_LIBRARIES}")
message(STATUS "      include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) # add this line, then no need to setup cuda path in vs

#setup OpenCV #11 
include_directories(${OpenCV_INCLUDE_DIRS}) 
link_directories(${OpenCV_LIB_DIRS}) 

#setup libtorch #12
include_directories(${Libtorch_Debug_INCLUDE_DIRS})
include_directories(${Libtorch_Debug_INCLUDE_DIRS}\\torch\\csrc\\api\\include)
link_directories(${Libtorch_Debug_LIB_DIRs})


add_executable(lenet5 ${PROJECT_SOURCE_DIR}/main.cpp) #13

#setup additional dependencies #14
target_link_libraries(lenet5 ${CUDA_LIBRARIES}) #15
target_link_libraries(lenet5 debug ${OpenCV_Debug_LIBS}) #16
target_link_libraries(lenet5 optimized ${OpenCV_Release_LIBS}) #17
target_link_libraries(lenet5 "caffe2_nvrtc.lib" "c10.lib" "c10_cuda.lib") #18
target_link_libraries(lenet5 "torch.lib" "torch_cuda.lib" "torch_cpu.lib") #19