cmake_minimum_required(VERSION 3.12)
project(calibration LANGUAGES CXX)

# Buscar OpenCV con todos los componentes necesarios
set(OpenCV_DIR "/usr/local/share/OpenCV")  # Opcional: Solo si no se detecta automáticamente
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc features2d highgui imgcodecs)

# Configuración del estándar de C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Definir macro para código legado: CV_GRAY2BGR se define igual que cv::COLOR_GRAY2BGR (valor 8)
add_compile_definitions(CV_GRAY2BGR=8)

# Buscar otras dependencias
find_package(Ceres 2.1 REQUIRED CONFIG)
find_package(Boost 1.74 REQUIRED COMPONENTS program_options)
find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(Glog REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 4 no encontrado")
else()
    message(STATUS "OpenCV encontrado: ${OpenCV_VERSION}")
    message(STATUS "Componentes: ${OpenCV_LIBS}")
endif()

# Agregar directorio de cabeceras globalmente
include_directories(${CMAKE_SOURCE_DIR}/include)

# Crear alias para los targets de OpenCV
add_library(OpenCV::core ALIAS opencv_core)
add_library(OpenCV::imgproc ALIAS opencv_imgproc)
add_library(OpenCV::features2d ALIAS opencv_features2d)
add_library(OpenCV::highgui ALIAS opencv_highgui)
add_library(OpenCV::imgcodecs ALIAS opencv_imgcodecs)

# Opciones de compilación
add_definitions(-D_USE_MATH_DEFINES)

# -------------------------------------------------------------------------------
# Targets de librerías
# -------------------------------------------------------------------------------

# Render: ahora se enlaza PUBLIC con highgui, imgcodecs y glog para propagar esas dependencias
add_library(render STATIC
    src/render/render.cpp
    src/render/plane.cpp
    src/render/background.cpp
    src/render/texture.cpp
    src/render/aa_filter_lt.cpp
)
target_link_libraries(render
    PUBLIC
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
    glog::glog
)

add_library(calibration STATIC
    src/calibration/corner_detector.cpp
    src/calibration/unified_calibration.cpp
    src/calibration/calib_cost_functions.cpp
    src/calibration/trajectory_generation.cpp
    src/calibration/odometry_cost_function.cpp
)
target_link_libraries(calibration
    PUBLIC
    Eigen3::Eigen
    OpenCV::core
    OpenCV::imgproc
    OpenCV::features2d
    OpenCV::highgui
    OpenCV::imgcodecs
    Ceres::ceres
    Boost::program_options
)

add_library(reconstruction STATIC
    src/reconstruction/eucm_sgm.cpp
    src/reconstruction/eucm_stereo.cpp
    src/reconstruction/eucm_motion_stereo.cpp
    src/reconstruction/eucm_epipolar.cpp
    src/reconstruction/depth_map.cpp
    src/reconstruction/triangulator.cpp
    src/reconstruction/scale_parameters.cpp
    src/reconstruction/epipoles.cpp
)
target_link_libraries(reconstruction
    PRIVATE
    OpenCV::core
    OpenCV::imgproc
)

add_library(localization STATIC
    src/localization/photometric.cpp
    src/localization/local_cost_functions.cpp
    src/localization/cost_function_mi.cpp
    src/localization/mono_odom.cpp
    src/localization/sparse_odom.cpp
    src/localization/mapping.cpp
)
target_link_libraries(localization
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::features2d    # Agregado para usar BFMatcher, BRISK, etc.
    Ceres::ceres
)
# -------------------------------------------------------------------------------
# Targets de ejecutables
# -------------------------------------------------------------------------------

add_executable(calib test/calibration/generic_calibration.cpp)
target_link_libraries(calib
    PRIVATE
    calibration
    Ceres::ceres
    Boost::program_options
)

add_executable(rectify test/calibration/rectify.cpp)
target_link_libraries(rectify
    PRIVATE
    calibration
    Ceres::ceres
    Boost::program_options
)

add_executable(rectify_ucm_mei test/calibration/rectify_ucm_mei.cpp)
target_link_libraries(rectify_ucm_mei
    PRIVATE
    calibration
    Ceres::ceres
    Boost::program_options
)

add_executable(rectify_no_inter test/calibration/rectify_no_inter.cpp)
target_link_libraries(rectify_no_inter
    PRIVATE
    calibration
    Ceres::ceres
    Boost::program_options
)

add_executable(odom_interpolation test/calibration/odometry_interpolation.cpp)
target_link_libraries(odom_interpolation
    PRIVATE
    Boost::program_options
)

add_executable(stereo_sync test/calibration/stereo_sync.cpp)
target_link_libraries(stereo_sync
    PRIVATE
    Boost::program_options
)

add_executable(optim_trajectory test/calibration/trajectory.cpp)
target_link_libraries(optim_trajectory
    PRIVATE
    calibration
    Ceres::ceres
)

add_executable(board_detector test/calibration/board_detector.cpp)
target_link_libraries(board_detector
    PRIVATE
    calibration
    Ceres::ceres
    Boost::program_options
)

# -------------------------------------------------------------------------------
# Tests de Reconstrucción
# -------------------------------------------------------------------------------

add_executable(hough_test test/reconstruction/hough_test.cpp)
target_link_libraries(hough_test
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(dense_ba_test test/reconstruction/dense_ba_test.cpp)
target_link_libraries(dense_ba_test
    PRIVATE
    reconstruction
    render
    localization
    OpenCV::core
)

add_executable(equalizer_test test/reconstruction/equalizer_test.cpp)
target_link_libraries(equalizer_test
    PRIVATE
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(renderer_test test/reconstruction/renderer_test.cpp)
target_link_libraries(renderer_test
    PRIVATE
    render
    reconstruction
    OpenCV::core
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(epipolar test/reconstruction/epipolar_trace.cpp)
target_link_libraries(epipolar
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)


add_executable(descriptor_step test/reconstruction/descriptor_step.cpp)
target_link_libraries(descriptor_step
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(stereo_test test/reconstruction/stereo_test.cpp)
target_link_libraries(stereo_test
    PRIVATE
    reconstruction
    render
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(stereo_single_pair test/reconstruction/stereo_single_pair.cpp)
target_link_libraries(stereo_single_pair
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)

add_executable(stereo_flow test/reconstruction/stereo_flow_test.cpp)
target_link_libraries(stereo_flow
    PRIVATE
    reconstruction
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
)


# -------------------------------------------------------------------------------
# Tests de Localización
# -------------------------------------------------------------------------------

add_executable(mapping test/localization/map_test.cpp)
target_link_libraries(mapping
    PRIVATE
    reconstruction
    localization
    render
    OpenCV::core
    Ceres::ceres
    Boost::program_options
)

add_executable(map_real_data test/localization/map_real_data.cpp)
target_link_libraries(map_real_data
    PRIVATE
    reconstruction
    localization
    render
    OpenCV::core
    Ceres::ceres
    Boost::program_options
)

add_executable(odometry_test test/localization/mono_odom_test.cpp)
target_link_libraries(odometry_test
    PRIVATE
    reconstruction
    localization
    render
    OpenCV::core
    Ceres::ceres
    Boost::program_options
)

add_executable(photometric test/localization/photometric_test.cpp)
target_link_libraries(photometric
    PRIVATE
    localization
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
    OpenCV::features2d
)

add_executable(mi_test test/localization/mi_test.cpp)
target_link_libraries(mi_test
    PRIVATE
    localization
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
    OpenCV::features2d
)

add_executable(sparse_odom_test test/localization/sparse_odom_test.cpp)
target_link_libraries(sparse_odom_test
    PRIVATE
    localization
    OpenCV::core
    OpenCV::imgproc
    OpenCV::highgui
    OpenCV::imgcodecs
    OpenCV::features2d
)


# -------------------------------------------------------------------------------
# Opciones de compilador
# -------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")
endif()
