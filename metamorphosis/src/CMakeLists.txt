cmake_minimum_required(VERSION 2.6)

project(metamorphosis)

find_package(ITK)

if(ITK_FOUND)
    include(${ITK_USE_FILE})
else(ITK_FOUND)
    message(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
endif(ITK_FOUND)

add_executable(metamorphosis metamorphosis.cxx itkMetamorphosisImageRegistrationMethodv4.h itkMetamorphosisImageRegistrationMethodv4.hxx itkSumImageFilter.h itkSumImageFilter.hxx itkDebug.h)
target_link_libraries(metamorphosis ${ITK_LIBRARIES})