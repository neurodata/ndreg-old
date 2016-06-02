cmake_minimum_required(VERSION 2.6)
project(ndreg)

set(PACKAGE_VERSION "0.0.1")

# Build metamorphosis binary
find_package(ITK)
if(NOT ITK_FOUND)
    message(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
endif()

find_program(PYTHON "python")
if(NOT PYTHON)
    message(FATAL_ERROR "Python not found.")
endif()


include(${ITK_USE_FILE})
add_subdirectory(ndreg)

# Configure setup.py in binary directory
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# Copy python source files to binary directory
file(GLOB PYTHON_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ndreg/*.py")
file(COPY ${PYTHON_SRC_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/ndreg/")

# Build ndreg egg
set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON} ${SETUP_PY} build
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
    DEPENDS ${PYTHON_SRC_FILES} metamorphosis)

add_custom_target(ndreg ALL DEPENDS ${OUTPUT})

# Install ndreg egg
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user)")




