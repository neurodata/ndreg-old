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

# Configure setup.py in binary directory
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# Copy python source files to binary directory
set(PYTHON_FILENAMES "ndreg.py;__init__.py")

foreach(FILENAME ${PYTHON_FILENAMES})
  set(PYTHON_SRC_FILES ${PYTHON_SRC_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/ndreg/${FILENAME})
  set(PYTHON_BIN_FILES ${PYTHON_BIN_FILES} ${CMAKE_CURRENT_BINARY_DIR}/ndreg/${FILENAME})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ndreg/${FILENAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/ndreg/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/ndreg/${FILENAME}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ndreg/${FILENAME})
endforeach()

# Build ndreg egg
include(${ITK_USE_FILE})
add_subdirectory(ndreg)

set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON} ${SETUP_PY} build
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
    DEPENDS ${PYTHON_BIN_FILES} metamorphosis)

add_custom_target(ndreg ALL DEPENDS ${OUTPUT})

# Install ndreg egg
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user)")





