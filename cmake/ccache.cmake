https://github.com/cyrusbehr/YOLOv9-TensorRT-CPP/blob/main/cmake/ccache.cmake

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "ccache: found")
else()
    message(STATUS "ccache: not found")
endif()