cmake_minimum_required(VERSION 3.2)

project(SpEst)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fopenmp -O2")

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc	
)

add_library(spest SHARED ${SRCS})

