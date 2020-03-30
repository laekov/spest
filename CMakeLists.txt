cmake_minimum_required(VERSION 3.2)

project(SpEst)

include_directories("include")

file(GLOB_RECURSE SRCS
	src/*.cc	
)

add_library(spest SHARED ${SRCS})
