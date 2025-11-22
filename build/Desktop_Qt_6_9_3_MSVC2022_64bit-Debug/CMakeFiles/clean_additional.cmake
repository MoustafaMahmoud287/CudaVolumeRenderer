# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\QtCudaVis_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\QtCudaVis_autogen.dir\\ParseCache.txt"
  "QtCudaVis_autogen"
  )
endif()
