from . import tetranerf_cpp_extension as cpp
# triangulate = cpp.triangulate # xxchange 1022
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(cpp)
wrapper = cpp.TriangulationWrapper()
