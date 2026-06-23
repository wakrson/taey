# Debug build
cmake -S . -B debug_build -DCMAKE_BUILD_TYPE=Debug
cmake --build debug_build --config Debug --target tum

# Release build
cmake -S . -B release_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_CXX_FLAGS="-DEIGEN_NO_DEBUG -DEIGEN_FAST_MATH"
cmake --build release_build -j --target tum
