# C++ API for MSCNN

## Requirements
- CUDA and supporting GPU
- Caffe
- OpenCV 3.x
- C++ compilier
- Cmake

## Setup
- Copy `libgpu_nms.so` to `/usr/lib/` (need sudo)
- Add the following line to your `CMakeLiskts.txt`:
```
target_link_libraries(${name} gpu_nms)
```

## Run the sample
Suppose you *make* the target with name "mscnn". Run the following command:
```
./mscnn test_img.jpg
```

## Call the function from other code
Please refer to `mscnn.hpp` and the main function in `mscnn.cpp`:
- **mscnn::mscnn**: Use this function to initilize the model.
- **mscnn::Predict**: Use this function to detect a loaded image.