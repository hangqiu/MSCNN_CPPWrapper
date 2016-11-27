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

## IO format
- Input: The image (Type: `Mat &`)
- Output: A list of detection results (Type: `vector<float>`). The cell length is 6, which means [0~5] is for the first bounding box, [6~11] is for the second bbox... The six numbers represent: 
	* `object_type` (1 means 'car')
	* `x_upper_left` (measured by pixel, x is horizonal, y is vertical)
	* `y_upper_left`
	* `x_bottom_right`
	* `y_bottom_right`
	* `confidenc_score` (range: 0~1)

## Call the function from other code
Please refer to `mscnn.hpp` and the main function in `mscnn.cpp`:
- **mscnn::mscnn**: Use this function to initilize the model.
- **mscnn::Predict**: Use this function to detect a loaded image.
