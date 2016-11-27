#ifndef MSCNN_HPP
#define MSCNN_HPP

// Author: xc
// Date: 11/26/16

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "gpu_nms.hpp"

using namespace caffe;  
using namespace std;
using namespace cv;

#define max_cnt 150

#define model_wid 2560  // wid and hei accptable by mscnn
#define model_hei 768

#define bbox_ind 20  // indent of bbox values
#define cls_ind 5 // indent of class values
#define prop_ind 6 // indent of proposal values

const float bbox_stds [] = {0.1, 0.1, 0.2, 0.2};
const int cls_ids [] = {1};  // 0-bg, 1-car, 2-van, 3-truck, 4-tram
const int num_cls = 1;  // size of cls_ids

class mscnn {
 public:
  mscnn(const string& model_file, const string& trained_file,
        const float proposal_thr_ = -10, 
        const float nms_thr_ = 0.3, 
        const float show_thr_ = 0.2);
  vector<float> Predict(const Mat& img);

 private:
  void WrapInputLayer(vector<Mat>* input_channels);
  void Preprocess(const Mat& img, vector<Mat>* input_channels);
  void Filter(const float*, const float*, const float*);
  void GenerateBBox(const int);
  void Display(const Mat &, const float *, const int *, int);

 private:
  shared_ptr<Net<float> > net_;
  float proposal_thr;
  float nms_thr;
  float show_thr;

  Size origin_geometry_;
  Size input_geometry_;
  int num_channels_;
  int keep_num_;
  float ratios_ [2];
  
  float bbox_preds [max_cnt][bbox_ind];
  float cls_pred [max_cnt][cls_ind];
  float proposal_pred [max_cnt][prop_ind];

  float bbset [max_cnt * 5];
  int ret_index [max_cnt];
  int ret_num;
};

#endif