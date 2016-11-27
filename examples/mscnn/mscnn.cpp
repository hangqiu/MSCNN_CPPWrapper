#include "mscnn.hpp"

// Author: xc
// Date: 11/26/16
// Note: 
//  - all images will be resized to 2560x768
//  - only support colored images (3 channels)
// TODO: 
//  - add more classes (van, truck, tram)
//  - visualize results

/* init func */
mscnn::mscnn(const string& model_file, const string& trained_file, 
          const float proposal_thr_, const float nms_thr_, const float show_thr_) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  proposal_thr = proposal_thr_;
  nms_thr = nms_thr_;
  show_thr = show_thr_;

  cout << "proposal_thr: " << proposal_thr;
  cout << ",  nms_thr: " << nms_thr; 
  cout << ",  show_thr: " << show_thr << endl;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  cout << "net loaded" << endl;

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 3) << "Network should have exactly three outputs.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = Size(input_layer->width(), input_layer->height());
}

/* Predict and filter the results */
vector<float> mscnn::Predict(const Mat& img) {
  // modify the image
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  vector<Mat> input_channels;
  WrapInputLayer(&input_channels);
  
  cout << "Preprocessing the image" << endl;
  Preprocess(img, &input_channels);

  // forward image to the net
  cout << "Forwarding the image to net" << endl;
  net_->ForwardPrefilled();

  // filter the results and get bbox
  Filter(net_->output_blobs()[0]->cpu_data(), 
        net_->output_blobs()[1]->cpu_data(),
        net_->output_blobs()[2]->cpu_data());
  // cout << "keep num: " << keep_num_ << endl;

  for (int cls_pos = 0; cls_pos < num_cls; cls_pos++) {
    cout << "Filtering class " << cls_ids[cls_pos] << " ...." << endl;
    GenerateBBox(cls_ids[cls_pos]);
    _nms(ret_index, &ret_num, bbset, keep_num_, 5, nms_thr, 0);
  }
  
  // return the result 
  vector<float> ret;
  // cout << "nms output: " << ret_num << endl;
  for (int i = 0; i < ret_num; i++) {
    if (bbset[5 * ret_index[i] + 4] < show_thr)
      continue;
    ret.push_back(1); // class is 1 (car)...
    for (int j = 0; j < 5; j++)
      ret.push_back(bbset[5 * ret_index[i] + j]);
  } 
  return ret;
}

void mscnn::WrapInputLayer(vector<Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void mscnn::Preprocess(const Mat& img,
                            vector<Mat>* input_channels) {
  origin_geometry_ = Size(img.cols, img.rows);

  Mat sample_resized;
  resize(img, sample_resized, input_geometry_);
  ratios_[0] = (float)input_geometry_.height / (float)img.rows;
  ratios_[1] = (float)input_geometry_.width / (float)img.cols;
  
  cout << "Original size: " << img.cols << " x " << img.rows;
  cout << ". Model size: " << input_geometry_.width << " x " << input_geometry_.height;
  cout << ". Ratios: [" << ratios_[0] << ", " << ratios_[1] << "]" << endl;

  Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  Mat sample_normalized = sample_float;
  for (int i = 0; i < sample_normalized.size().width; i++)
	  for (int j = 0; j < sample_normalized.size().height; j++) {
		  sample_normalized.at<Vec3f>(j, i).val[0] -= 104.0;
	  	  sample_normalized.at<Vec3f>(j, i).val[1] -= 117.0;
		  sample_normalized.at<Vec3f>(j, i).val[2] -= 123.0;
	  }

  cv::split(sample_normalized, *input_channels);
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

/* filter the outputs */
void mscnn::Filter(const float * bbox_raw, const float * cls_raw, 
                    const float * proposal_raw) {
  vector<int> keep_id;
  for (int i = 0; i < max_cnt * prop_ind; i += prop_ind)  // filter with thresh
    if (proposal_raw[i + 5] >= proposal_thr && 
        proposal_raw[i + 2] != proposal_raw[i + 4] &&
        proposal_raw[i + 1] != proposal_raw[i + 3])
      keep_id.push_back(i / prop_ind);

  keep_num_ = keep_id.size();
  for (int i = 0; i < keep_num_; i++) {  // load the data
    int j = 0;
    for (j = 0; j < bbox_ind; j++)
      bbox_preds[i][j] = bbox_raw[i * bbox_ind + j];
    for (j = 0; j < cls_ind; j++)
      cls_pred[i][j] = cls_raw[i * cls_ind + j];
    for (j = 0; j < prop_ind; j++)
      proposal_pred[i][j] = proposal_raw[i * prop_ind + j];
    
    proposal_pred[i][3] -= proposal_pred[i][1];
    proposal_pred[i][4] -= proposal_pred[i][2];
  }
}

void mscnn::GenerateBBox(const int cls) {
  float bbox_pred[keep_num_][4];
  for (int i = 0; i < keep_num_; i++)
    for (int j = 0; j < 4; j++) {
      bbox_pred[i][j] = bbox_preds[i][cls * 4 + j];
      bbox_pred[i][j] *= bbox_stds[j];
    }

  float prob [keep_num_];
  float tx [keep_num_];
  float ty [keep_num_];
  float tw [keep_num_];
  float th [keep_num_];

  // cout << "Prob: " << endl;
  for (int i = 0; i < keep_num_; i++) {
	float sum_exp_score = 0;
	for (int j = 0; j < cls_ind; j++)
		sum_exp_score += exp(cls_pred[i][j]);
    prob[i] = exp(cls_pred[i][cls]) / sum_exp_score;
	  // cout << prob[i] << " ";
    float ctr_x = proposal_pred[i][1] + 0.5 * proposal_pred[i][3];
    float ctr_y = proposal_pred[i][2] + 0.5 * proposal_pred[i][4];
    
    tx[i] = bbox_pred[i][0] * proposal_pred[i][3] + ctr_x;
    ty[i] = bbox_pred[i][1] * proposal_pred[i][4] + ctr_y;
    tw[i] = proposal_pred[i][3] * exp(bbox_pred[i][2]);
    th[i] = proposal_pred[i][4] * exp(bbox_pred[i][3]);
    
    tx[i] -= tw[i] / 2.0;
    ty[i] -= th[i] / 2.0;

    tx[i] /= ratios_[1];
    tw[i] /= ratios_[1];
    ty[i] /= ratios_[0];
    th[i] /= ratios_[0];

    tx[i] = (tx[i] > 0) ? tx[i] : 0; // clip bbox to image boundaries
    ty[i] = (ty[i] > 0) ? ty[i] : 0;
    float dw = (float)origin_geometry_.width - tx[i];
    float dh = (float)origin_geometry_.height - ty[i];
    tw[i] = (tw[i] < dw) ? tw[i] : dw;
    th[i] = (th[i] < dh) ? th[i] : dh;

    bbset[i * 5] = tx[i];
    bbset[i * 5 + 1] = ty[i];
    bbset[i * 5 + 2] = tx[i] + tw[i];
    bbset[i * 5 + 3] = ty[i] + th[i];
    bbset[i * 5 + 4] = prob[i];
  }
  // cout << endl;
}

void mscnn::Display(const Mat & img, const float * bbset, const int * ret_index, int ret_num) {
  int final_num = 0;
  for (int i = 0; i < ret_num; i++) {
  	if (bbset[5 * ret_index[i] + 4] < show_thr)
		  continue;
    final_num++;
  }
  cout << "final num: " << final_num << endl;
}


/* --- Main func --- */
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  // load the model
  // NOTE: you must use absolute path here..
  string model_file = "/home/xiaocheng/Documents/MSCNN_CPPWrapper/examples/kitti_car/mscnn-8s-768-trainval-pretrained/mscnn_deploy.prototxt";
  string trained_file = "/home/xiaocheng/Documents/MSCNN_CPPWrapper/examples/kitti_car/mscnn-8s-768-trainval-pretrained/mscnn_kitti_trainval_2nd_iter_35000.caffemodel";
  mscnn mscnn(model_file, trained_file);

  // load image
  string file = argv[1];
  Mat img = imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;

  // do detection
  vector<float> ret = mscnn.Predict(img);
  int bbox_num = ret.size() / 6;

  // show the results
  cout << "------- results --------" << endl;
  cout << bbox_num << " objects detected" << endl;
  cout << "class\t\tx_top_left\ty_top_left\t";
  cout << "x_btm_right\ty_btm_right\tconf_score\n";
  for (int i = 0; i < bbox_num; i++) {
    for (int j = 0; j < 6; j++)
      cout << ret[i * 6 + j] << "\t\t";
    cout << endl;
  }

  return 0;
}

