#ifndef CAFFE_CCONV_DW_LAYER_HPP_
#define CAFFE_CCONV_DW_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
namespace caffe {

template <typename Dtype>
//class CConvolutionDepthwiseLayer : public BaseConvolutionLayer<Dtype> {
class CConvolutionDepthwiseLayer : public Layer<Dtype> {
 public:
  explicit CConvolutionDepthwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "CConvolutionDepthwise"; }

  static bool sortPairAscend(const std::pair<Dtype, size_t>& pair1,
				const std::pair<Dtype, size_t>& pair2 ){
	return pair1.first < pair2.first;
  }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  unsigned int kernel_h_;
  unsigned int kernel_w_;
  unsigned int stride_h_;
  unsigned int stride_w_;
  unsigned int pad_h_;
  unsigned int pad_w_;
  unsigned int dilation_h_;
  unsigned int dilation_w_;
  bool bias_term_;
  Blob<Dtype> weight_buffer_;
  Blob<Dtype> weight_multiplier_;
  Blob<Dtype> bias_buffer_;
  Blob<Dtype> bias_multiplier_;
 private:
  Blob<Dtype> weight_tmp_;

  Dtype bound_weight_;
  bool is_pruning_;
  Dtype upper_bound_;
  int iter_stop_;
  int inter_iter_;

};

}  // namespace caffe

#endif  // CAFFE_CONV_DW_LAYER_HPP_
