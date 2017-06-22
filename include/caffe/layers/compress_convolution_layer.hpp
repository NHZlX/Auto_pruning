#ifndef CAFFE_CCONVOLUTION_LAYER_HPP_
#define CAFFE_CCONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include <string>

namespace caffe{
/**
 * @brief Convolves the input with a bank of compressed filters, 
 *  and (optionally) adds biases.
 */

template <typename Dtype>
class CConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit CConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "CConvolution"; } 
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
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  
 private:
  Blob<Dtype> weight_tmp_;
  size_t blob_num_;
  Dtype bound_weight_;
  bool is_pruning_;
  Dtype upper_bound_;
  int iter_stop_;
  int inter_iter_; 
  string name_;
  
};

}
#endif
