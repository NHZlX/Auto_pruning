#ifndef CAFFE_CINNERPRODUCT_LAYER_HPP_
#define CAFFE_CINNERPRODUCT_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>

namespace caffe {
/**
 * @brief The compressed InnerProduct layer, also known as a compressed 
 *  "fully-connected" layer
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CInnerProductLayer : public Layer<Dtype> {
 public:
  explicit CInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  static bool sortPairAscend(const std::pair<Dtype, size_t>& pair1,
				const std::pair<Dtype, size_t>& pair2 ){
	return pair1.first < pair2.first;
  }

  virtual inline const char* type() const { return "CInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

 private:
  Blob<Dtype> weight_tmp_;

  Dtype bound_weight_;
  bool is_pruning_;
  Dtype upper_bound_;
  int iter_stop_;
  int inter_iter_;
  string name_;
  int blob_num_;
};

}
#endif
