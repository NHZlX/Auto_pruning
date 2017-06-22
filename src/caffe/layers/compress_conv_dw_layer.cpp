#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/compress_conv_dw_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CConvolutionDepthwiseLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    if (conv_param.kernel_size_size() == 1) {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(0);
    } else {
      kernel_h_ = conv_param.kernel_size(0);
      kernel_w_ = conv_param.kernel_size(1);
    }
  }
  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    if (conv_param.stride_size() == 1) {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(0);
    } else {
      stride_h_ = conv_param.stride(0);
      stride_w_ = conv_param.stride(1);
    }
  }
  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    if (conv_param.pad_size() == 1) {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(0);
    } else {
      pad_h_ = conv_param.pad(0);
      pad_w_ = conv_param.pad(1);
    }
  }
  if (conv_param.dilation_size() > 0) {
    if (conv_param.dilation_size() == 1) {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(0);
    } else {
      dilation_h_ = conv_param.dilation(0);
      dilation_w_ = conv_param.dilation(1);
    }
  } else {
    dilation_h_ = 1;
    dilation_w_ = 1;
  }
  vector<int> weight_shape(4);
  weight_shape[0] = bottom[0]->channels();
  weight_shape[1] = 1;
  weight_shape[2] = kernel_h_;
  weight_shape[3] = kernel_w_;
  vector<int> bias_shape;
  if (conv_param.bias_term()) {
    this->bias_term_ = 1;
    bias_shape.push_back(bottom[0]->channels());
  }else{
    this->bias_term_ = 0;
  }
  if (this->blobs_.size() == 0) {
    if (conv_param.bias_term()) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (conv_param.bias_term()) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  /************ For dynamic network surgery ***************/
  CConvolutionDepthwiseParameter cconv_dw_param = this->layer_param_.cconv_depthwise_param();
  size_t weight_num = this->blobs_.size();
  this->blobs_.resize(weight_num + 1);
    // Intialize and fill the weightmask & biasmask
  this->blobs_[weight_num].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        cconv_dw_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[weight_num].get());

  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
	
	// Intialize the hyper-parameters
  
  this->is_pruning_ = cconv_dw_param.is_pruning();
  this->upper_bound_ = cconv_dw_param.upper_bound();
  this->iter_stop_ = cconv_dw_param.iter_stop();
  this->inter_iter_ = cconv_dw_param.inter_iter();
  this->bound_weight_ = this->upper_bound_ / log(
        this->iter_stop_ / (Dtype)this->inter_iter_);
  /********************************************************/
	
}

template <typename Dtype>
void CConvolutionDepthwiseLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bottom[0]->channels());
  top_shape.push_back((bottom[0]->height() + 2 * pad_h_
        - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1);
  top_shape.push_back((bottom[0]->width() + 2 * pad_w_
        - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1);
  top[0]->Reshape(top_shape);
  vector<int> weight_buffer_shape;
  weight_buffer_shape.push_back(bottom[0]->channels());
  weight_buffer_shape.push_back(kernel_h_);
  weight_buffer_shape.push_back(kernel_w_);
  weight_buffer_shape.push_back(bottom[0]->num());
  weight_buffer_shape.push_back(top[0]->height());
  weight_buffer_shape.push_back(top[0]->width());
  weight_buffer_.Reshape(weight_buffer_shape);
  vector<int> weight_multiplier_shape;
  weight_multiplier_shape.push_back(bottom[0]->num());
  weight_multiplier_shape.push_back(top[0]->height());
  weight_multiplier_shape.push_back(top[0]->width());
  weight_multiplier_.Reshape(weight_multiplier_shape);
  caffe_set(weight_multiplier_.count(), Dtype(1),
        weight_multiplier_.mutable_cpu_data());
  if (this->layer_param_.convolution_param().bias_term()) {
    vector<int> bias_buffer_shape;
    bias_buffer_shape.push_back(bottom[0]->channels());
    bias_buffer_shape.push_back(bottom[0]->num());
    bias_buffer_shape.push_back(top[0]->height());
    bias_buffer_shape.push_back(top[0]->width());
    bias_buffer_.Reshape(bias_buffer_shape);
    vector<int> bias_multiplier_shape;
    bias_multiplier_shape.push_back(bottom[0]->num());
    bias_multiplier_shape.push_back(top[0]->height());
    bias_multiplier_shape.push_back(top[0]->width());
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
          bias_multiplier_.mutable_gpu_data());
  }
}

template <typename Dtype>
void CConvolutionDepthwiseLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void CConvolutionDepthwiseLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CConvolutionDepthwiseLayer);
#endif

INSTANTIATE_CLASS(CConvolutionDepthwiseLayer);
REGISTER_LAYER_CLASS(CConvolutionDepthwise);

}  // namespace caffe
