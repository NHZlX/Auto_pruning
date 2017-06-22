#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/compress_inner_product_layer.hpp"
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

namespace caffe {

template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];    
  }
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {    

  Dtype* weightTmp = this->weight_tmp_.mutable_gpu_data();  
  const Dtype* bias = NULL;
  if (this->bias_term_) {  
    bias = this->blobs_[1]->mutable_gpu_data();   
  }   
    
  if (this->phase_ == TRAIN){
    if (this->iter_ % this->inter_iter_ == 0 && (this->iter_) < (this->iter_stop_) && this->is_pruning_){      
  		Dtype* weightCPU = this->blobs_[0]->mutable_cpu_data();
  		Dtype* weightMaskCPU = this->blobs_[this->blob_num_]->mutable_cpu_data();
        // compute the weight mask based on the inter_inter
        Dtype sparsity_ratio = this->bound_weight_ * log(2 + (this->iter_ / this->inter_iter_));
		// compute the mask
		caffe_set(this->blobs_[this->blob_num_]->count(), (Dtype)1.0, weightMaskCPU);
		vector<std::pair <Dtype, size_t> > param_temp;
		for (size_t i = 0; i < this->blobs_[this->blob_num_]->count(); i++)
			param_temp.push_back(std::make_pair(fabs(weightCPU[i]), i));

		std::sort(param_temp.begin(), param_temp.end(), sortPairAscend);
		for (size_t i = 0; i < this->blobs_[0]->count() * sparsity_ratio; i++)
			weightMaskCPU[param_temp[i].second] = 0.0;

		LOG(INFO) << sparsity_ratio << " " << param_temp[this->blobs_[0]->count()*sparsity_ratio].first \
							<< " " << param_temp[this->blobs_[0]->count() - 1].first;
	/* record mask into file
	std::ofstream outfile;
	outfile.open(this->name_.c_str(), std::ofstream::app);
	for (size_t i = 0; i < this->blobs_[this->blob_num_]->count(); i++){
		outfile << weightMaskCPU[i] << " ";
	}
	outfile << "\n";
        outfile.close();*/
	}
  }  
  
  const Dtype* weight = this->blobs_[0]->mutable_gpu_data();  
  Dtype* weightMask = this->blobs_[this->blob_num_]->mutable_gpu_data();
  // Calculate the current (masked) weight and bias
  CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
   
	// Forward calculation with (masked) weight and bias 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weightTmp, bottom_data, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            bias, top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weightTmp, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            bias, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
	const Dtype* weightMask = this->blobs_[this->blob_num_]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
    //Gradient with respect to weight
	CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[this->blob_num_]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[this->blob_num_]->count(), weight_diff, weightMask, weight_diff);
    CUDA_POST_KERNEL_CHECK; 
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,bias_diff);
  }	
  if (propagate_down[0]) {
		const Dtype* weightTmp = this->weight_tmp_.gpu_data();        
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CInnerProductLayer);

}  // namespace caffe
