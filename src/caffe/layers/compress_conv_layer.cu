#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/compress_convolution_layer.hpp"
#include <cmath>
#include <fstream>

using namespace std;
namespace caffe {

// The constant NUM_THREADS should be equal to the value in CCMomentCalc
template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];    
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  

  Dtype* weightTmp = this->weight_tmp_.mutable_gpu_data(); 

  const Dtype* bias = NULL;
  if (this->bias_term_) {  
    bias = this->blobs_[1]->mutable_gpu_data();   
  }
  
  if (this->phase_ == TRAIN){
		// Calculate the mean and standard deviation of learnable parameters 		
    if (this->iter_ % this->inter_iter_ == 0 && (this->iter_) < (this->iter_stop_) && this->is_pruning_){      
  		Dtype *weightMaskCPU = this->blobs_[this->blob_num_]->mutable_cpu_data();
  		Dtype *weightCPU = this->blobs_[0]->mutable_cpu_data();
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

		LOG(INFO) << sparsity_ratio << " " << param_temp[0].first<< " " << param_temp[this->blobs_[0]->count() - 1].first;
	/* record the mask into file
	std::ofstream outfile;
	outfile.open(this->name_.c_str(), std::ofstream::app);
	for (size_t i = 0; i < this->blobs_[this->blob_num_]->count(); i++)
		outfile << weightMaskCPU[i] << " ";
	outfile << "\n";
        outfile.close(); */

	}
	
		
  }   

  const Dtype* weight = this->blobs_[0]->mutable_gpu_data();  
  Dtype* weightMask = this->blobs_[this->blob_num_]->mutable_gpu_data();
 
  // Calculate the current (masked) weight and bias
  CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
      
	// Forward calculation with (masked) weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.gpu_data();  	
  const Dtype* weightMask = this->blobs_[this->blob_num_]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();  	

  for (int i = 0; i < top.size(); ++i) {    
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();			
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weightTmp,
              bottom_diff + bottom[i]->offset(n));
        }
      }
      CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[this->blob_num_]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[this->blob_num_]->count(), weight_diff, weightMask, weight_diff);
      CUDA_POST_KERNEL_CHECK; 			
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CConvolutionLayer);

}  // namespace caffe
