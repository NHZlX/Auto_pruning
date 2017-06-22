
import sys
sys.path.insert(0, "/home/xingzhaolong/caffe_project/caffe_mobile/python")

import caffe
import numpy as np
import cv2
import time


class ModelConverter(object):
    def __init__(self, caffe_model_file, caffe_pretrained_file):
	#caffe.set_mode_gpu()
        #caffe.set_device(1);
		
        self.net = caffe.Net(caffe_model_file, caffe_pretrained_file, caffe.TEST)

    def caffe_predict(self,img):
        net = self.net
	im_put = np.random.rand(50, 3, 224, 224)
        net.blobs['data'].reshape(*im_put.shape)
        #net.blobs['data'].data[...] = load_image(img)
        net.blobs['data'].data[...] = im_put
	start = time.time()
        out = net.forward()
	end = time.time()
	print end - start 

        output_prob = net.blobs['fc7'].data[0].flatten()	
	print np.argmax(output_prob)

def load_image(file, resize_size=256, crop_size=224):
    # load image
    im = cv2.imread(file)
    # resize
    h, w = im.shape[:2]
    h_new, w_new = resize_size, resize_size
    if h > w:
        h_new = resize_size * h / w
    else:
        w_new = resize_size * w / h
    im = cv2.resize(im, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
    # crop
    h, w = im.shape[:2]
    h_start = (h - crop_size) / 2
    w_start = (w - crop_size) / 2
    h_end, w_end = h_start + crop_size, w_start + crop_size
    im = im[h_start:h_end, w_start:w_end, :]
    # transpose to CHW order
    mean = np.array([103.94,116.78,123.68])
    im = im - mean
    im = im.transpose((2, 0, 1))

    im = im * 0.017
    return im


if __name__ == "__main__":
    caffe_model_file = "./mobilenet/mobilenet_deploy.prototxt_test"
    caffe_pretrained_file = "./mobilenet/mobilenet.caffemodel"

    converter = ModelConverter(
        caffe_model_file=caffe_model_file,
        caffe_pretrained_file=caffe_pretrained_file)


    converter.caffe_predict("./image/cat.jpg")
