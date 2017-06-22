import numpy as np

def ana(filename):
	data = np.loadtxt(filename, dtype='int')
	h, w = data.shape
	print 'element num :', w
	head_zero =  set(np.where(data[0] == 0)[0])
	for i in xrange(1, h):
		end_one = set(np.where(data[i] == 1)[0])
		num = end_one.intersection(head_zero)
		print i , " iter  ", len(num )
		head_zero = set(np.where(data[i] == 0)[0])


if __name__ == '__main__':
	ana('ip2')
