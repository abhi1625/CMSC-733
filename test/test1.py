import numpy as np
import scipy.stats as st
import skimage.transform
import matplotlib.pyplot
import cv2

def gkern(kernlen, nsig):
	"""Returns a 2D Gaussian kernel array."""

	interval = (2*nsig+1.)/(kernlen)
	x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel

def main():
	scales = 1
	orient = 15
	size = 9
	scale=range(1,scales+1)
	orients = np.linspace(0,360,orient)

	for each in scale:
		kernel=gkern(size,3)
		border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
		sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=5, borderType=border)
		ker = list()
		for i,eachOrient in enumerate(orients):
			matplotlib.pyplot.figure(figsize=(16,2))
			image=skimage.transform.rotate(sobelx64f,eachOrient)
			matplotlib.pyplot.subplots_adjust(hspace=0.3,wspace=0.5)
			matplotlib.pyplot.subplot(scales,orient,i+1)
			matplotlib.pyplot.imshow(image,cmap='binary')
			ker.append(image)
			image=0

	matplotlib.pyplot.show()
if __name__ == '__main__':
	main()
