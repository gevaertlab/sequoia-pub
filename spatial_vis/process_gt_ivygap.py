import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

if __name__=='__main__':
    slide = 'W5-1-1-L.1.04'
    gene = 'GOLPH3'
    src_path = '/oak/stanford/groups/ogevaert/data/Gen-Pred/'
    gt_path = '/visualizations/IvyGap/ground_truth/'

    img = plt.imread(f'{src_path}/{gt_path}/{slide}_{gene}.jpg')
    img_bw = np.sum(img,axis=2)
    img_bw = ((img_bw>0)*255)
    plt.imsave(f'{src_path}/{gt_path}/{slide}_{gene}_bw.png',img_bw)

    result = gaussian_filter(img_bw, sigma=2)
    plt.imsave(f'{src_path}/{gt_path}/{slide}_{gene}_filt.png',result)
    #import pdb; pdb.set_trace()