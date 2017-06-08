from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

class DataPreprocessing(object):
    """
         Note: 
            Contain all function needed for preprocessing data
    """
    @staticmethod
    def generateNonLinearData(x,order):
        xnew = x
        for i in np.arange(2,order+1):
            xnew = np.concatenate([xnew,x**i],axis=1)
        return xnew
    
    @staticmethod
    def flattenImage(data):
        """
            Note: 
                Convert 2D array image into 1D array
            Args:
                data : 2D (an image) or 3D array (nSample x Image)
            Return
                ndarray as a 1D or 2D (flattened) array.
        """
        flattenData = np.array(data).flatten()
        return flattenData.reshape([len(data), int(len(flattenData)/len(data))])
    
    @staticmethod
    def train_validate_split(data,label, train_percent=.8, validate_percent=.2, seed=None):
        """
            Note: 
                split data into train and validation set.
            Args:
                data : 3D array (nSample x Image)
                label : 1D array
                train_percent : weight of training set
                train_percent : weight of validation set
                seed: pseudo random number
            Return
                A list contains 4 ndarray for training and validation set
        """
        # init
        np.random.seed(seed)
        m = len(data)
        idx = np.arange(m)
        perm = np.random.permutation(idx)
        train_end = int(train_percent * m)

        # filter data
        train_data = data[perm[:train_end]]
        validate_data = data[perm[train_end:]]
        train_labels = label[perm[:train_end]]
        validate_labels = label[perm[train_end:]]

        return [train_data,train_labels, validate_data,validate_labels]

    @staticmethod
    def HOG(colorMode, dataset, orientation=8,pPerCell=16,cPerBlock=1):
        """
            Note: 
                Comptue Histogram Of Gradient
            Args:
                colorMode: color.(rgb2yuv,rgb,rgb2gray)
                dataset : 3 or 4 dimension array of image
                orientation: number of orientation in each cell
                pPercell: the size of the cell
            Return
                ndarray HOG for the image as a 1D (flattened) array.
        """
        # CASE: image is gray-scale
        if (len(np.shape(dataset))==3):
            return [hog(x,orientations=orientation, 
                pixels_per_cell=(pPerCell,pPerCell),cells_per_block=(cPerBlock,cPerBlock),visualise=False) for x in dataset]
        
        # CASE: change color kernel
        if (colorMode != 'rgb'):
            images = colorMode(dataset)
            
        newFeature = [hog(x,orientations=orientation, 
                          pixels_per_cell=(pPerCell,pPerCell),cells_per_block=(1,1),visualise=False) for x in images[:,:,:,0]]
        for i in range(1,3):
            newFeature = np.concatenate((newFeature,[hog(x,orientations=orientation, 
                pixels_per_cell=(pPerCell,pPerCell),cells_per_block=(1,1),visualise=False) for x in images[:,:,:,i]]),axis=1)
        return newFeature