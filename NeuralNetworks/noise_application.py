import numpy as np
import copy
from PIL import Image

class DirectNoise():
    '''
    Contains functions that apply each noise transformation on any set of input images.
    '''
    def __init__(self, data, noise_type):
        self.data = copy.deepcopy(data) #assumes data is 2d array with dimensions (# of images, 784)
        self.noise_type = noise_type

    def apply_noise(self):
        if self.noise_type == 'none':
            return self.data

        elif self.noise_type == 'vert_shrink25':
            result = np.zeros(self.data.shape)
            top_padding = np.zeros((4,28))
            bottom_padding = np.zeros((3,28))
            for i in xrange(self.data.shape[0]):
                transformed_image = np.array(Image.fromarray(np.uint8(self.data[i,:].reshape((28,28))*255)).resize((28,21), Image.ANTIALIAS))/255.0
                result[i,:] = np.vstack((top_padding, transformed_image, bottom_padding)).flatten()
            return result

        elif self.noise_type == 'horiz_shrink25':
            result = np.zeros(self.data.shape)
            left_padding = np.zeros((28,4))
            right_padding = np.zeros((28,3))
            for i in xrange(self.data.shape[0]):
                transformed_image = np.array(Image.fromarray(np.uint8(self.data[i,:].reshape((28,28))*255)).resize((21,28), Image.ANTIALIAS))/255.0
                result[i,:] = np.hstack((left_padding, transformed_image, right_padding)).flatten()
            return result

        elif self.noise_type == 'both_shrink25':
            inter_result = np.zeros(self.data.shape)
            left_padding = np.zeros((28,4))
            right_padding = np.zeros((28,3))
            for i in xrange(self.data.shape[0]):
                transformed_image = np.array(Image.fromarray(np.uint8(self.data[i,:].reshape((28,28))*255)).resize((21,28), Image.ANTIALIAS))/255.0
                inter_result[i,:] = np.hstack((left_padding, transformed_image, right_padding)).flatten()

            final_result = np.zeros(self.data.shape)
            top_padding = np.zeros((4,28))
            bottom_padding = np.zeros((3,28))
            for i in xrange(self.data.shape[0]):
                transformed_image = np.array(Image.fromarray(np.uint8(inter_result[i,:].reshape((28,28))*255)).resize((28,21), Image.ANTIALIAS))/255.0
                final_result[i,:] = np.vstack((top_padding, transformed_image, bottom_padding)).flatten()
            return final_result

        elif self.noise_type == 'light_tint':
            background = 0.2 * np.ones((28,28))
            return np.clip(self.data + background.flatten(), 0.0, 1.0)

        elif self.noise_type == 'gradient':
            background = np.zeros((28,28))
            for i in xrange(28):
                for j in xrange(28):
                    background[i,j] = ((i+j)/54.0) * 0.4
            return np.clip(self.data + background.flatten(), 0.0, 1.0)
        
        elif self.noise_type == 'checkerboard':
            background = np.zeros((28,28))
            for i in xrange(28):
                if (i % 4 == 0) or (i % 4 == 1):
                    background[i,:] = 0.4
                    background[:,i] = 0.2
            return np.clip(self.data + background.flatten(), 0.0, 1.0)

        elif self.noise_type == 'pos_noise':
            random_values = (np.random.random(self.data.shape) * 0.1) + 0.05
            return np.clip((self.data + random_values), 0.0, 1.0)

        elif self.noise_type == 'mid_noise':
            random_values = (np.random.random(self.data.shape) * 0.1) - 0.05
            return np.clip((self.data + random_values), 0.0, 1.0)

        elif self.noise_type == 'neg_noise':
            random_values = ((np.random.random(self.data.shape) * 0.1) + 0.05) * (-1.0)
            return np.clip((self.data + random_values), 0.0, 1.0)

        else:
            raise ValueError("This noise type is not currently supported.")