import ast 
import numpy as np
from scipy.signal import convolve2d

class Processor:
    def __init__(self):
        self.kernels = { "gblur"  : (1/16) * np.array([[1, 2, 1],
                                                     [2, 4, 2],
                                                    [1, 2, 1]]),
                         "sharpen": np.array([[ 0, -1,  0],
                                              [-1,  5, -1],
                                              [ 0, -1,  0]]),
                         "boxblur": (1/9) * np.array([[1, 1, 1],
                                                      [1, 1, 1],
                                                      [1, 1, 1]]),
                        
                         "hEdgeDetect": np.array([[-1, -1, -1],
                                                  [ 0,  0,  0],
                                                  [ 1,  1,  1]]),

                         "vEdgeDetect": np.array([[-1, 0, 1],
                                                  [-1, 0, 1],
                                                  [-1, 0, 1]])
                        }
    
    def processImage(self, x, kernel = None):

        if kernel == "EdgeDetection":

            gray = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.144 * x[:,:,2]
            gx = convolve2d(gray, self.kernels["hEdgeDetect"], mode='same', boundary='symm')
            gy = convolve2d(gray, self.kernels["vEdgeDetect"], mode='same', boundary='symm')
            mag = np.sqrt(gx**2 + gy**2)
            print(mag.shape)
            edge_rgb = np.stack([mag, mag, mag], axis=-1)
            return np.clip(edge_rgb, 0, 255).astype(np.uint8)

        elif isinstance(kernel, tuple):
            if kernel[0] == "colorTransform":
                init = ast.literal_eval(kernel[1])
                change = ast.literal_eval(kernel[2])
                mask = (x[:,:,0] == init[0]) & (x[:,:,1] == init[1]) & (x[:,:,2] == init[2])
                x[mask] = ast.literal_eval(change)
                return x

        elif kernel in self.kernels:
            output = np.zeros_like(x)
            height, width, channels = x.shape
            K                       = self.kernels[kernel]

            for c in range(channels):
                output[:,:,c] = convolve2d(x[:,:,c], K, mode='same', boundary='symm')
            
            return np.clip(output, 0, 255).astype(x.dtype)
        
        else:
            print("Not valid or implemented kernel")
            return
    

