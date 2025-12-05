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
                                                  [-1, 0, 1]]),

                         "Emboss"     : np.array([[-2, -1, 0],
                                                  [-1,  0, 1],
                                                  [ 0,  1, 2]]),

                         "HighPass"   : np.array([[-1,-1, -1],
                                                  [-1, 8, -1],
                                                  [-1,-1, -1]])
                        }
    

    def processImage(self, x, operationData = None):

        if operationData == "EdgeDetection":

            gray = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.144 * x[:,:,2]
            gx = convolve2d(gray, self.kernels["hEdgeDetect"], mode='same', boundary='symm')
            gy = convolve2d(gray, self.kernels["vEdgeDetect"], mode='same', boundary='symm')
            mag = np.sqrt(gx**2 + gy**2)
            print(mag.shape)
            edge_rgb = np.stack([mag, mag, mag], axis=-1)
            return np.clip(edge_rgb, 0, 255).astype(np.uint8)

        elif isinstance(operationData, tuple):
            if operationData[0] == "colorTransform":
                print(x.dtype)
                init = np.array(ast.literal_eval(operationData[1])).astype(np.uint8)
                change = np.array(ast.literal_eval(operationData[2])).astype(np.uint8)
                mask = (x[:,:,0] == init[0]) & (x[:,:,1] == init[1]) & (x[:,:,2] == init[2])
                x[mask] = change
                return x

            elif operationData[0] == "add":
                return np.add(x, int(operationData[1]))

        elif operationData in self.kernels:
            output = np.zeros_like(x)
            height, width, channels = x.shape
            K                       = self.kernels[operationData]

            for c in range(channels):
                output[:,:,c] = convolve2d(x[:,:,c], K, mode='same', boundary='symm')
            
            return np.clip(output, 0, 255).astype(x.dtype)
        
        else:
            print("Not valid or implemented kernel")
            return
    

