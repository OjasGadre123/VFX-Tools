
import sys
import numpy as np
from PIL import Image
from Processing import Processor


def main():
    img = Image.open(sys.argv[1])
    img.show()
    
    imgdata   = np.array(img)
    processor = Processor()
    print(imgdata.shape)
    print(len(sys.argv))

    if len(sys.argv) == 2:
        print("No kernel type provided")
    
    elif len(sys.argv) == 3:
        out    = processor.processImage(imgdata, operationData=sys.argv[2])
        print(out.shape)
        outimg = Image.fromarray(out)
        outimg.show()
        return 

    else:
        out = np.zeros_like(imgdata)
        idx = 2 
        while idx < len(sys.argv):
            if sys.argv[idx] == "colorTransform":
                operationData = (sys.argv[idx], sys.argv[idx+1], sys.argv[idx+2],)
                
                if idx == 2:
                    out = processor.processImage(imgdata, operationData=operationData)
                else:
                    out = processor.processImage(out, operationData=operationData)
                idx = idx + 3
            
            elif sys.argv[idx] == "add":
                operationData = (sys.argv[idx], sys.argv[idx+1],)
                print(operationData) 

                if idx == 2:
                    out = processor.processImage(imgdata, operationData=operationData)
                else:
                    out = processor.processImage(out, operationData=operationData)
                idx = idx + 2

            else:
                if idx == 2:
                    out = processor.processImage(imgdata, operationData=sys.argv[idx])
                else:
                    out = processor.processImage(out, operationData=sys.argv[idx])
                idx = idx + 1

        outimg = Image.fromarray(out)
        outimg.show()
        return     


if __name__ == "__main__":
    main()
