
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
        out    = processor.processImage(imgdata, kernel=sys.argv[2])
        print(out.shape)
        outimg = Image.fromarray(out)
        outimg.show()
        return 

    else:
        out = processor.processImage(imgdata, kernel=sys.argv[2])
        idx = 3
        while idx < len(sys.argv):
            if sys.argv[idx] == "colorTransform":
                kernel = (sys.argv[idx], sys.argv[idx+1], sys.argv[idx+2],)
                print(kernel)
                out = processor.processImage(out, kernel=kernel)
                idx = idx + 3
            else:
                out = processor.processImage(out, kernel=sys.argv[idx])
                idx = idx + 1

        outimg = Image.fromarray(out)
        outimg.show()
        return     


if __name__ == "__main__":
    main()
