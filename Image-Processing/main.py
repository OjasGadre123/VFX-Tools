
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

        for k in sys.argv[3:]:
            out = processor.processImage(out, kernel=k)

        outimg = Image.fromarray(out)
        outimg.show()
        return     


if __name__ == "__main__":
    main()
