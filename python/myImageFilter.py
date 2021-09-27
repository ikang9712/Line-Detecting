import numpy as np

def myImageFilter(img0, h):
    # we are guaranteed to have odd numbered rows and cols for h
    hRows = np.shape(h)[0]
    hCols = np.shape(h)[1]
    hRowPadSize = int((hRows - 1) / 2)
    hColPadSize = int((hCols - 1) / 2)
    numOfPoints = hRows * hCols

    # set padding value to 0
    paddedImg = np.pad(img0, ((hRowPadSize,hRowPadSize),(hColPadSize,hColPadSize)), 'edge')

    # this will be our output image
    filteredImg = np.empty(np.shape(img0))
    filteredImg[:] = np.NaN
    # we want to center the kernel to a non-padded part of the image
    for row in range(hRowPadSize, len(paddedImg)-hRowPadSize): 
        for col in range(hColPadSize, len(paddedImg[0])-hColPadSize):
            imgPoint = paddedImg[row-hRowPadSize:row+hRowPadSize+1, col-hColPadSize:col+hColPadSize+1]
            output = np.sum(np.multiply(imgPoint, h))/numOfPoints
            filteredImg[row-hRowPadSize, col-hColPadSize] = output
    
    # check that we do not have any NaN value
    for row in filteredImg:
        for elem in row:
            if np.isnan(elem):
                raise ValueError("we have nan value after filterting the image.")
    return filteredImg



