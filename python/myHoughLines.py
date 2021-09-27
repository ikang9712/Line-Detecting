import numpy as np

def clearArea(img, row, col):
    output_img = np.copy(img)
    threshold = 7
    for r in range(0, img.shape[0]):
    # delete row
        if abs(row - r) < threshold: 
            for c in range(0, img.shape[1]):
                # delete col
                if abs(col - c) < threshold: output_img[r,c] = 0
    return output_img

def myHoughLines(H, nLines):
    #print(np.shape(H)) # (401, 180) row: roh
    rhos = []
    thetas = [] 
    HCopy = np.copy(H)

    #find lines
    for i in range(0, nLines):
        location = np.unravel_index(np.argmax(HCopy, axis=None), HCopy.shape)
        rhos.append(location[0])
        thetas.append(location[1])
        HCopy = clearArea(HCopy, location[0], location[1])
    return [rhos, thetas]
