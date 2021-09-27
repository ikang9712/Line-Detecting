import numpy as np

def findClosestIndex(array, key):
    array = np.asarray(array)
    idx = (np.abs(array - key)).argmin()
    return idx


def myHoughTransform(Im, rhoRes, thetaRes):
    ImHeight = int(np.shape(Im)[0]) 
    ImWidth = int(np.shape(Im)[1]) 
    M = np.ceil(np.sqrt(ImHeight**2 + ImWidth**2) + 1)
    rhoScale = np.arange(0,M,rhoRes)
    thetaScale = np.arange(0,np.pi*2,thetaRes)

    # we save votes here
    img_hough = np.zeros((np.shape(rhoScale)[0], np.shape(thetaScale)[0]), dtype=np.int32)
    for row in range(0,ImHeight):
        for col in range(0, ImWidth):
            if Im[row,col] > 0:
                for theta in thetaScale:
                    p = col * np.cos(theta) + row * np.sin(theta) 
                    if p >= 0:
                        rho_i = findClosestIndex(rhoScale, p)
                        theta_j = findClosestIndex(thetaScale, theta)
                        img_hough[rho_i, theta_j] +=1
    return [img_hough, rhoScale, thetaScale]


