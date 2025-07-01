import numpy as np
import math
import cv2
LAMDBA = 50

def response_curve_Debevec(imgs, exposures):
    Z = [cv2.resize(img, (25, 25)).flatten() for img in imgs]
    B = [math.log(e) for e in exposures]
    w = [i if i <= 255/2 else 255-i for i in range(256)]
    
    n = 256
    A = np.zeros((np.size(Z, 1) * np.size(Z, 0) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros((np.size(A, 0), 1), dtype=np.float32)
    
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            zij = Z[j][i]
            wij = w[zij]
            A[k][zij] = wij
            A[k][n+i] = -wij
            b[k] = wij * B[j]
            k += 1
    
    A[k][127] = 1
    k += 1
    
    for i in range(n-1):
        A[k][i] = LAMDBA * w[i+1]
        A[k][i+1] = -2 * LAMDBA * w[i+1]
        A[k][i+2] = LAMDBA * w[i+1]
        k += 1
    
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256]
    lnE = x[256:]
    
    return g, lnE