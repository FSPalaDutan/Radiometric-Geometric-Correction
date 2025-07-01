import numpy as np
from scipy.ndimage import gaussian_filter1d

class RobertsonMethod:
    def __init__(self):
        self.wtype = 'triangle'
    
    def get_weights(self, Z):
        if self.wtype == 'triangle':
            weights = np.concatenate((np.arange(1, 129), np.arange(1, 129)[::-1]), axis=0)
            return weights[Z].astype(np.float32)
        return np.ones(Z.shape, dtype=np.float32) * 128
    
    def fitE(self, Z, G, st):
        P = st.shape[0]
        Wz = self.get_weights(Z).reshape(P, -1) / 128
        Gz = G[Z].reshape(P, -1)
        upper = np.sum(Wz * Gz * st, axis=0).astype(np.float32)
        bottom = np.sum(Wz * st * st, axis=0).astype(np.float32)
        return upper / bottom
    
    def fitG(self, Z, G, E, st):
        P = st.shape[0]
        Z = Z.reshape(P, -1)
        Wz = self.get_weights(Z).reshape(P, -1) / 128
        Wz_Em_st = Wz * (E * st)
        for m in range(256):
            index = np.where(Z == m)
            upper = np.sum(Wz_Em_st[index]).astype(np.float32)
            lower = np.sum(Wz[index]).astype(np.float32)
            if lower > 0:
                G[m] = upper / lower
        G /= G[127]
        return G
    
    def solve(self, Z_bgr, initG, shutter_times, epochs=5):
        G_bgr = np.array(initG)
        st = shutter_times.reshape(-1, 1)
        for c in range(3):
            Z = np.array(Z_bgr[c])
            G = np.array(initG[c])
            for e in range(epochs):
                print(f'\r[Robertson] Channel {c}, epoch {e+1}', end='    ')
                E = self.fitE(Z, G, st)
                G = self.fitG(Z, G, E, st)
            G_bgr[c] = G
        print()
        return np.log(G_bgr).astype(np.float32)

def robertson_method(imgs, exposures):
    robertson = RobertsonMethod()
    Z_bgr = [[img.flatten() for img in channel] for channel in imgs]
    initG = [np.exp(np.arange(0, 1, 1 / 256))] * 3
    curve_robertson = robertson.solve(Z_bgr, initG, np.array(exposures))
    curve_robertson_smoothed = [gaussian_filter1d(curve_robertson[c], sigma=4) for c in range(3)]
    return curve_robertson_smoothed