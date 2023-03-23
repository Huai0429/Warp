import numpy as np
from matplotlib import pyplot as plt

class flow_test:
    def gen_flow(self,center, height, width):
        x0, y0 = center
        if x0 >= height or y0 >= width:
            raise AttributeError('ERROR')
        flow = np.zeros((height, width, 2), dtype=np.float32)

        print(np.arange(width))
        print(np.expand_dims(np.arange(width), 0))
        print([height, 1])
        print(np.tile(np.expand_dims(np.arange(width), 0), [height, 1]))

        grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1]) #創造內為[0 - width-1]，維度為height x 1的array 
        grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

        grid_x0 = np.tile(np.array([x0]), [height, width])#[x0] x軸複製width次 y軸複製height次
        grid_y0 = np.tile(np.array([y0]), [height, width])

        flow[:,:,0] = grid_x0 - grid_x
        flow[:,:,1] = grid_y0 - grid_y

        return flow

    def make_color_wheel(self):
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
        colorwheel[col:col+YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
        colorwheel[col:col+CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
        col += + BM

        # MR
        colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col+MR, 0] = 255

        return colorwheel

    def compute_color(self,u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = self.make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2+v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a+1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols+1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel,1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0-1] / 255
            col1 = tmp[k1-1] / 255
            col = (1-f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1-rad[idx]*(1-col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

        return img

    def visual_flow(self,flow):
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.
        UNKNOWN_FLOW_THRESH = 1e7
        SMALLFLOW = 0.0
        LARGEFLOW = 1e8

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)

        img = self.compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)




if __name__ == "__main__":
    flow = flow_test()
    # Function: gen_flow_circle
    center = [500, 500]
    Flow = flow.gen_flow(center, height=1000, width=1000)
    # flow = flow / 2 # 改变光流的值，也就是改变像素的偏移量，这个不重要
    
    img = flow.visual_flow(Flow)
    plt.imshow(img)
    plt.show()
