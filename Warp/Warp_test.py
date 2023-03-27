import numpy as np
from flow_generation import flow_gen
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2


def image_warp(im, flow, mode): # from csdn 
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    flag = 4
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im[np.newaxis, :, :, np.newaxis]
        flow = flow[np.newaxis, :, :]
        flag = 2
    elif im.ndim == 3:
        height, width, channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
        flag = 3
    elif im.ndim == 4:
        num_batch, height, width, channels = im.shape
        flag = 4
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = np.reshape(im, [-1, channels])
    flow_flat = np.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    flow_floor = np.floor(flow_flat).astype(np.int32)

    # Construct base indices which are displaced with the flow
    pos_x = np.tile(np.arange(width), [height * num_batch])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]

    x0 = pos_x + x
    y0 = pos_y + y

    x0 = np.clip(x0, zero, max_x)
    y0 = np.clip(y0, zero, max_y)

    dim1 = width * height
    batch_offsets = np.arange(num_batch) * dim1
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, dim1])
    base = np.reshape(base_grid, [-1])

    base_y0 = base + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - np.floor(flow_flat)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = np.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = np.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = np.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = np.expand_dims(xw * yw, 1) # bottom right pixel

        x1 = x0 + 1
        y1 = y0 + 1

        x1 = np.clip(x1, zero, max_x)
        y1 = np.clip(y1, zero, max_y)

        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = np.reshape(warped_flat, [num_batch, height, width, channels])

    if flag == 2:
        warped = np.squeeze(warped)
    elif flag == 3:
        warped = np.squeeze(warped, axis=0)
    else:
        pass
    warped = warped.astype(np.uint8)

    return warped

def bilinear_warp(img,flow): # for pile of tensor from M_LVC
    batch = 100
    grid_b,grid_y,grid_x= tf.meshgrid(tf.range(batch),tf.range(img.shape[1]),tf.range(img.shape[0]),indexing = 'ij')
    grid_b = tf.cast(grid_b,tf.float32)
    grid_y = tf.cast(grid_y,tf.float32)
    grid_x = tf.cast(grid_x,tf.float32)

    fx,fy = tf.unstack(flow,axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    #warping indice 
    h_lim = tf.cast(img.shape[1]-1,tf.float32)
    w_lim = tf.cast(img.shape[0]-1,tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)

    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # print(g_00)
    # print(g_01)
    # print(g_10)
    # print(g_11)

    # gather contents
    x_00 = tf.gather_nd(img, g_00)
    x_01 = tf.gather_nd(img, g_01)
    x_10 = tf.gather_nd(img, g_10)
    x_11 = tf.gather_nd(img, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

def bilinear_warp_img(img,flow): #from chatGPT
    x, y = tf.unstack(flow, axis = -1)
    # 分别计算四个相邻像素点的坐标
    x1 = tf.floor(x)
    x2 = x1 + 1
    y1 = tf.floor(y)
    y2 = y1 + 1
    
    # 分别计算四个相邻像素点的权重
    w11 = (x2 - x) * (y2 - y)
    w12 = (x2 - x) * (y - y1)
    w21 = (x - x1) * (y2 - y)
    w22 = (x - x1) * (y - y1)
    
    # 分别获取四个相邻像素点的像素值
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, w-1)
    x2 = tf.clip_by_value(tf.cast(x2, tf.int32), 0, w-1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, h-1)
    y2 = tf.clip_by_value(tf.cast(y2, tf.int32), 0, h-1)
    I11 = tf.gather_nd(img, tf.stack([y1, x1], axis=-1))
    I12 = tf.gather_nd(img, tf.stack([y2, x1], axis=-1))
    I21 = tf.gather_nd(img, tf.stack([y1, x2], axis=-1))
    I22 = tf.gather_nd(img, tf.stack([y2, x2], axis=-1))
    
    I11 = tf.cast(I11,tf.float32)
    I12 = tf.cast(I12,tf.float32)
    I21 = tf.cast(I21,tf.float32)
    I22 = tf.cast(I22,tf.float32)

    # 根据四个相邻像素点的像素值和权重，计算目标像素的像素值
    return w11*I11 + w12*I12 + w21*I21 + w22*I22

if __name__ == '__main__':
    img = cv2.imread('./img/test.png',cv2.IMREAD_GRAYSCALE)
    Flow = flow_gen()
    flow = Flow.gen_flow(center=[img.shape[0]-1,img.shape[1]-1],height=img.shape[0],width=img.shape[1])
    
    img_warped = bilinear_warp_img(img,flow)
    vis_flow = Flow.visual_flow(flow)
    print(img_warped)
    print(img.dtype)
    img_warped = img_warped.numpy()
    # vis_flow = vis_flow.numpy()
    cv2.imshow("warped image",img_warped)
    cv2.imshow("flow",vis_flow)
    cv2.waitKey(0)
    
    

    # warped = image_warp(img,flow,mode = 'nearest') # nearest bilinear
    # plt.figure(1)
    # plt.imshow(img)
    # plt.figure(2)
    # plt.imshow(Flow.visual_flow(flow))
    # plt.figure(3)
    # plt.imshow(warped)
    # plt.show()