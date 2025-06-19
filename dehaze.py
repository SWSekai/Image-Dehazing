import cv2 as cv
import numpy as np
import math

from psnrssim import *

def multi_scale_retinex(img, scales, weights, mode= 'MSR'):
    """
        多尺度 Retinex 演算法 - 未正規化
    """
    img_log = np.log1p(img.astype(np.float32)) 

    msr = np.zeros_like(img, dtype=np.float32)

    for i, scale in enumerate(scales):
        ksize = int(scale)
        if ksize % 2 == 0:
            ksize += 1
        if ksize <= 0:
            ksize = 1

        img_blur = cv.GaussianBlur(img_log, (ksize, ksize), 0)

        retinex = img_log - np.log1p(img_blur)

        msr += weights[i] * retinex
    
    if mode == 'MSR':
        msr = cv.normalize(msr, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    else:
        pass # MSRCR 模式不進行正規化，返回未正規化

    return msr

def multi_scale_retinex_with_color_restoration(img, scales, weights, alpha=125, beta=46, G=192):
    """
        多尺度 Retinex 演算法 (MSRCR)
        hyper parameter: alpha, beta, G
    """
    img_float = img.astype(np.float32)

    # 執行 MSR (獲取未正規化的結果)
    msr_unnormalized = multi_scale_retinex(img_float, scales, weights, mode= 'MSRCR')

    # 顏色恢復
    img_max_channel = np.max(img_float, axis=2) # 每個像素的 R, G, B 最大值
    img_max_channel[img_max_channel == 0] = 1 # 避免 log(0)

    color_restoration_factor = beta * np.log1p(alpha * img_max_channel / 255.0) # 將像素值範圍調整到 0-1
    color_restoration_factor = np.expand_dims(color_restoration_factor, axis=2)

    # 應用顏色恢復和增益
    msrcr_result = G * (msr_unnormalized * color_restoration_factor)

    # 3. 正規化到 [0, 255]
    msrcr_result = cv.normalize(msrcr_result, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return msrcr_result

def get_dark_channel_size(img_shape, rate= 0.01):
    min_dim = min(img_shape[0], img_shape[1])
    size = int(min_dim * rate)
    if size % 2 == 0: # 確保為奇數
        size += 1
    if size < 3: # 最小不小於3
        size = 3
        
    return size

def DarkChannel(img, size):
    """
        計算影像的 Dark Channel
    """
    b, g, r = cv.split(img)
    dc = cv.min(cv.min(r, g), b) 
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    darkC = cv.erode(dc, kernel)
    
    return darkC

def AtmLight(img, dark):
    """
        計算大氣光照明強度
    """
    [h, w] = img.shape[:2]
    imgsize = h * w
    numpx = int(max(math.floor(imgsize / 1000), 1))
    darkvec = dark.reshape(imgsize)
    imvec = img.reshape(imgsize, 3)

    indices = darkvec.argsort()
    indices = indices[imgsize - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    
    return A

def AtmLight_Quadtree(img, dark_channel, min_area_ratio=0.01, top_percentage=0.001, mode= 'median'):
    """
        使用四叉樹分解估計大氣光照明強度。
        min_area_ratio: 最小分解區域佔總圖像面積的比例。
        top_percentage: 在選定區域中取Dark Channel最亮的前N%像素。
    """
    h, w = img.shape[:2]

    def _quadtree_decompose(current_img_region, current_dark_region, rect):
        x, y, cw, ch = rect
        
        # 計算當前區域的 Dark Channel 平均值和標準差
        dc_mean = np.mean(current_dark_region)
        dc_std = np.std(current_dark_region)

        # 判斷是否停止分解：
        # 1. 區域太小 (防止無限遞歸，且小區域可能不具代表性)
        # 2. 區域內的 Dark Channel 均勻性達到閾值 (表示該區域可能是大氣光區域)
        # 3. 區域位於圖像頂部 (天空區域通常在圖像頂部)

        if (cw * ch <= h * w * min_area_ratio) or \
           (dc_std < 0.15 and cw * ch > h * w * 0.05) or \
           (y + ch // 2 < h // 4 and dc_mean > 0.8):

            # 估計大氣光
            region_size = cw * ch
            num_pixels_to_consider = int(max(math.floor(region_size * top_percentage), 1))
            
            dark_vec_region = current_dark_region.reshape(-1)
            img_vec_region = current_img_region.reshape(-1, 3)

            indices = dark_vec_region.argsort()
            # 選擇 Dark Channel 最亮的像素
            indices = indices[len(dark_vec_region) - num_pixels_to_consider::]

            atmsum = np.zeros([1, 3])
            for idx in range(num_pixels_to_consider): # 從0開始迭代
                atmsum += img_vec_region[indices[idx]]

            return [atmsum / num_pixels_to_consider]
        
        else:
            # 分解成四個子區域
            results = []
            hw = cw // 2
            hh = ch // 2
            
            # 左上
            results.extend(_quadtree_decompose(current_img_region[0:hh, 0:hw], 
                                               current_dark_region[0:hh, 0:hw], 
                                               (x, y, hw, hh)))
            # 右上
            results.extend(_quadtree_decompose(current_img_region[0:hh, hw:cw], 
                                               current_dark_region[0:hh, hw:cw], 
                                               (x + hw, y, cw - hw, hh)))
            # 左下
            results.extend(_quadtree_decompose(current_img_region[hh:ch, 0:hw], 
                                               current_dark_region[hh:ch, 0:hw], 
                                               (x, y + hh, hw, ch - hh)))
            # 右下
            results.extend(_quadtree_decompose(current_img_region[hh:ch, hw:cw], 
                                               current_dark_region[hh:ch, hw:cw], 
                                               (x + hw, y + hh, cw - hw, ch - hh)))
            return results

    atm_candidates = _quadtree_decompose(img, dark_channel, (0, 0, w, h))
    
    # 融合所有候選大氣光值 (例如取平均或中位數)
    if not atm_candidates:
        print("Warning: No suitable regions found for quadtree atmospheric light estimation. Falling back to original method.")
        return AtmLight(img, dark_channel) # 回退到原來的AtmLight函數

    if mode == 'median':
        A = np.median(atm_candidates, axis=0) # 中位數
    elif mode == 'mean':
        A = np.mean(atm_candidates, axis=0) # 平均值
    
    return A.reshape(1, 3)

def TransmissionEstimate(img, A, size):
    """
        計算 transmission map(透射率圖)
    """
    omega = 0.95
    img3 = np.empty(img.shape, img.dtype)

    for ind in range(0, 3):
        img3[:, :, ind] = img[:, :, ind] / A[0, ind] # 將影像除以大氣光照明強度 

    transmission = 1 - omega * DarkChannel(img3, size)
    
    return transmission

def get_guided_filter_params(img_shape, transmission_map):
    min_dim = min(img_shape[0], img_shape[1])
    ksize = int(min_dim * 0.05)
    if ksize % 2 == 0: ksize += 1
    if ksize < 5: ksize = 5 # 最小 ksize

    eps = np.mean(transmission_map**2) * 0.001
    if eps < 1e-5:
        eps = 1e-5
    
    return ksize, eps

def Guidedfilter(img, p, ksize, eps):
    """
        應用 guided filter 進行細化
    """
    mean_I = cv.boxFilter(img, cv.CV_64F, (ksize, ksize))
    mean_p = cv.boxFilter(p, cv.CV_64F, (ksize, ksize))
    mean_Ip = cv.boxFilter(img * p, cv.CV_64F, (ksize, ksize))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv.boxFilter(img * img, cv.CV_64F, (ksize, ksize))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv.boxFilter(a, cv.CV_64F, (ksize, ksize))
    mean_b = cv.boxFilter(b, cv.CV_64F, (ksize, ksize))

    q = mean_a * img + mean_b
    
    return q

def TransmissionRefine(img, teMap, k_size, eps):
    """
        細化 transmission map(透射率圖)
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    transmissionMap = Guidedfilter(gray, teMap, k_size, eps)

    return transmissionMap


def Recover(img, tMap, A, t0=0.1):
    """
        重建除霧後的影像
    """
    res = np.empty(img.shape, img.dtype)
    tMap = cv.max(tMap, t0)

    for ind in range(0, 3):
        res[:, :, ind] = (img[:, :, ind] - A[0, ind]) / tMap + A[0, ind]

    return res

def white_balance(img):
    """
        白平衡
    """

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    img_balanced = img.copy()
    img_balanced[:, :, 0] = np.clip(img[:, :, 0] * avg_gray / avg_b, 0, 255)
    img_balanced[:, :, 1] = np.clip(img[:, :, 1] * avg_gray / avg_g, 0, 255)
    img_balanced[:, :, 2] = np.clip(img[:, :, 2] * avg_gray / avg_r, 0, 255)

    return img_balanced.astype(np.uint8) # 將影像轉換回 uint8 格式

def color_correction_lab(img):
    """
        使用 Lab 色彩空間進行顏色校正
    """
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)

    # 計算平均值
    l_mean = np.mean(l)
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    # 計算增益
    l_gain = 100 / l_mean
    a_gain = 128 / a_mean
    b_gain = 128 / b_mean

    # 調整顏色
    l = np.clip(l * l_gain, 0, 255).astype(np.uint8)
    a = np.clip(a * a_gain, 0, 255).astype(np.uint8)
    b = np.clip(b * b_gain, 0, 255).astype(np.uint8)

    lab_corrected = cv.merge((l, a, b))
    
    return cv.cvtColor(lab_corrected, cv.COLOR_Lab2BGR)

def increase_brightness_hsv(img, factor=1.2):
    """
        使用 HSV 模式調整亮度
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv.merge((h, s, v))

    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def apply_clahe(img):
    """
        對影像亮度通道使用 CLAHE 增亮
    """
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

if __name__ == '__main__':
    input_image_path = './input_image/'
    output_image_path = './output_image/'
    tmp_path = './tmp/'
    
    ratinex_mode = 'MSRCR' # 選擇處理模式，'MSR' 或 'MSRCR'
    AtmLight_mode = 'default' # 選擇大氣光照明估計模式，'quadtree' 或 'default'
    
    for i in range(1, 10):
        fn = 'hazy0'+ str(i) + '.jpg'
        src = cv.imread(input_image_path + fn)
        
        if src is None: # 檢查圖片是否成功載入
            print(f"無法載入圖片: {input_image_path + fn}, 跳過此圖。")
            continue
        
        if ratinex_mode == 'MSR':
            # 設定 MSR 參數
            scales = [15, 80, 250]  # 高斯濾波器大小
            weights = [1/3, 1/3, 1/3]  # 尺度權重
        
            dehazed_img_msr = multi_scale_retinex(src.astype(np.float32), scales, weights)
        elif ratinex_mode == 'MSRCR':
            # MSRCR 參數
            scales = [5, 100, 400]  # 高斯濾波器大小
            weights = [0.1, 0.25, 0.65]  # 尺度權重, 小、中、大
            msrcr_alpha = 200 # MSRCR 顏色恢復參數
            msrcr_beta = 60  # MSRCR 顏色恢復參數
            msrcr_G = 220   # MSRCR 亮度增強

            dehazed_img_msr = multi_scale_retinex_with_color_restoration(src, scales, weights, 
                                                                        alpha= msrcr_alpha, 
                                                                        beta= msrcr_beta, 
                                                                        G= msrcr_G)
        else:
            print("無效的模式選擇，請選擇 'MSR' 或 'MSRCR'")
            continue
        
        I = dehazed_img_msr.astype('float64') / 255
        
        size = get_dark_channel_size(I.shape, rate=0.01)
        dark = DarkChannel(I, size)
        
        if AtmLight_mode == 'quadtree':
            # 使用四叉樹分解估計大氣光照明強度
            min_area_ratio = 0.01  # 最小分解區域佔總圖像面積的比例
            top_percentage = 0.001 # 區域中 Dark Channel 最亮的前 N% 像素
            
            A = AtmLight_Quadtree(I, dark, min_area_ratio=min_area_ratio, top_percentage=top_percentage, mode='median')
        elif AtmLight_mode == 'default':
            # 使用傳統方法估計大氣光照明強度
            A = AtmLight(I, dark)
        else:
            print("無效的大氣光照明估計模式，請選擇 'quadtree' 或 'default'")
            continue
        
        teMap = TransmissionEstimate(I, A, size)
        
        ksize_gf, eps_gf = get_guided_filter_params(src.shape, teMap)
        RefineMap = TransmissionRefine(src, teMap, ksize_gf, eps_gf)
        DeHazeImg = Recover(I, RefineMap, A, 0.1)
        
        DeHazeImg = (DeHazeImg * 255).astype('uint8')
        DeHazeImg = white_balance(DeHazeImg)
        # DeHazeImg = color_correction_lab(DeHazeImg)
        # DeHazeImg = increase_brightness_hsv(DeHazeImg, 1.2)
        # DeHazeImg = apply_clahe(DeHazeImg)

        cv.imwrite(tmp_path + 'drak'+ fn, dark)
        cv.imwrite(tmp_path + 'teMap' + fn, teMap)
        cv.imwrite(tmp_path + 'reMap' + fn, RefineMap)
        cv.imwrite(output_image_path + fn, DeHazeImg)

        print("第" + str(i) + "張影像處理完成")
        
        cv.waitKey()
    
    compute_metrics('./gt/', output_image_path)