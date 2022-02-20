import cv2
import numpy as np


def get_max_win(img, center_x, center_y, win_size):
    left = int(max(0, center_y - win_size/2))
    top = int(max(0, center_x - win_size/2))
    right = int(min(img.shape[1]-1, center_y + win_size/2))
    down = int(min(img.shape[0]-1, center_x + win_size/2))
    max_val = 0
    for y in range(left, right+1):
        for x in range(top, down+1):
            max_val = max(max_val, img[x][y])
    return max_val


def get_min_win(img, center_x, center_y, win_size):
    left = int(max(0, center_y - win_size/2))
    top = int(max(0, center_x - win_size/2))
    right = int(min(img.shape[1]-1, center_y + win_size/2))
    down = int(min(img.shape[0]-1, center_x + win_size/2))
    min_val = 256
    for y in range(left, right+1):
        for x in range(top, down+1):
            min_val = min(min_val, img[x][y])
    return min_val


def local_contrast(intensity):
    alpha, win_size = 0.5, 5
    dx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1)
    amplitude = ((dx ** 2) + (dy ** 2)) ** 0.5
    lc = intensity.copy()
    for x in range(lc.shape[0]):
        for y in range(lc.shape[1]):
            max_intensity = get_max_win(intensity, x, y, win_size)
            min_intensity = get_min_win(intensity, x, y, win_size)
            max_amplitude = get_max_win(amplitude, x, y, win_size)
            lc[x][y] = alpha * (max_intensity - min_intensity) + (1 - alpha) * max_amplitude
    return lc


def get_fusion_map(lc_y, lc_nir):
    fusion_map = np.maximum(0, lc_nir - lc_y) / lc_nir
    return fusion_map


def hpf(img, kernel_size=19):
    gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img, gaussian = np.float32(img), np.float32(gaussian)
    hpf = img - gaussian
    return hpf


if __name__ == '__main__':
    img = cv2.imread('./data/fig_4_b_rgb.tiff')
    nir = np.float32(cv2.imread('./data/fig_4_b_nir.tiff')[:, :, 0])
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img, yuv = np.float32(img), np.float32(yuv)
    luminance = yuv[:, :, 0]

    lc_y = local_contrast(luminance)
    lc_nir = local_contrast(nir)
    fusion_map = get_fusion_map(lc_y, lc_nir)
    hpf = hpf(nir)

    result = img.copy()
    result[:, :, 0] += fusion_map * hpf
    result[:, :, 1] += fusion_map * hpf
    result[:, :, 2] += fusion_map * hpf
    result = np.minimum(255, np.maximum(0, result))
    result = np.uint8(result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.imwrite('./data/result.tiff', result)