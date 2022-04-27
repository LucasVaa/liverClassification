import numpy as np
import cv2

image_path = r"C:\Users\lucas\Documents\nk\Postgraduate\Graduate Year One\deep learning\midterm assignments" \
             r"\liverClassification\data\train\4\4_1.jpg"

# UltraSound图像增强
# 1. 自适应中值滤波-降噪
# 2. 直方图均值化-图像增强

def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(7, 7))
    dst = clahe.apply(image)
    return dst

'''   自适应中值滤波器的python实现   '''
def AdaptProcess(src, i, j, minSize, maxSize):

    filter_size = minSize

    kernelSize = filter_size // 2
    rio = src[i-kernelSize:i+kernelSize+1, j-kernelSize:j+kernelSize+1]
    minPix = np.min(rio)
    maxPix = np.max(rio)
    medPix = np.median(rio)
    zxy = src[i,j]

    if (medPix > minPix) and (medPix < maxPix):
        if (zxy > minPix) and (zxy < maxPix):
            return zxy
        else:
            return medPix
    else:
        filter_size = filter_size + 2
        if filter_size <= maxSize:
            return AdaptProcess(src, i, j, filter_size, maxSize)
        else:
            return medPix


def adapt_meadian_filter(img, minsize, maxsize):

    borderSize = maxsize // 2

    src = cv2.copyMakeBorder(img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REFLECT)

    for m in range(borderSize, src.shape[0] - borderSize):
        for n in range(borderSize, src.shape[1] - borderSize):
            src[m,n] = AdaptProcess(src, m, n, minsize, maxsize)

    dst = src[borderSize:borderSize+img.shape[0], borderSize:borderSize+img.shape[1]]
    return dst

gray_level = 256  # 灰度级


def pixel_probability(img):
    """
    计算像素值出现概率
    :param img:
    :return:
    """
    assert isinstance(img, np.ndarray)

    prob = np.zeros(shape=(256))

    for rv in img:
        for cv in rv:
            prob[cv] += 1

    r, c = img.shape
    prob = prob / (r * c)

    return prob


def probability_to_histogram(img, prob):
    """
    根据像素概率将原始图像直方图均衡化
    :param img:
    :param prob:
    :return: 直方图均衡化后的图像
    """
    prob = np.cumsum(prob)  # 累计概率

    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

   # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]

    return img






cv2.namedWindow("img", 0)
cv2.resizeWindow('img', 400, 300)
cv2.namedWindow("img0", 0)
cv2.resizeWindow('img0', 400, 300)
cv2.namedWindow("img1", 0)
cv2.resizeWindow('img1', 400, 300)
img = cv2.imread(image_path, 0)
cv2.imshow("img", img)

img0 = adapt_meadian_filter(img, 3, 7)
cv2.imshow("img0", img0)

prob = pixel_probability(img0)
img1 = probability_to_histogram(img0, prob)
cv2.imshow("img1", img1)

cv2.waitKey(0)