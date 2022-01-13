# -*-coding:utf-8-*-
import cv2
import numpy as np
import os
from utils.mk_cam import *

def make_mask(cam_img, islarge=0):
    # cam_img = cv2.imread(cam_path)
    # cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    if(islarge):
        # kernal1 = np.ones((20, 20), np.uint8)
        kernal1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        cam_img = cv2.dilate(cam_img, kernal1, iterations=2)
    # retv, biimg = cv2.threshold(dilatedimg, 122, 255, cv2.THRESH_BINARY)
    retv, biimg = cv2.threshold(cam_img, 122, 255, cv2.THRESH_BINARY)
    return biimg

def add_mask(mask, raw_img):
    mask = mask / 255
    mask = cv2.resize(mask, (299, 299))
    # raw_img = cv2.imread(raw_path)
    if(raw_img.shape[0] != 299):
        raw_img = cv2.resize(raw_img, (299, 299))
    masked = np.multiply(raw_img, mask[:,:,None]).astype(np.uint8)
    return masked

def img_balance(RGBImg):
    YCrCbimg = cv2.cvtColor(RGBImg, cv2.COLOR_RGB2YCrCb)
    (y, cr, cb) = cv2.split(YCrCbimg)
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
    # yH = clahe.apply(y)
    # yH = cv2.equalizeHist(y)
    # result = cv2.merge((yH, cr, cb))
#     LABimg = cv2.cvtColor(RGBimg, cv2.COLOR_RGB2LAB)
#     (l, a, b) = cv2.split(LABimg)

    # (b, g, r) = cv2.split(RGBImg)
    cv2.imshow('y', y)
    cv2.imshow('cr', cr)
    cv2.imshow('cb', cb)
    cv2.waitKey()
    print(cr)
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)


    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
    # bH = clahe.apply(b)
    # gH = clahe.apply(g)
    # rH = clahe.apply(r)
    # result = cv2.merge((bH, gH, rH))

    # result = cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB)
#     plt.imshow(result)
#     plt.show()

    return result

if __name__ == '__main__':
    cam_maker = MkCam()
    _, _, cam_img = cam_maker.test_model('D:/DeepSolar/coup/seq14_test/1334099/1334099_2017_0.png')

    cam_img = np.array(cam_img)
    mask = make_mask(cam_img, 1)
    cv2.imshow('mask', mask)
    cv2.waitKey()
    raw_img = cv2.imread('D:/DeepSolar/coup/seq14_test/1334099/1334099_2017_0.png')
    masked = add_mask(mask, raw_img)
    cv2.imshow('img', masked)
    cv2.waitKey()

    