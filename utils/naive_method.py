# -*-coding:utf-8 -*-
import cv2
import numpy as np
import os
import json
import ast
import shutil
import matplotlib.pyplot as plt
from utils.mk_mask import *
from utils.mk_cam import *
from PIL import Image


class NaiveMethod():
    def __init__(self):
        self.tree_low = np.int32([25, 43, 46])
        self.tree_up = np.int32([77, 255, 255])
        self.white_low = np.int32([0, 0, 170])
        self.white_up = np.int32([180, 10, 255])
        self.black_low = np.int32([0, 0, 0])
        self.black_up = np.int32([180, 10, 16])
        self.lightblue_low = np.int32([56, 12, 120])
        self.lightblue_up = np.int32([98, 68, 255])
        self.brown_low = np.int32([0, 0, 80])
        self.brown_up = np.int32([44, 86, 255])
        self.cam_maker = MkCam()

    def find_range(self, hist, width=15):
        hist = list(hist)
        max_index = []
        result = []
        for i in range(0, width):
            if(max(hist) <= 5):
                break
            index = hist.index(max(hist))
            max_index.append(index)
            hist[index] = 0
        for i in range(0, len(max_index)):
            result.append(max_index[i])
        result.sort()
        return result

    def extract_ref(self, img_path, img_ref, cam_maker):
        img_ref = cv2.bilateralFilter(img_ref, 20, 75, 75)
        if(img_ref.shape[0] != 700):
            img_ref = cv2.resize(img_ref, (700, 700))

        _, _, cam_img = cam_maker.test_model(img_path)
        cam_img = np.array(cam_img)
        cam_img = cv2.resize(cam_img, (299, 299))
        mask = make_mask(cam_img)
        large_mask = make_mask(cam_img, islarge=1)
        masked = add_mask(mask, img_ref)
        
        masked_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        
        img_ref = cv2.resize(img_ref, (299,299))
        ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        biimg = cv2.adaptiveThreshold(ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
        biimg = 255 - biimg

        histh = cv2.calcHist([masked_hsv], [0], None, [180], [1, 180])
        hists = cv2.calcHist([masked_hsv], [1], None, [255], [1, 255])
        histv = cv2.calcHist([masked_hsv], [2], None, [255], [1, 255])

        rangeh = self.find_range(histh, 20)
        ranges = self.find_range(hists, 40)
        rangev = self.find_range(histv, 40)

        img_hsv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)

        lowerb = np.int32([rangeh[0], ranges[0], rangev[0]])
        upperb = np.int32([rangeh[-1], ranges[-1], rangev[-1]])
        empty = cv2.inRange(img_hsv, lowerb, upperb)
        # cv2.imshow('empty0', empty)
        empty = cv2.bitwise_and(empty, biimg)
        empty = cv2.bitwise_and(empty, large_mask)

        # empty = cv2.bitwise_and(biimg, large_mask)

        tree = cv2.inRange(img_hsv, self.tree_low, self.tree_up)
        white = cv2.inRange(img_hsv, self.white_low, self.white_up)
        black = cv2.inRange(img_hsv, self.black_low, self.black_up)
        lightblue = cv2.inRange(img_hsv, self.lightblue_low, self.lightblue_up)
        brown = cv2.inRange(img_hsv, self.brown_low, self.brown_up)

        empty = cv2.subtract(empty, tree)
        empty = cv2.subtract(empty, white)
        empty = cv2.subtract(empty, black)
        empty = cv2.subtract(empty, lightblue)
        empty = cv2.subtract(empty, brown)
        result = empty
        result[0:25, 0:100] = 0
        result[289:298, 0:299] = 0
        return result, large_mask

    def extract_tar(self, img_tar, large_mask):
        # img_tar = cv2.fastNlMeansDenoisingCololightblue(img_tar, None, 10, 10, 7, 21)
        tar_hsv = cv2.cvtColor(img_tar, cv2.COLOR_BGR2HSV)
        tar_gray = cv2.cvtColor(img_tar, cv2.COLOR_BGR2GRAY)
        biimg = cv2.adaptiveThreshold(tar_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
        biimg = 255 - biimg

        tree = cv2.inRange(tar_hsv, self.tree_low, self.tree_up)
        white = cv2.inRange(tar_hsv, self.white_low, self.white_up)
        brown = cv2.inRange(tar_hsv, self.brown_low, self.brown_up)
        lightblue = cv2.inRange(tar_hsv, self.lightblue_low, self.lightblue_up)
        biimg = cv2.subtract(biimg, tree)
        biimg = cv2.subtract(biimg, white)
        biimg = cv2.subtract(biimg, brown)
        biimg = cv2.subtract(biimg, lightblue)
        
        large_mask = cv2.dilate(large_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)), iterations=1)
        
        result = cv2.bitwise_and(biimg, large_mask)
        result = cv2.erode(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
        result[0:25, 0:100] = 0
        result[289:298, 0:299] = 0
        return result

    def cal_sim(self, img_ref, img_tar):
        img_ref[0:25, 0:100] = 0
        img_ref[289:298, 0:299] = 0
        img_tar[0:25, 0:100] = 0
        img_tar[289:298, 0:299] = 0

        base = (img_ref * img_ref).sum()
        score = (img_ref * img_tar).sum()
        return score / (base + 0.00001)

    def perform_extract(self, img_path, is_ref=1, large_mask=None):
        img = cv2.imread(img_path)
        # print(img_path)
        img = cv2.resize(img, (299, 299))
        if(is_ref):
            ref_bi, large_mask = self.extract_ref(img_path, img, self.cam_maker)
            return ref_bi, large_mask
        else:
            tar_bi = self.extract_tar(img, large_mask)
            return tar_bi

    def test_single(self, ref_path='', tar_path=''):
        ref_bi, large_mask = self.perform_extract(ref_path, is_ref=1)
        tar_bi = self.perform_extract(tar_path, is_ref=0, large_mask=large_mask)
        sim = self.cal_sim(ref_bi, tar_bi)
        return sim
    
    def mk_dataset(self, img_info_dic_path, result_dic_path):
        result_dic = {}
        with open(img_info_dic_path, 'r') as f:
            img_info_dic = json.load(f)
        count = 0
        for each_key in img_info_dic:
            ref_path = each_key
            if(os.path.exists(ref_path)):
                # each_ref_dic = ast.literal_eval(img_info_dic[each_key])
                each_ref_dic = img_info_dic[each_key]
                if((len(each_ref_dic['1']) != 0) | (len(each_ref_dic['0']) != 0)):
                    ref_bi, large_mask = self.perform_extract(ref_path, is_ref=1)
                    ref_bi_dst = ref_path.replace('/couple_classify/finaltry/', '/couple_classify_naive/')
                    cv2.imwrite(ref_bi_dst, ref_bi)
                    result_dic[ref_bi_dst] = {}
                    result_dic[ref_bi_dst]['1'] = []
                    result_dic[ref_bi_dst]['0'] = []
                    for each_pos in each_ref_dic['1']:
                        if(os.path.exists(each_pos)):
                            tar_bi = self.perform_extract(each_pos, is_ref=0, large_mask=large_mask)
                            tar_bi_dst = each_pos.replace('/couple_classify/finaltry/', '/couple_classify_naive/')
                            cv2.imwrite(tar_bi_dst, tar_bi)
                            result_dic[ref_bi_dst]['1'].append(tar_bi_dst)
                    for each_neg in each_ref_dic['0']:
                        if(os.path.exists(each_neg)):
                            tar_bi = self.perform_extract(each_neg, is_ref=0, large_mask=large_mask)
                            tar_bi_dst = each_neg.replace('/couple_classify/finaltry/', '/couple_classify_naive/')
                            cv2.imwrite(tar_bi_dst, tar_bi)
                            result_dic[ref_bi_dst]['0'].append(tar_bi_dst)
            count += 1
            print(count)
        with open(result_dic_path, 'w') as f:
            json.dump(result_dic, f, indent=1)
        return result_dic

def move_dataset(ref_src, ref_dst, img_path):
    ref_src_list = os.listdir(ref_src)
    ref_dic = {}
    result_dic = {}
    for each in ref_src_list:
        ref_dic[each[0: -11]] = each
    img_list = os.listdir(img_path)
    count = 0
    for each in img_list:
        src = ref_src + '/' + ref_dic[each[0: -11]]
        dst = ref_dst + '/' + ref_dic[each[0: -11]]
        shutil.copy(src, dst)
        count += 1
        print(count)
    
def match_dataset(base_path, result_path):
    result_dic = {}
    ref_dic = {}
    path_ref = base_path + '/ref'
    path_1 = base_path + '/1'
    path_0 = base_path + '/0'
    for each in os.listdir(path_ref):
        ref_dic[each[0: -11]] = path_ref + '/' + each
        result_dic[path_ref + '/' + each] = {}
        result_dic[path_ref + '/' + each]['1'] = []
        result_dic[path_ref + '/' + each]['0'] = []
    
    count = 0
    for each in os.listdir(path_1):
        result_dic[ref_dic[each[0: -11]]]['1'].append(path_1 + '/' + each)
        count += 1
        print(count)
    
    count = 0
    for each in os.listdir(path_0):
        result_dic[ref_dic[each[0: -11]]]['0'].append(path_0 + '/' + each)
        count += 1
        print(count)

    with open(result_path, 'w') as f:
        json.dump(result_dic, f, indent=1)

def clean_dataset(dic_path):
    with open(dic_path, 'r') as f:
        m_dic = json.load(f)
    count_1 = 0
    count_0 = 0
    start_1_from = 500
    start_0_from = 0
    plt.ion()
    for each_key in m_dic:
        local_ref_path = each_key.replace('/home/ubuntu/projects/data/deepsolar2/', 'E:/MyData/DeepSolar/')
        ref = cv2.imread(local_ref_path)
        for each_1 in m_dic[each_key]['1']:
            count_1 += 1
            if(count_1 >= start_1_from):
                local_tar_1_path = each_1.replace('/home/ubuntu/projects/data/deepsolar2/', 'E:/MyData/DeepSolar/')
                tar_1 = cv2.imread(local_tar_1_path)
                cv2.imshow('pos ' + str(count_1) + ' ' + local_tar_1_path.split('/')[-1], np.hstack((ref, tar_1)))
                cmd = cv2.waitKey(0)
                cv2.destroyAllWindows()
                # plt.figure(0, dpi=200)
                # plt.subplot(1, 2, 1)
                # plt.imshow(ref)
                # plt.title('ref')
                # plt.subplot(1, 2, 2)
                # plt.imshow(tar_1)
                # plt.title('tar_1')
                # plt.suptitle(str(count_1) + '   ' + 'postive ' + local_tar_1_path.split('/')[-1])
                # cmd = plt.waitforbuttonpress()
                if(cmd == 9):
                    src = local_tar_1_path
                    dst = local_tar_1_path[:-4] + 'nouse.png'
                    os.rename(src, dst)
                    print('nouse: ' + local_tar_1_path)

    for each_key in m_dic:
        local_ref_path = each_key.replace('/home/ubuntu/projects/data/deepsolar2/', 'E:/MyData/DeepSolar/')
        ref = cv2.imread(local_ref_path)
        for each_0 in m_dic[each_key]['0']:
            count_0 += 1
            if(count_0 >= start_0_from):
                local_tar_0_path = each_0.replace('/home/ubuntu/projects/data/deepsolar2/', 'E:/MyData/DeepSolar/')
                tar_0 = cv2.imread(local_tar_0_path)
                cv2.imshow('neg ' + str(count_0) + ' ' + local_tar_0_path.split('/')[-1], np.hstack((ref, tar_0)))
                cmd = cv2.waitKey(0)
                cv2.destroyAllWindows()
                # plt.subplot(1, 2, 1)
                # plt.imshow(ref)
                # plt.title('ref')
                # plt.subplot(1, 2, 2)
                # plt.imshow(tar_0)
                # plt.title('tar_0')
                # plt.suptitle(str(count_0) + '   ' + 'negtive ' + local_tar_0_path.split('/')[-1])
                # plt.waitforbuttonpress()
                if(cmd == 9):
                    src = local_tar_0_path
                    dst = local_tar_0_path[:-4] + 'nouse.png'
                    os.rename(src, dst)
                    print('nouse: ' + local_tar_0_path)

def dic2list(dic_path, train_num=0):
    f = open(dic_path, 'r')
    m_dic = json.load(f)
    f.close()
    m_train_list_pos = []
    m_train_list_neg = []
    count = 0
    count_pos = 0
    count_neg = 0
    for each_key in m_dic:
        if(os.path.exists(each_key)):
            for each_0 in m_dic[each_key]['0']:
                if(os.path.exists(each_0)):
                    m_train_list_neg.append([each_key, each_0, 0])
                    count_neg += 1
            for each_1 in m_dic[each_key]['1']:
                if(os.path.exists(each_1)):
                    m_train_list_pos.append([each_key, each_1, 1])
                    count_pos += 1
        count += 1
    return m_train_list_pos, m_train_list_neg

if __name__ == '__main__':
    ref_path = 'D:/DeepSolar/coup/seq14_test/1334099/1334099_2017_0.png'
    tar_path = 'D:/DeepSolar/coup/seq14_test/1334099/1334099_2011_0.png'
    # img_list = os.listdir(tar_path_root)
    # for i in range(100, 200):
        # ref_path = ref_path_root + '/' + img_list[i]
        # tar_path = tar_path_root + '/' + img_list[i]
    m_naive_method = NaiveMethod()
    m_naive_method.test_single(ref_path, tar_path)
    # m_naive_method.mk_dataset(img_info_dic_path='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test/dataset_info_dic.txt',
    #                           result_dic_path='/home/ubuntu/projects/data/deepsolar2/couple_classify_naive/test/dataset_info_dic.txt')
    # m_naive_method.mk_dataset(img_info_dic_path='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/val/dataset_info_dic.txt',
    #                           result_dic_path='/home/ubuntu/projects/data/deepsolar2/couple_classify_naive/val/dataset_info_dic.txt')
    # sim = m_naive_method.test_single(ref_path='D:/DeepSolar/coup/seq14_test/1334177/2014_0.png',
    #                                  tar_path='D:/DeepSolar/coup/seq14_test/1334177/2008_0.png')
    # move_dataset(ref_src='/home/ubuntu/projects/data/deepsolar2/couple_classify/val/ref',
    #               ref_dst='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test/ref',
    #               img_path='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test/0')
    # match_dataset(base_path='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test',
    #               result_path='/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test/dataset_info_dic.txt')
    # clean_dataset('E:/MyData/DeepSolar/couple_classify_naive/test/dataset_info_dic.txt')
    # files = os.listdir('E:/MyData/DeepSolar/couple_classify_naive/val/1')
    # count = 0
    # for each in files:
    #     if('nouse' in each):
    #         count += 1
    # print(count)

    # file_list = os.listdir('E:/MyData/DeepSolar/couple_classify_naive/val/1')
    # for each in file_list:
    #     if('nouse' in each):
    #         src = 'E:/MyData/DeepSolar/couple_classify_naive/val/1/' + each
    #         dst = 'E:/MyData/DeepSolar/couple_classify_naive/val/1/' + each.replace('nouse.png', '.png')
    #         os.rename(src, dst)
    # list_pos, list_neg = dic2list('/home/ubuntu/projects/data/deepsolar2/couple_classify_naive/train/dataset_info_dic.txt')
    # pos_path = '/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/train/1'
    # neg_path = '/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/train/0'
    # ref_path = '/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/train/ref'
    # src_path = '/home/ubuntu/projects/data/deepsolar2/couple_classify/finaltry/test/ref'
    # dst_path = '/home/ubuntu/projects/data/deepsolar2/couple_classify_naive/test/ref_high_noise'
    # count = 0
    # for each in os.listdir(src_path):
    #     img = cv2.imread(src_path + '/' + each)
    #     result, *_ = m_naive_method.extract_ref(src_path + '/' + each, img, m_naive_method.cam_maker)
    #     cv2.imwrite(dst_path + '/' + each, result)
    #     count += 1
    #     print(count)
    


