#!/usr/bin/env python
# coding=utf-8
'''
@Author: FishSzh
@Date: 2020-04-26 15:31:30
@LastEditTime: 2020-04-26 15:32:45
'''

import numpy as np
import cv2

def get_valued_nums(numRegion, nums_locs, ref_nums):
    '''
    Description:
        提取每个数字的二值化灰度图，用于与参照对比，三种方法。
        a. 二值化 
        b. Canny边缘检测
        c. tophat凸出高亮区域
    Parameters:
        numRegion[array[uint8]]-> 数字区域灰度图
        nums_locs[list[list[int]]]-> 数字坐标集合
    Returns:
        nums[list[array[uint8]]]->数字二值化灰度图集合
    '''
    best_valued_nums = []
    for num_loc in nums_locs:
        # 切取第i个数字，固定size便于与reference对比,并进行abc三种分析。
        x,y,w,h = num_loc
        num_i_region = cv2.resize(numRegion[y:y+h, x:x+w], (30,30))
        # 进行一步对比度增强，可以不做
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)) 
        num_i_region = clahe.apply(num_i_region)
        # a. 二值化 
        num_i_bi = cv2.threshold(num_i_region, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        # b. Canny, 形态学close操作用于填补数字边界中间的黑色空白
        num_i_canny = cv2.Canny(num_i_region, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        num_i_canny_close = cv2.morphologyEx(num_i_canny, cv2.MORPH_CLOSE, kernel)
        # 其实这里可以再进行一次open操作，可以去除一些孤立的线,这里暂时不用
        # num_i_canny_close_open = cv2.morphologyEx(num_i_canny_close, cv2.MORPH_OPEN, kernel)
        # c. TOPHAT，形态学open操作可以弥补部分
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        num_i_tophat = cv2.morphologyEx(num_i_region, cv2.MORPH_TOPHAT, kernel)
        num_i_tophat_bi = cv2.threshold(num_i_tophat, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1] 
        best_valued_num = get_best_valued_num(num_i_bi, num_i_canny_close, num_i_tophat_bi, ref_nums)
        best_valued_nums.append(best_valued_num)
    return best_valued_nums

def get_best_valued_num(num_i_bi, num_i_canny_close, num_i_tophat_bi, ref_nums):
    '''
    Description:
        三种方式的结果，通过与reference对比，得出最匹配的值。
    Parameters:
        num_i_bi[array[uint8]]-> 单个数字的二值化灰度图
        num_i_canny[array[uint8]]-> 通过canny得出的，单个数字的二值化灰度图
        num_i_tophat_bi[array[uint8]]-> 通过TOPHAT得出的，单个数字的二值化灰度图
    Returns:
        best_valued_num[int]->最佳匹配值
    '''
    scores = []
    for ref_num in ref_nums:
        score_bi = cv2.matchTemplate(num_i_bi, ref_num, cv2.TM_CCOEFF)[0][0]
        score_canny = cv2.matchTemplate(num_i_canny_close, ref_num, cv2.TM_CCOEFF)[0][0]
        score_tophat = cv2.matchTemplate(num_i_tophat_bi, ref_num, cv2.TM_CCOEFF)[0][0]
        scores.append(max([score_bi, score_canny, score_tophat]))
    best_valued_num = np.argmax(scores)
    return best_valued_num