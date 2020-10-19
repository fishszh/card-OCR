#!/usr/bin/env python
# coding=utf-8
'''
@Author: FishSzh
@Date: 2020-04-26 15:39:06
@LastEditTime: 2020-04-26 15:39:42
'''

import numpy as np
import cv2

def get_nums_locs(contours):
    '''
    Description:
        利用contour提取数字的坐标信息[x,y,w,h]
    Parameters:
        contours[list[array[int]]]->数字的轮廓信息
    Returns:
        num_locs[list[int]]->数字的矩形信息[x,y,w,h],按从左到右排序。
    '''
    num_locs = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if 0.3<w/h<0.90 and 8<w<25 and 15<h<35:
            num_locs.append([x,y,w,h])
    num_locs = sorted(num_locs, key=lambda x: x[0])
    return num_locs

def get_nums(numRegion):
    '''
    Description:
        提取每个数字的坐标，三种方法。
        a. 二值化
        b. Canny边缘检测
        c. tophat凸出高亮区域
    Parameters:
        numRegion[array[uint8]]-> 数字区域灰度图
    '''
    # 首选二值化
    num_locs, numRegion_bi = get_nums_bi(numRegion)
    if 16<=len(num_locs)<=19:
        return num_locs, numRegion_bi
    # 其次tophat
    num_locs, numRegion_tophat_dilate_bi = get_nums_tophat(numRegion)
    if 16<=len(num_locs)<=19:
        return num_locs, numRegion_tophat_dilate_bi
    # 最后Canny
    num_locs, numRegion_canny_dilate = get_nums_canny(numRegion)
    if 16<=len(num_locs)<=19:
        return num_locs, numRegion_canny_dilate
    return None, None

def get_nums_bi(numRegion):
    '''
    Description:
        直接利用二值化提取数字。
        适用情况需要满足三个条件：
        a. 数字高亮，
        b. 与背景有明显亮度差异的卡片。
        c. 凸面字体
    Parameters:
        numRegion[array[uint8]]-> 数字区域灰度图
    Returns:
        num_locs[list[int]]->数字的矩形信息[x,y,w,h],按从左到右排序。
    '''
    numRegion_bi = cv2.threshold(numRegion, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    contours_bi, _ = cv2.findContours(numRegion_bi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_locs = get_nums_locs(contours_bi)
    N = len(num_locs)
    # if 16<=N<=19:
    print('利用二值化，未能识别全部数字...识别出其中 %d 个' %N)
    return num_locs, numRegion_bi
    # else:
    #     return None, None

def get_nums_canny(numRegion):
    '''
    Description:
        利用canny提取数字。
        适用情况：数字边缘清晰,满足下列之一
        a. 数字高亮或者数字磨损
        b. 与背景有明显亮度差异的卡片。
        c. 凸面字体|印刷体
    Problems:
        a. 轮廓可能不清晰
        b. 背景可能有轮廓
    Parameters:
        numRegion[array[uint8]]-> 数字区域灰度图
    Returns:
        num_locs[list[int]]->数字的矩形信息[x,y,w,h],按从左到右排序。
    '''
    numRegion_canny = cv2.Canny(numRegion, 100, 200)
    # 纵向膨胀一下使单个数字连在一起，kernel可以自己调
    numRegion_canny_dilate = cv2.dilate(numRegion_canny, np.ones((2,1), 'uint8'))
    contours_canny, _ = cv2.findContours(numRegion_canny_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_locs = get_nums_locs(contours_canny)
    num_locs = sorted(num_locs, key=lambda x: x[0])
    N = len(num_locs) 
    # if 16<=N<=19:
    # else:
        # return None, None
    print('利用canny，未能识别全部数字...识别出其中 %d 个' %N)
    return num_locs, numRegion_canny_dilate

def get_nums_tophat(numRegion):
    '''
    Description:
        先进行一步tophat可以凸出高亮部分，去除背景，可以解决部分直接canny，背景线条带来的影响。
    Parameters:
        numRegion[array[uint8]]-> 数字区域灰度图
    Returns:
        num_locs[list[int]]->数字的矩形信息[x,y,w,h],按从左到右排序。
    '''
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (4,5))
    # 先进行一步tophat,凸出高亮部分，去除背景
    numRegion_tophat = cv2.morphologyEx(numRegion.copy(), cv2.MORPH_TOPHAT, kernel)
    # tophat 后数字可能丢失部分信息，dilate可以使原来数字区域饱满些
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,25))
    numRegion_tophat_dilate = cv2.dilate(numRegion_tophat,kernel)
    # 二值化使轮廓更清晰
    numRegion_tophat_dilate_bi = cv2.threshold(numRegion_tophat_dilate, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    contours_tophat, _ = cv2.findContours(numRegion_tophat_dilate_bi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_locs = get_nums_locs(contours_tophat)
    N = len(num_locs)
    # if 16<=N<=19:
    print('利用tophat，未能识别全部数字...识别出其中 %d 个' %N)
    return num_locs, numRegion_tophat_dilate_bi
    # else:
    #     return None, None
            
def get_ref_nums():
    ref = cv2.imread('reference.png')
    refGray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    refGray = cv2.threshold(refGray, 0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    contours, *_ = cv2.findContours(refGray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_nums_locs = []
    ref_nums = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if 0.5<w/h<0.95 and 30<w<50 and 40<h<70:
            ref_nums_locs.append([x,y,w,h])
            ref_nums_i = cv2.resize(refGray[y:y+h,x:x+w],(30,30))
            ref_nums.append(ref_nums_i)
    ref_nums_join = list(zip(ref_nums_locs, ref_nums)) 
    ref_nums_join = sorted(ref_nums_join, key=lambda x: x[0][0])
    ref_nums_locs, ref_nums = list(zip(*ref_nums_join))
    return ref_nums
