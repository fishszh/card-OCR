#!/usr/bin/env python
# coding=utf-8
'''
@Author: FishSzh
@Date: 2020-04-22 11:33:41
@LastEditTime: 2020-04-26 15:40:46
'''

import cv2 
import numpy as np

def get_card_region(gray_img, RE_ROWS=300, RE_COLS=450):
    '''
    Description:
        提取卡片区域。高斯模糊->Canny边缘检测->二值化->找轮廓->轮廓判断。
        分两种情况：
        a. 已裁剪，直接返回，固定size灰度图片。
        b. 未裁剪，找到轮廓，利用形态转换，返回固定size图片。
    Parameters:
        grayImg[array[uint8]]-> 灰度图片
        RE_ROWS[int]-> 返回图片row尺寸
        RE_COLS[int]-> 返回图片column尺寸
    Returns:
        cardRegion[array[uint8]]: 返回固定size(RE_COLS, RE_ROWS)灰度图片
    '''
    # 高斯模糊：去除小的噪声影响，平滑图像。
    grayBlur = cv2.GaussianBlur(gray_img, (8,6), 0)
    # 2. Canny边缘检测：检测整个银行卡的边缘
    edged = cv2.Canny(grayBlur, 100, 255)
    # 3. 二值化：将图像转化为黑白图像，便于轮廓提取
    edgedBi = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    edgedBi = cv2.erode(edgedBi, np.ones((5,3), 'uint8'))
    # 4. 找轮廓：画出检测的轮廓
    contours, hierarchy = cv2.findContours(edgedBi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 固定size
    cols, rows = gray_img.shape[:2]
    cardCnt = None

    if len(contours) > 0:
        cnts = sorted(contours, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
        x,y,w,h = cv2.boundingRect(cnts[0])
        for c in cnts:
            peri = cv2.arcLength(c, True)                           # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
            # 未分割提取卡片部分的图片
            con1 = (cols+rows)*2*0.5 <=peri<=(cols+rows)*2*0.95
            # 已分割的图片
            con2 = 261/176*0.9<w/h<261/176*1.1
            if len(approx) == 4 and  (con1 or con2):
                cardCnt = approx
                break
    if cardCnt is None:
        print('未能识别卡片')
        return None
    else:   
        pts1 = cardCnt.reshape((4,2)).astype('float32')
        pts1 = pts1[np.argsort(np.sum(pts1, axis=1))]
        pts2 = np.float32([[0, 0], [0, RE_ROWS], [RE_COLS, 0],  [RE_COLS, RE_ROWS]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        cardRegion = cv2.warpPerspective(gray_img, m, (RE_COLS, RE_ROWS))
        return cardRegion

def get_nums_region(cardRegion):
    '''
    Description:
        获取数字区域。
        高斯模糊->tophat->二值化|横向膨胀->找轮廓->轮廓判断。
        二值化和横向膨胀顺序可以互换，但先横向膨胀，再二值化，效果好些。
    Parameters:
        cardRegion[array[uint8]]->灰度图
    Returns:
        numRegion[array[uint8]]->数字区域灰度图
        numRegion_locs[list[int]]-> 数字区域[x,y,w,h]
        tophat_dilate_bi-> 卡片的tophat—dilate-二值化后效果图
    '''
    # 1. 高斯模糊
    cardRegion_blur = cv2.GaussianBlur(cardRegion, (5,5), 0)
    # 2. 礼帽操作，凸出高亮区域
    carRegion_tophat = cv2.morphologyEx(cardRegion_blur, cv2.MORPH_TOPHAT, np.ones((3,3), 'uint8'))
    # 3. 横向膨胀，将数字区连接起来
    cardRegion_tophat_dilate = cv2.dilate(carRegion_tophat, kernel=np.ones((2,45), 'uint8'))
    # 4. 二值化
    cardRegion_tophat_dilate_bi = cv2.threshold(cardRegion_tophat_dilate, 0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # 5. 利用轮廓条件筛选出数字区域。
    contours, _ = cv2.findContours(cardRegion_tophat_dilate_bi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_size = cardRegion_tophat_dilate_bi.shape[:2]
    numRegion_locs = None
    numRegion = None
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if 0.6*card_size[1]<w<0.98*card_size[1] and 0.05*card_size[0]<h<0.2*card_size[0]:
            numRegion_locs = [x,y,w,h]
            # 由于横向膨胀导致这里的区域变大，所以这里缩小5
            numRegion = cardRegion[y-2:y+h+2,x+5:x+w]
            break
    return  numRegion, numRegion_locs, cardRegion_tophat_dilate_bi


