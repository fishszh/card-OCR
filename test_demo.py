#!/usr/bin/env python
# coding=utf-8
'''
@Author: FishSzh
@Date: 2020-04-22 11:47:41
@LastEditTime: 2020-05-04 18:17:07
'''
#!/usr/bin/env python
# coding=utf-8

# %%
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
# sys.path.append('../')
from lib.imgPrep import roi, split_num
from lib.plot import fig_demo
from lib.number_rec import value_pic


# %%
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 读取参照系
    ref_nums = split_num.get_ref_nums()
    test_path = 'data/test_img/'
    img_path = 'data/JPEGImages/'
    cards = {'RegOK': ['card8.jpg', '1.jpg', '3.jpg', 'card1.png', 'card2.png', 'card3.png', 'card4.jpg', 'card5.jpg', 'card6.jpg', 'card8.jpg'],
             'RegOKAgr':  ['2.jpg', 'candy7.jpg'],
             'candy': ['candy2.jpg', 'candy6.jpeg', 'candy8.jpg', 'candy9.jpg'],
             'candy_black': ['candy3.jpg', 'candy7.jpg'],
             'cannyNo': ['candy1.jpg', 'candy4.jpg', 'candy5.jpg'],
             'test': ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg'],
             'numOK': ['1.jpg', 'card1.png', 'card4.jpg']}
    for key in ['test', 'numOK', 'RegOK', 'candy', 'candy_black'][2:3]:
        for card in cards[key]:
            print('正在识别', card)
            img = cv2.imread(test_path+card)
            print(img_path+card)
            img = cv2.resize(img, (450, 300))
            img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 获取数字区域的灰度图，坐标结构信息[x,y,w,h]，及获取数字区前的标记信息
            numRegionGray, numRegion_locs, cardRegion_tophat_dilate_bi = roi.get_nums_region(
                img2gray.copy())
            # 如果未识别数字区域直接识别下一张图
            if numRegionGray is None:
                print('未识别出数字区域...开始识别下一张')
                continue
            # 若已识别出数字区，在原图中标记
            x0, y0, w0, h0 = numRegion_locs
            img_mark = img.copy()
            cv2.rectangle(img_mark, (x0, y0), (x0+w0, y0+h0), [0, 0, 255], 3)

            # 准备需要展示的结果
            imgs = [img_mark, cardRegion_tophat_dilate_bi]
            titles = []

            # 二值化的识别结果：每个数字的坐标信息和识别判断前的效果图
            each_num_locs, numRegionGray_bi = split_num.get_nums_bi(
                numRegionGray.copy())
            # 利用二值化提取出的每个数字的坐标，来甄别数字
            valued_nums = value_pic.get_valued_nums(
                numRegionGray, each_num_locs, ref_nums)
            if each_num_locs is not None:
                img_bi = img.copy()
                for i, loc in zip(valued_nums, each_num_locs):
                    x, y, w, h = loc
                    x += 5  # 之前坐标有变动过5
                    # 将识别效果标记到原图中
                    cv2.rectangle(img_bi, (x0+x, y0+y-2),
                                  (x0+x+w, y0+y+h), [0, 0, 255], 1)
                    # 将最匹配的数字，标记在原图中
                    cv2.putText(img_bi, str(i), (x0+x-3, y0+y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                imgs.extend([numRegionGray_bi, img_bi])
                titles.append('BINARY')

            # Canny识别结果：每个数字的坐标信息和识别判断前的效果图
            each_num_locs, numRegion_canny_dilate = split_num.get_nums_canny(
                numRegionGray.copy())
            # 利用canny提取出的每个数字的坐标，来甄别数字
            valued_nums = value_pic.get_valued_nums(
                numRegionGray, each_num_locs, ref_nums)
            if each_num_locs is not None:
                img_canny = img.copy()
                for i, loc in zip(valued_nums, each_num_locs):
                    x, y, w, h = loc
                    x += 5  # 之前坐标有变动过5
                    # 将识别效果标记到原图中
                    cv2.rectangle(img_canny, (x0+x, y0+y-2),
                                  (x0+x+w, y0+y+h), [0, 0, 255], 2)
                    # 将最匹配的数字，标记在原图中
                    cv2.putText(img_canny, str(i), (x0+x-3, y0+y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                imgs.extend([numRegion_canny_dilate, img_canny])
                titles.append('CANNY')

            # Tophat 识别结果
            each_num_locs, numRegion_tophat_dilate_bi = split_num.get_nums_tophat(
                numRegionGray.copy())
            # 利用tophat提取出的每个数字的坐标，来甄别数字
            valued_nums = value_pic.get_valued_nums(
                numRegionGray, each_num_locs, ref_nums)
            if each_num_locs is not None:
                img_tophat = img.copy()
                for i, loc in zip(valued_nums, each_num_locs):
                    x, y, w, h = loc
                    x += 5  # 之前坐标有变动过5
                    # 将识别效果标记到原图中
                    cv2.rectangle(img_tophat, (x0+x, y0+y-2),
                                  (x0+x+w, y0+y+h), [0, 0, 255], 1)
                    # 将最匹配的数字，标记在原图中
                    cv2.putText(img_tophat, str(i), (x0+x-3, y0+y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                imgs.extend([numRegion_tophat_dilate_bi, img_tophat])
                titles.append('TOPHAT')

            fig_demo.roi_num_demo(imgs, titles)
 # %%


# %%


# %%


# %%

# %%
