import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

import pandas as pd

path='./game1/'

events=[]

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

if __name__=='__main__':
    cnt=0
    for img in sorted(os.listdir(path)):
        cnt+=1
        total_kills_teamr=0
        total_kills_teaml=0
    #        print("IMAGE: ", img)
        img_rgb = cv.imread(path+img)
#        img_rgb=cv.imread('./game1/output000047.png')
        left_crop=img_rgb[465:650,0:35,:]
        left_gray=cv.cvtColor(left_crop,cv.COLOR_BGR2GRAY)

        right_crop=img_rgb[465:650,1245:1280,:]
        right_gray=cv.cvtColor(right_crop,cv.COLOR_BGR2GRAY)

        bomb_crop=img_rgb[275:335,1225:1270,:]
        bomb_gray=cv.cvtColor(bomb_crop,cv.COLOR_BGR2GRAY)

        gun_crop=img_rgb[45:145,1080:1280,:]
        gun_gray=cv.cvtColor(gun_crop,cv.COLOR_BGR2GRAY)

        template_skull= cv.imread('./skull.png',0)
        w, h = template_skull.shape[::-1]
        res_left = cv.matchTemplate(left_gray,template_skull,cv.TM_CCOEFF_NORMED)
        res_right = cv.matchTemplate(right_gray,template_skull,cv.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc_left = np.where( res_left >= threshold)
        loc_right = np.where( res_right >= threshold)
        pts_left=[]
        pts_right=[]
        for pt in zip(*loc_left[::-1]):
            pts_left.append((pt[0],pt[1]+465,pt[0] + w, pt[1]+465 + h))
        left_arr=np.asarray(pts_left)
    #
        bbox_l=non_max_suppression_fast(left_arr,0.5)
        for (startX,startY,endX,endY) in bbox_l:
            total_kills_teaml+=1
            cv.rectangle(img_rgb, (startX,startY), (endX, endY), (0,0,255), 2)
    #        print("Total kills by Team Astralis uptil now: ", total_kills_teaml)

        for pt in zip(*loc_right[::-1]):
            pts_right.append((pt[0]+1245,pt[1]+465,pt[0] + w+1245, pt[1]+465 + h))
        right_arr=np.asarray(pts_right)

        bbox_r=non_max_suppression_fast(right_arr,0.5)
        for (startX,startY,endX,endY) in bbox_r:
            total_kills_teamr+=1
            cv.rectangle(img_rgb, (startX,startY), (endX,endY), (0,255,255), 2)
    #        print("Total kills by Team Avangar uptil now: ", total_kills_teamr)
    #
        template_bomb=cv.imread('./temp_bomb.png',0)
        wb,hb=template_bomb.shape[::-1]
        res_bomb=cv.matchTemplate(bomb_gray,template_bomb,cv.TM_CCOEFF_NORMED)
        loc_bomb=np.where(res_bomb>=threshold)
        pts_bomb=[]
        for pt in zip(*loc_bomb[::-1]):
            pts_bomb.append((pt[0]+1225,pt[1]+275,pt[0] + wb+1225, pt[1]+275 + hb))

        bomb_planted=False
        bomb_arr=np.asarray(pts_bomb)
        bbox_bomb=non_max_suppression_fast(bomb_arr,0.5)
        for (startX,startY,endX,endY) in bbox_bomb:
    #            print("BOMB PLANTED!")
            bomb_planted=True
            cv.rectangle(img_rgb, (startX,startY), (endX, endY), (255,0,255), 2)
    #
        headshot=img_rgb[45:125,1080:1280,:]
        headshot_gray=cv.cvtColor(headshot,cv.COLOR_BGR2GRAY)
        temp_headshot=cv.imread('./headshot.png',0)

        wh, hh = temp_headshot.shape[::-1]
        res_headshot = cv.matchTemplate(headshot_gray,temp_headshot,cv.TM_CCOEFF_NORMED)

        loc_headshot = np.where( res_headshot >= threshold+0.15)
        pts_headshot=[]

        for pt in zip(*loc_headshot[::-1]):
            pts_headshot.append((pt[0]+1080,pt[1]+45,pt[0] + wh+1080, pt[1]+45 + hh))
        left_headshot=np.asarray(pts_headshot)
        
        headshot=False
        bbox_headshot=non_max_suppression_fast(left_headshot,0.5)
        for (startX,startY,endX,endY) in bbox_headshot:
            headshot=True
            cv.rectangle(img_rgb, (startX,startY), (endX, endY), (50,150,55), 2)
    #
        map=img_rgb[0:227,0:207,:]
        map_gray=cv.cvtColor(map,cv.COLOR_BGR2GRAY)
        temp_map=cv.imread('./temp_map.png',0)
        #    temp_headshot=temp_headshot[50:62,1220:1236]
        
        wm, hm = temp_map.shape[::-1]
        res_map = cv.matchTemplate(map_gray,temp_map,cv.TM_CCOEFF_NORMED)
        
        loc_map = np.where( res_map >= threshold-0.15)
        pts_map=[]
        
        for pt in zip(*loc_map[::-1]):
            pts_map.append((pt[0],pt[1],pt[0] + wm, pt[1]+ hm))
        
    #        map_loc=True
    #        if pts_map is None:
    #            map_loc=False
    #        print(map_loc)
        left_map=np.asarray(pts_map)
    #
        map_loctn=False
        bbox_map=non_max_suppression_fast(left_map,0.5)
        for (startX,startY,endX,endY) in bbox_map:
            map_loctn=True
            cv.rectangle(img_rgb, (startX,startY), (endX, endY), (150,150,255), 2)

        events.append([cnt,total_kills_teaml,total_kills_teamr,bomb_planted,headshot])
        cv.imwrite('./result_game1/res_'+img,img_rgb)

    df=pd.DataFrame(events,columns=["TIME-STAMP","KILLS BY TEAM A","KILLS BY TEAM B", "BOMB", "HEADSHOT"])
    df.to_csv (r'/Users/chaitralikshirsagar/Desktop/csgo_project/events.csv', index = False, header=True)

