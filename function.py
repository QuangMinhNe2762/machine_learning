import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import random
import glob
from yolo import YOLO

path_du_lieu_hl = ".\\train_solution_bounding_boxes (1).csv"
path_hinh_anh = ".\\training_images\\"
path_hinh_anh_test = (".\\testing_images\\")

def lay_du_lieu_hl():
    data = pd.read_csv(path_du_lieu_hl)
    return data


def lay_toa_do_doi_tuong():
    data = pd.read_csv(path_du_lieu_hl)
    data.head()
    return len(data)


def random_doi_tuong_hl():
    data = lay_du_lieu_hl()
    data.head()
    rdnumber = random.randint(0, len(data) - 1)
    return rdnumber


def hien_thi_doi_tuong_random(rdnumber):
    data = lay_du_lieu_hl()
    ten_hinh_anh = data.loc[rdnumber, "image"]
    hinh_anh = plt.imread(path_hinh_anh + ten_hinh_anh)
    print(rdnumber, ten_hinh_anh)
    return hinh_anh


def phat_hien_doi_tuong_bang_du_lieu_HL(rdnumber):
    data = lay_du_lieu_hl()
    ten_hinh_anh = data.loc[rdnumber, "image"]
    number = rdnumber
    sl = (data["image"] == ten_hinh_anh).sum()
    hinh_anh = plt.imread(path_hinh_anh + ten_hinh_anh)
    while True:
        xmin = data.loc[number, "xmin"]
        ymin = data.loc[number, "ymin"]
        xmax = data.loc[number, "xmax"]
        ymax = data.loc[number, "ymax"]
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        hinh_anh = cv2.rectangle(hinh_anh, pt1, pt2, (255, 0, 0), 2)
        if number - rdnumber == sl - 1:
            break
        else:
            number += 1
    return hinh_anh


def anh_rcnn_hoatDong(rdnumber):
    data = lay_du_lieu_hl()
    ten_hinh_anh = data.loc[rdnumber, "image"]
    im = cv2.imread(path_hinh_anh + ten_hinh_anh)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def rcnn_hoatDong(rdnumber):
    data = lay_du_lieu_hl()
    ten_hinh_anh = data.loc[rdnumber, "image"]
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    im = cv2.imread(path_hinh_anh + ten_hinh_anh)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    i = 0
    # vòng lặp vẽ các hình chữ nhật vào hình ảnh thông qua dữ liệu của biến rects
    for rect in rects:
        x, y, w, h = rect
        imOut = cv2.rectangle(im, (x, y), (x + w, y + h), (i, i, i), 1, cv2.LINE_AA)  # cv2.LINE_AA kiểu viền
        i += 1
    return imOut, len(rects)


def lay_du_lieu_test():
    datatest = glob.glob(path_hinh_anh_test + "*.jpg")
    return datatest


def sl_du_lieu_test():
    datatest = glob.glob(path_hinh_anh_test + "*.jpg")
    return len(datatest)


def random_STT_hinh_anh_test():
    sl = sl_du_lieu_test()
    rdnumber = random.randint(0, sl - 1)
    return rdnumber


def lay_hinh_anh_test():
    rdnumber = random_STT_hinh_anh_test()
    datatest = lay_du_lieu_test()
    img_path = datatest[rdnumber]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def funcyolo(img):
    vdYolo = YOLO()
    vehicle_boxes = vdYolo.detect_vehicles(img)
    return vehicle_boxes


def get_iou(bb1, bb2):
    assert bb1["x1"] < bb1["x2"]  # bb1
    assert bb1["y1"] < bb1["y2"]

    assert bb2["x1"] < bb2["x2"]  # bb2
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def test_rcnn(img):
    result_rcnn = []
    k = 0
    l = 0
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    resultyolo = funcyolo(img)
    percentyolo=[]
    for box in resultyolo:
        x, y, w, h = box[0]
        percentyolo.append(box[1])
        bb1 = {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)}
        try:
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()
            for i in rects:
                x1, y1, w1, h1 = i  # Selective bounty boxxes
                bb2 = {"x1": x1, "y1": y1, "x2": x1 + w1, "y2": y1 + h1}
                boxrcnn=(bb2["x1"],bb2["y1"],bb2["x2"],bb2["y2"])
                if k < l:
                    if 0.5 < get_iou(bb1, bb2):
                        result_rcnn.append([boxrcnn, 1,get_iou(bb1, bb2)])
                        k += 1
                else:
                    if 0.5 < get_iou(bb1, bb2):
                        result_rcnn.append([boxrcnn, 1,get_iou(bb1, bb2)])
                        k += 1
                    else:
                        result_rcnn.append([boxrcnn, 0,get_iou(bb1, bb2)])
                        l += 1
        except Exception as e:
            print("lỗi : ", e)
    data = []
    data_label = []
    percentrcnn=[]
    for features, label,percent in result_rcnn:
        data.append(features)
        data_label.append(label)
        percentrcnn.append(percent)
    data=np.asarray(data)
    data_label=np.asarray(data_label)
    slPhatHienDoiTuong="Số đối tượng phát hiện của RCNN:", len(data)
    phanLoaidoiTuong='Không có hình xe:',len(data_label[data_label==0]),'Có hình xe:',len(data_label[data_label==1])
    for i in range(0,len(data)):
        if(data_label[i]==1):
            x1=data[i][0]
            y1=data[i][1]
            x2=data[i][2]
            y2=data[i][3]
            img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1,cv2.LINE_AA)
    for box in resultyolo:
        x, y, w, h = box[0]
        img=cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0),1,cv2.LINE_AA)
    return img,percentrcnn,percentyolo,slPhatHienDoiTuong,phanLoaidoiTuong,len(resultyolo),len(data_label[data_label==1])
