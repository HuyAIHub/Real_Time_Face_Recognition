import cv2
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
from skimage import transform as trans
from models.experimental import attempt_load
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    print("load model done!!")
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, name):
    print('img.shape:',img.shape)
    # print('name:',name)
    h,w,c = img.shape
    tl = 2
    # tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    # for i in range(5):
    #     point_x = int(landmarks[2 * i])
    #     point_y = int(landmarks[2 * i + 1])
    #     cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    # tf = max(tl - 1, 1)  # font thickness
    score = str(conf)[:5]
    cv2.putText(img,name[0], (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img

def processing_input(im,img_size,model):
    im = letterbox(im,new_shape = img_size)[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    im = np.ascontiguousarray(im)
    if len(im.shape) == 4:
        orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
    else:
        orgimg = im.transpose(1, 2, 0)

    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s= model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert from w,h,c to c,w,h
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def process_output(pred,img,im0,conf_thres,iou_thres):
    bbox = []
    score = []
    landmark = []
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                bbox.append(xyxy)
                score.append(conf)
                landmark.append(landmarks)
            #     im0 = show_results(im0, xyxy, conf, landmarks)
            # cv2.imshow('result', im0)
            # cv2.waitKey(0)
    return bbox, score, landmark

def img_warped_preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
    else: #do align using landmark
        
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

        return warped

        