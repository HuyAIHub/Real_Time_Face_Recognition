import cv2
import copy
import torch
from utils.general import check_img_size,non_max_suppression_face,scale_coords,xyxy2xywh
from utils.datasets import letterbox
from detect_face import scale_coords_landmarks
from utils_processing import show_results,img_warped_preprocess
from Read_message_consumer import ReadMessageConsumer,meta_data
from torch2trt.trt_model import TrtModel
from load_model import load_model_arcface
import threading
import numpy as np
import time, os
from datetime import datetime
from glob_var import GlobVar

def img_process(image,long_side=640,stride_max=32):
    img0 = copy.deepcopy(image)
    h0, w0 = image.shape[:2]  # orig hw
    r = long_side/ max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz,auto=False)[0] # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img,image

def Process_detections(img,orgimg,pred,vis_thres = 0.6):
    bbox = []
    score = []
    landmark = []
    no_vis_nums=0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                if det[j, 4].cpu().numpy() < vis_thres:
                    no_vis_nums+=1
                    continue
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                bbox.append(xyxy)
                score.append(conf)
                landmark.append(landmarks)
    return bbox, score, landmark

Vcc_people = []
output_shape = [1,25200,16]
model_face_detect =TrtModel('/home/aitraining/Desktop/yolov5-face/weights/last.trt')
Check_name = []

class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rtsp = "/home/aitraining/Desktop/yolov5-face/folder_test/output.avi"
        # self.rtsp = 'rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1'
        self.name_check = None
        self.time_check1 = None
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)
        
        while True:
            timer = cv2.getTickCount()
            
            # try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            if not ret:
                cap = cv2.VideoCapture(self.rtsp)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img,orgimg=img_process(frame_rgb) 
            pred=model_face_detect(img.numpy()).reshape(output_shape) # forward
            # print('pred:',pred)
            # model_face_detect.destroy()
            pred = non_max_suppression_face(torch.from_numpy(pred), conf_thres=0.5, iou_thres=0.6)
            result_boxes, result_scores, result_landmark = Process_detections(img,orgimg,pred)

            for i, (bbox,score,landmarks) in enumerate(zip(result_boxes, result_scores, result_landmark)):
                current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
                landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                    landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                landmark = landmark.reshape((2,5)).T
                # Align face
                nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                cv2.imshow('check face',nimg)
                labels, np_feature = GlobVar.arcface.predict(nimg, print_info=True)
                frame = show_results(frame, bbox, score, landmarks,labels)
                if labels != 'unknown' and labels not in Vcc_people:
                    
                #     # Vcc_people.append(labels)
                    Vcc_people[labels] = time.time()
                    Vcc_people.append([{'name':labels,
                    'time_start':time.time(),
                    'duration':30,
                    'flag':False}])
                    if Vcc_people['flag'] 

                #     # self.time_check1 = time.time()
                #     # self.name_check = labels
                #     # if int(time.time() - self.time_check1) == 1 :
                #     print('========================SEND=====================')
                #     cv2.imwrite(os.getcwd() + '/outputs/'+labels[0]+'_'+current_time +'.jpg',frame)
                #     # Vcc_people
                # else:
                #     Check_name.append(labels)
                #     if Check_name.count(labels) > 10 and labels not in Vcc_people :
                #         Vcc_people.append(labels)
            
            
            print('Vcc_people:',Vcc_people)
            FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # print("FPS:", round(FPS))
            cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
            cv2.imshow("vid_out", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame_count += 1
            # except Exception as error:
            #     print("Error:",error)
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     time.sleep(1)

if __name__ == '__main__':
    readMessageConsumer = ReadMessageConsumer()
    readMessageConsumer.start()
    runModel = RunModel()
    runModel.start()
    loadmodel = load_model_arcface()

