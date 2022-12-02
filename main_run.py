# -*- coding: UTF-8 -*-
import cv2
import torch
from utils_processing import load_model,processing_input,process_output,show_results
from utils_processing import Push_MinIo,Push_database,img_warped_preprocess
from Read_message_consumer import ReadMessageConsumer
from load_model import load_model_arcface
import threading
import numpy as np
import time, os,io
from datetime import datetime,date
from glob_var import GlobVar,db_connect,minio_connect,Const

#Declare prameter 
weights = '/home/aitraining/Desktop/yolov5-face/weights/last.pt'
device = torch.device(Const.CUDA)
img_size = 640
conf_thres = 0.6
iou_thres = 0.5
imgsz=(640, 640)

#Declare minio and db postgres
conn, cur = db_connect()
minio_address,minio_address1, bucket_name , client = minio_connect()

#Load model detect face
model_face_detect = load_model(weights,device)
model_face_detect = model_face_detect.to(device)
#Declare Variable 
dict_check = dict.fromkeys([],1)
lst_people = []
number_reg = 20

def Gun(frame,labels,number_reg):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    dict_check[labels[0]] = dict_check[labels[0]] + 1 if labels[0] in dict_check else 1
    if dict_check[labels[0]] >= number_reg and labels[0] not in lst_people:
        lst_people.append(labels[0])
        path_img = Push_MinIo(frame,labels)
        Push_database(labels[0],path_img)
        cv2.imwrite(os.getcwd() + '/outputs/'+labels[0]+'_'+current_time +'.jpg',frame)
        print('Gun at: ',current_time)
class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        # self.rtsp = '/home/aitraining/Desktop/yolov5-face/folder_test/output.avi'
        # self.rtsp = "/home/aitraining/workspace/datnh14/LPR/dataset/20220722_095149_CAM2.mp4"
        self.rtsp = 'rtsp://vcc_cam:Vcc12345678@172.18.0.21:554/stream1'
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)
        error_frame = 0
        no_people_check = 0
        while True:
            timer = cv2.getTickCount()
            if frame_count % 1 == 0:
                try:
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
                except:
                    print('camera loi roi!')
                    cap = cv2.VideoCapture(self.rtsp)
                    error_frame += 1
                    if error_frame == 100: break
                    continue
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img_processed = processing_input(frame,img_size,model_face_detect)
                results = model_face_detect(img_processed)[0]
                result_boxes, result_scores, result_landmark = process_output(results,img_processed,frame,conf_thres,iou_thres)

                for bbox,landmarks in zip(result_boxes, result_landmark):
                    
                    landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                        landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                    landmark = landmark.reshape((2,5)).T
                    # Align face
                    nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    # cv2.imshow('check face',nimg)

                    labels, np_feature = GlobVar.arcface.predict(nimg, print_info=True)

                    frame = show_results(frame, bbox,labels[0])
                    Gun(frame,labels,number_reg)
                    if dict_check[labels[0]] >= number_reg:
                        break
                if len(result_boxes) == 0:
                    # 40 frame lien tiep k co box se xoa
                    no_people_check += 1
                    if no_people_check == 40:
                        dict_check.clear()
                        lst_people.clear()
                        no_people_check = 0
                
                else: print('dict_check:',dict_check)
                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # print("FPS:", round(FPS))
                cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
                # cv2.imshow("vid_out", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.01)
if __name__ == '__main__':
    readMessageConsumer = ReadMessageConsumer()
    readMessageConsumer.start()
    runModel = RunModel()
    runModel.start()
    loadmodel = load_model_arcface()
    