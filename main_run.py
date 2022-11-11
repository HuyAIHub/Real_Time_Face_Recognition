# -*- coding: UTF-8 -*-
import numpy as np
import time
from datetime import datetime
import cv2
import torch
from utils_processing import load_model,processing_input,process_output,img_warped_preprocess,show_results
from load_model import load_model_arcface
import threading, os
from Read_message_consumer import ReadMessageConsumer
from glob_var import minio_connect
from glob_var import GlobVar

minio_address, minio_address1, bucket_name, client = minio_connect()
weights = os.getcwd() + '/weights/last.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 640
conf_thres = 0.6
iou_thres = 0.5
imgsz=(640, 640)
model_face_detect = load_model(weights,device)

class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        # self.rtsp = "/home/aitraining/Desktop/yolov5-face/datasets/videos_input/Ok_Daenerys Targaryen.mp4"
        self.rtsp = 'rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1'
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)
        error_frame = 0
        while True:
            timer = cv2.getTickCount()
            try:
                # print('check!')
                # print('GlobVar:',GlobVar.dict_cam[-1].status)
                if GlobVar.arcface.args.max_batch_size == 1 and frame_count % 1 == 0:
                    try:
                        ret, frame = cap.read()
                        frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
                    except:
                        cap = cv2.VideoCapture(self.rtsp)
                        error_frame += 1
                        if error_frame == 5: break
                        continue
                    
                    # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                    if not ret:
                        cap = cv2.VideoCapture(self.rtsp)
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_count += 1
                    img_processed = processing_input(frame,img_size,model_face_detect)
                    results = model_face_detect(img_processed)[0]
                    result_boxes, result_scores, result_landmark = process_output(results,img_processed,frame,conf_thres,iou_thres)

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

                    FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    # # print("FPS:", round(FPS))
                    cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
                    cv2.imshow("vid_out", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    
            except Exception as error:
                print("Error:",error)
                time.sleep(1)
                continue
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.01)
if __name__ == '__main__':
    readMessageConsumer = ReadMessageConsumer()
    readMessageConsumer.start()
    runModel = RunModel()
    runModel.start()
    loadmodel = load_model_arcface()