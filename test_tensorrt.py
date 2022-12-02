import cv2
import copy
import torch
from utils.general import check_img_size,non_max_suppression_face,scale_coords,xyxy2xywh
from utils.datasets import letterbox
from detect_face import scale_coords_landmarks
from utils_processing import show_results2,img_warped_preprocess, show_results2
from Read_message_consumer import ReadMessageConsumer,meta_data
from module.torch2trt.trt_model import TrtModel
from load_model import load_model_arcface
import threading
import numpy as np
import time
from datetime import datetime
from glob_var import GlobVar
from IOU import check_polygon_sort, clear_polygons, check_overlap
from shapely.geometry import Polygon

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
    classid = []
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
                classid.append(class_num)
    return bbox, score, landmark,classid

output_shape = [1,25200,16]
model_face_detect =TrtModel('/home/aitraining/Desktop/yolov5-face/weights/last.trt')

class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        self.rtsp = "/home/aitraining/Desktop/yolov5-face/folder_test/output.avi"
        # self.rtsp = '/home/aitraining/workspace/huydq46/yolov5_arcface/datasets/folder_test/20221104_073126_test1.mp4'
        # self.rtsp = 'rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1'
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)
        polygon_people_checks = []
        len_no_people_plate = 0
        while True:
            timer = cv2.getTickCount()
            try:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            except:
                cap = cv2.VideoCapture(self.rtsp)
                error_frame += 1
                if error_frame == 5: break
                continue
            h,w,_ = frame.shape
            coordinates= [(0,0),(w,0),(w,h),(0,h)]
            polygon_check = Polygon(coordinates)

            if frame_count % 2 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img,orgimg=img_process(frame_rgb) 
                pred=model_face_detect(img.numpy()).reshape(output_shape) # forward
                # print('pred:',pred)
                # model_face_detect.destroy()
                pred = non_max_suppression_face(torch.from_numpy(pred), conf_thres=0.5, iou_thres=0.6)
                result_boxes, result_scores, result_landmark,classid_original = Process_detections(img,orgimg,pred)
                result_boxes = np.array(result_boxes)
                labels = []

                if len(result_boxes) == 0:
                    # 40 frame lien tiep k co box se xoa
                    len_no_people_plate += 1
                    if len_no_people_plate == 40:
                        polygon_people_checks = []
                        len_no_people_plate = 0
                else:
                    # check overlap
                    boxes, scores, classid = check_overlap(result_boxes, result_scores, classid_original)
                    # Neu co bien nam trong vung polygon se khong xoa nua
                    clear = clear_polygons(boxes, polygon_check)
                    if clear == True:
                        len_no_people_plate += 1
                        if len_no_people_plate == 40:
                            polygon_people_checks = []
                            len_no_people_plate = 0
                    for i, box in enumerate(boxes):
                        if classid[i] ==0:
                            xmin, ymin, xmax, ymax = box
                            # Check line
                            box_people_point = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                            polygon_people = Polygon(box_people_point)
                            intersect = polygon_people.intersection(polygon_check).area / polygon_people.area

                            if int(round(intersect * 100, 2)) >= 90:
                                frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
                                
                                # Check box sort
                                polygon_people_checks, push = check_polygon_sort(polygon_people, polygon_people_checks, polygon_check)
                                len_no_people_plate = 0
                                if push == True:
                                    cv2.putText(frame, 'scores: ' + str(scores[i]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 70), 2)
                                    push_data = True
                                    break

                # for i in range(len(result_boxes)):
                #     landmark = np.array([result_landmark[i][0], result_landmark[i][2], result_landmark[i][4], result_landmark[i][6], result_landmark[i][8],
                #                         result_landmark[i][1], result_landmark[i][3], result_landmark[i][5], result_landmark[i][7], result_landmark[i][9]])
                #     landmark = landmark.reshape((2,5)).T
                #     # Align face
                #     nimg = img_warped_preprocess(frame_rgb, result_boxes[i], landmark, image_size='112,112')
                #     nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                #     # cv2.imshow('check face',nimg)
                #     label, np_feature = GlobVar.arcface.predict(nimg, print_info=True)
                #     labels.append(label)

                # frame = show_results2(frame, result_boxes, result_scores,labels)

                # FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # # print("FPS:", round(FPS))
                # cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
                cv2.imshow("vid_out", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == '__main__':
    readMessageConsumer = ReadMessageConsumer()
    readMessageConsumer.start()
    runModel = RunModel()
    runModel.start()
    loadmodel = load_model_arcface()

