import ctypes
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import time
from module.arcface_paddle import insightface_paddle as face
from warnings import filterwarnings
import threading, os
from glob_var import GlobVar
from glob_var import minio_connect
minio_address, minio_address1, bucket_name, client = minio_connect()

filterwarnings(action='ignore', category=DeprecationWarning, message='Use execute_async_v2 instead')
PLUGIN_LIBRARY = "model/retinaface/mobilenet/libdecodeplugin.so"


parser = face.parser()
args = parser.parse_args()
args.output = "output/"
args.use_gpu = True
args.rec = True
args.enable_mkldnn = False
args.rec_model = "ArcFace" #ArcFace
args.index = "datasets/index.bin"
# args.index = '/home/aitraining/Desktop/yolov5-face/datasets/index copy.bin'

class load_model_arcface(threading.Thread):
    def __init__(self):
        super().__init__()
        if not threading.Thread.is_alive(self):
            self.start()

    def run(self):
        GlobVar.arcface = face.InsightFace(args)
        while True:
            try:
                if GlobVar.dict_data.__len__() != 0:
                    embedding_name = 'FACE/data/index.bin'
                    client.fget_object(bucket_name=bucket_name,object_name =embedding_name ,file_path=os.getcwd()+"/datasets/index.bin")
                    GlobVar.arcface = face.InsightFace(args)
                    print("load arcface done!")
                    GlobVar.dict_data.clear()
            except BaseException as error:
                print('loi me roi:',error)
            time.sleep(0.01)