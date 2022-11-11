from glob_var import Const,db_connect, kafka_connect, minio_connect
from kafka.errors import KafkaError
from kafka import KafkaConsumer, TopicPartition
import threading, json
import time
from datetime import datetime
from glob_var import GlobVar

topic_event,topic_face,producer = kafka_connect()
conn,cur = db_connect()

var = Const()

TOPIC_EVENT     = var.TOPIC_EVENT
kafka_broker = var.KAFKA_BROKER

consumer = KafkaConsumer(TOPIC_EVENT,bootstrap_servers=kafka_broker,auto_offset_reset ='latest')
class ReadMessageConsumer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.name = "Thread -- ReadMessageConsumer"
    
    def run(self):
        while True:
            print("reading...")
            GlobFunc.readMessage()
            time.sleep(2)

class GlobFunc():
    def __init__(self, parent =None):
        super().__init__()

    def readMessage():
        message = consumer.poll(1.0)
        if len(message.keys()) == 0:
            pass
        if TopicPartition(topic=TOPIC_EVENT, partition=0) in message.keys():
            data = message[TopicPartition(topic=TOPIC_EVENT, partition=0)]
            print(data)
            try:
                GlobVar.dict_cam =[]
                for _ in range(data.__len__()):
                    cam = camera()
                    cam.status = json.loads(data[_].value)['status']
                    cam.indexURL = json.loads(data[_].value)['indexURL']
                    GlobVar.dict_cam.append(cam)
                print("mess done!")
                return True

            except Exception as e:
                print(e)
                return False
            
class camera():
    status = None
    indexURL = None

