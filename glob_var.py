import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from kafka import KafkaProducer
import os,sys
from minio import Minio

class GlobVar(object):
    dict_data = []
    arcface = None
    check_run = False

class Const():
# PostgreSQL
    DB_HOST         = '10.248.243.110'
    DB_NAME         = 'vcc_ai_events'
    DB_USER         = 'postgres'
    DB_PASS         = 'Vcc_postgres@2022'
    DB_PORT         = 5432
# Kafka define
    KAFKA_BROKER    = '10.248.243.110:39092' #8006 kafdrop
    TOPIC_EVENT     = 'face_vcc'
    TOPIC_FACE      = 'FACE_Msg'
# MinIo define
    # MINIO_ADDRESS   = 'minio.congtrinhviettel.com.vn' 
    MINIO_ADDRESS   = '10.248.243.110:9000' #9001
    BUCKET_NAME     = 'ai-images'
    ACCESS_KEY      = 'minioadmin'
    SECRET_KEY      = 'Vcc_AI@2022'
# cuđa
    CUDA            = 'cuda:2'

var = Const()

def db_connect():
    try:
        conn = psycopg2.connect(host    =   var.DB_HOST, 
                                database=   var.DB_NAME, 
                                user    =   var.DB_USER, 
                                password=   var.DB_PASS,
                                port    =   var.DB_PORT)
        cur = conn.cursor()
    except psycopg2.DatabaseError as error:
        print("error while run",error)
    return conn, cur

def kafka_connect():
    kafka_broker    = var.KAFKA_BROKER
    topic_event     = var.TOPIC_EVENT
    topic_face      = var.TOPIC_FACE
    producer        = KafkaProducer(bootstrap_servers=kafka_broker)
    return topic_event,topic_face,producer

def minio_connect():
    minio_address   = var.MINIO_ADDRESS
    minio_address1  = 'minio.congtrinhviettel.com.vn'
    bucket_name     = var.BUCKET_NAME
    client          = Minio(
                            minio_address,
                            access_key=var.ACCESS_KEY,
                            secret_key=var.SECRET_KEY,
                            secure=False  # check ssl
                            )
    return minio_address,minio_address1,bucket_name,client