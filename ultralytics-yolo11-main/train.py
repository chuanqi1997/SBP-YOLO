import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import sys
import argparse
import os

yaml_path = r'models/SBP-YOLO.yaml'
data_path = r'datasets/yolo_0422-ALL_7K5_7_1_1.yaml'

def main(opt):
    yaml = opt.cfg
    data = opt.data
    model = YOLO(yaml)
    model.info()
    results = model.train(
                cache = False,
                data =  data,
                imgsz=640,  # 训练图片大小，默认640
                epochs=300,  # 训练轮次，默认100
                batch=16,  # 训练批次，默认16 32 64
                project='zzz',
                name='exp',  # 用于保存训练文件夹名，默认exp，依次累加
                device='0',  # 要运行的设备 device =0 是GPU显卡训练，device = cpu
                patience=30,
                seed=0,
                lr0=0.001,
                optimizer='Adam', # Adam 'SGD', 'AdamW', 'NAdam', 'RAdam'
                close_mosaic=0,
                amp=True,
                # amp=False,
                workers=8, #  需要设置为8 
                simplify=True # 默认为true
            )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=yaml_path, help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--data', type=str, default=data_path, help='data  path')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)