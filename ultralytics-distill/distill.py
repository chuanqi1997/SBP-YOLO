import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller


if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/home/server/Work/yolo11/mask/ultralytics-yolo11-main/zzz/exp559/weights/best.pt',
        'data' : '/home/server/Work/yolo11/mask/ultralytics-yolo11-main/datasets/yolo_0422-ALL_7K5_7_1_1.yaml',
        'imgsz': 640,
        'epochs': 500,
        'batch': 32,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD', # 

        'device': '0',

        'patience' : 50,
        
        'close_mosaic': 0, # 20
        'project':'runs/distill',
        'name':'yolov8n-chsim-exp1',
        
        # distill
        'prune_model': False,
        'teacher_weights':'/home/server/Work/yolo11/mask/ultralytics-yolo11-main/zzz/exp561/weights/best.pt',
        'teacher_cfg': '/home/server/Work/yolo11/mask/ultralytics-yolo11-main/zzz/exp561/config_backup.yaml',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.50,
        
        'teacher_kd_layers': '12,15,18,21,24,27',
        'student_kd_layers': '12,15,18,21,24,27',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.50
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()