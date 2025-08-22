from ultralytics import YOLO

model = YOLO(r'pre_trained_weights/sbp-yolo-nwdloss.pt')

res = model.val(
            cache = False,
            # data = r'datasets/yolo_0422-ALL_7K5_7_1_1_Test.yaml',
            data = r'datasets/yolo_0422-ALL_7K5_7_1_1.yaml',
            imgsz=640, 
            batch=32,   # batch  需要设置为两倍训练batch
            half = True,#  需要加这个Half        
            # iou=0.7,    # iou0.7也是缺省参数
            # conf=0.25, # 这个conf 0.25是缺省参数
            save_json=True,
            device="0")

