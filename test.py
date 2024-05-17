from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('cocbest.pt')

#results = model.train(data='/work/cur/ccauto2/yolodata/coc/yolo8.yaml', epochs=10, imgsz=640, workers=0)

# Perform object detection on an image using the model
results = model('../../../../work/cur/ccauto2/yolodata/coc/images/all/testfullimg_cap_2024-01-13-044136.png', imgsz=640, workers=0)

print(results[0].boxes.xywh,'results.0','results[0].boxes.cls',results[0].boxes.cls)

path = model.export(format="onnx")
print('path',path)