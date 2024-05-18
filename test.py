from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO('cocbest.pt')
#path = model.export(format="onnx")
#print('path',path)
#results = model.train(data='/work/cur/ccauto2/yolodata/coc/yolo8.yaml', epochs=10, imgsz=640, workers=0)

# Perform object detection on an image using the model
# results = model('../../../../work/cur/ccauto2/yolodata/coc/images/all/testfullimg_cap_2024-01-13-044136.png', imgsz=640, workers=0)

# D:\work\cur\ccauto2\ConsoleApp1\bin\Debug\net8.0\t1.png
imgFileName = '../../../../work/cur/ccauto2/ConsoleApp1/bin/Debug/net8.0/t1.png'
results = model(imgFileName, imgsz=640, workers=0)

print(results[0].boxes.xywh,'results.0','results[0].boxes.cls',results[0].boxes.cls)

img = cv2.imread(imgFileName)
for rec in results[0].boxes.xywh:    
    xy = (int(rec[0].item()), int(rec[1].item()))
    wh = (int(rec[2].item()), int(rec[3].item())) 
    xy = (xy[0] - int(wh[0]/2), xy[1] - int(wh[1]/2))
    wh = (wh[0] + xy[0], wh[1] + xy[1])
    print(xy, 'xy', wh)
    r = cv2.rectangle(img, xy, wh, color=(255,255,0), thickness=2)

cv2.imwrite('testout.jpg', img)


