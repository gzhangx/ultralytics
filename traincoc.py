from ultralytics import YOLO
import cv2
import shutil
import os

#data argument.py contains the loader
# Load a pretrained YOLO model (recommended for training)
if os.path.exists('./runs/detect/train/weights/best.pt'):
    print('copying best to cocbest.pt')
    shutil.copy('./runs/detect/train/weights/best.pt', 'cocbest992.pt')

shutil.rmtree('./runs/detect/train')    
model = YOLO('cocbest992.pt')

# 982,567 => 992
results = model.train(data='../yolodata/cocyolo8.yaml', epochs=100, imgsz=992, workers=0)
path = model.export(format="onnx")
print('path',path)
print('copy onnx to ccauto2')
shutil.copy('runs/detect/train/weights/best.onnx', './cocbest992.onnx')
shutil.copy('./cocbest992.onnx', '/work/cur/ccauto2/ScreenCapture/bin/Debug/cocbest992.onnx')
shutil.copy('./runs/detect/train/weights/best.pt', 'cocbest992.pt')

#aa=bb
# Perform object detection on an image using the model
# results = model('../../../../work/cur/ccauto2/yolodata/coc/images/all/testfullimg_cap_2024-01-13-044136.png', imgsz=640, workers=0)

# D:\work\cur\ccauto2\ConsoleApp1\bin\Debug\net8.0\t1.png
imgFileName = '/utils/src/vision/yolodata/tt.png'
results = model(imgFileName, imgsz=992, workers=0)

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


