from ultralytics import YOLO
import cv2
import shutil
import os

model = YOLO('cocbest1792.pt')

imgFileName = '/utils/src/vision/yolodata/captured.jpg'
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


