import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
model_name='best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = True)
"""
results = model(img)
print(type(results))
print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)"""

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=500)
    #PIL = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    PIL = Image.fromarray(frame)
    results = model(PIL)
    
    data = results.pandas().xyxy[0]
    data = pd.DataFrame(data)
    kelas = data.shape[0]
    conf = data.get("confidence").tolist()
    cv2.imshow('YOLO', np.squeeze(results.render()))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()

