import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import numpy as np
import socket
import redis

#redis server
redis_host = "localhost"
redis_port = 6379
redis_password = ""
r = redis.StrictRedis(host=redis_host,
                      port=redis_port,
                      password=redis_password,
                      decode_responses=True)

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)
Lower_hsv = np.array([20, 70, 100])
Upper_hsv = np.array([30, 255, 255])
#size
width = 480
height = 480
dim = (width, height)

url = "http://192.168.100.18:4747/video"
vcap = VideoStream(src=url).start()
time.sleep(2.0)
fps = FPS().start()
while(True):
    frame = vcap.read()
    if frame is not None:
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        rotate = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        hsv = cv2.cvtColor(rotate, cv2.COLOR_BGR2HSV)
        Mask1 = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        #blur
        blur = cv2.GaussianBlur(Mask1, (9, 9),
                       cv2.BORDER_DEFAULT)
        ret, thresh = cv2.threshold(blur, 200, 255,
                           cv2.THRESH_BINARY)
        #filter
        Mask2 = cv2.erode(thresh, kernel, iterations=1)
        Mask3 = cv2.morphologyEx(Mask2, cv2.MORPH_OPEN, kernel)
        Mask4 = cv2.dilate(Mask3, kernel, iterations=1)
        #Find countours
        contours, hierarchy = cv2.findContours(Mask4, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area>=2000:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.drawContours(rotate, [c], -1, (0, 255, 0), 2)
                    cv2.circle(rotate, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(rotate, "center", (cx - 20, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    if cx>=220 and cx<=260:
                        state = 0
                    elif cx>=160 and cx<220:
                        state = 1
                    elif cx>0 and cx<160:
                        state = 2
                    elif cx>260 and cx<=320:
                        state = -1
                    elif cx>320 and cx<4800:
                        state = -2 
                    r.set("CX",state)
                    r.set("CY",cy)
                    print(f"x: {cx} y: {cy} state : {state}")        
        cv2.imshow('frame',rotate)
        cv2.imshow('Final',Mask4)
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
        fps.update()
    else:
        print ("Frame is None")
        break
r.set("CX",999)
r.set("CY",999)
fps.stop()
cv2.destroyAllWindows()
vcap.stop()
print ("Video stop")
