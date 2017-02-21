# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ
# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ
# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ
# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ
# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ
# LINK DO VIDEO COM A ATIVIDADE FUNCIONANDO https://youtu.be/Gf0iUGcrRUQ


import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    #return the edged image
    return edged

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(7,7),0)
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    
    hpixel = 235
    d = 38
    h = 6.3
    f = (hpixel*d)/h    
        
    circles = []
    
    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    listaY = []
    listaX = []
    
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles=cv2.HoughCircles(bordas,cv.CV_HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=2,maxRadius=60)
    
    if circles != None and len(circles) < 3:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            #listaX e Y append            
            listaY.append(i[1])
            listaX.append(i[0])
         
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            d_ = f*h/(i[2]*4)

            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)            
            
            
    if len(listaX) > 2:
        print("distancia = {0} cm" .format(d_))
        dx = max(listaX) - min(listaX)
        dy = max(listaY) - min(listaY)
        if dx > dy:
            print("Horizontal")
            text = "Horizontal"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bordas_color,text,(0,50), font, 1,(255,255,255),2,cv2.CV_AA)
        else:
            print("Vertical")
            text = "Vertical"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bordas_color,text,(0,50), font, 1,(255,255,255),2,cv2.CV_AA)
   
    cv2.imshow('Detector de circulos',bordas_color)            
    # Display the resulting frame
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()