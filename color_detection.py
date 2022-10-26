import cv2
import numpy as np
from collections import deque

#Tespit edilen merkezin memoryde ne kadar boyutda depolanacagı için belirtilir.
buffer_size=16
pts=deque(maxlen=buffer_size)

#mavi renk tespiti hsv
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

#VideoCapture
cap=cv2.VideoCapture(0)
cap.set(3,980)
cap.set(4,480)
width1=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height1=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Video kaydedici fourcc çerçeveleri kaydetmek için codec kodu
writer=cv2.VideoWriter("color_detection.mp4",cv2.VideoWriter_fourcc(*"DIVX"),
                       20,(width1,height1))

while True:
    
    success,Orgimage=cap.read()
    
    #Eğer okunan frame'ler başarılı ise 
    if success:
        
        #gürültüleri azaltmak için öncelikle blur yapılır.
        blured=cv2.GaussianBlur(Orgimage,(11,11),0)
        
        #image BGR formatından HSV formatına döndürülür
        hsv=cv2.cvtColor(blured,cv2.COLOR_BGR2HSV)
        #cv2.imshow("HSV Image",hsv)
        
        #mavi renk için maske oluşturulur.
        mask=cv2.inRange(hsv,blueLower,blueUpper)
        #cv2.imshow("Mask Image",mask)
        
        #Maskenin etrafında gürültüleri silmek için erizyon ve genişleme yapılır.
        mask=cv2.erode(mask,None,iterations=8)
        mask=cv2.dilate(mask,None,iterations=8)
        #cv2.imshow("Mask ,erozyon,genişleme",mask)
        
        #kontor tespiti
        (contours,_) = cv2.findContours(mask.copy(), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        center = None
        
        if len(contours) > 0:
            
            # en büyük kontor al
            c=max(contours,key=cv2.contourArea)
            
            #tespitin etrafına dikdörtgen çizdir.
            rect=cv2.minAreaRect(c)
            
            #çizilen rectangel değerlerini ekrana yazdır.
            ((x,y),(width,height),rotation)=rect
            
            s="x :{},y :{},width:{},height:{},rotation:{}".format(np.round(x),
                                                                  np.round(y),
                                                                  np.round(width),
                                                                  np.round(height),
                                                                  np.round(rotation))
           
            #Kutu oluşturulur.
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            
            #Tespit edilen nesnenin etrafına çizilen dikdörtgenin ortan noktası tespiti
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # konturu çizdir: sarı
            cv2.drawContours(Orgimage, [box], 0, (0,255,255),2)
            
            # merkere bir tane nokta çizelim: pembe
            cv2.circle(Orgimage, center, 5, (255,0,255),-1)
            
            # bilgileri ekrana yazdır
            cv2.putText(Orgimage, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        
        # deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(Orgimage, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Orijinal Tespit",Orgimage)
        
        writer.write(Orgimage)
        
        
    if cv2.waitKey(1) & 0xFF==ord("q"):break
    
cap.release()  # stop capture
writer.release()
cv2.distroyALLWindows()
     