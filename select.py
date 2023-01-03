import numpy as np
import cv2
def detectsquare(img):
    global QuantityOfSquare
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for obj in contours:
        area = cv2.contourArea(obj)
        #cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4) 
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)

        x, y, w, h = cv2.boundingRect(approx)
        Squarearray[QuantityOfSquare] = (x+(w//2),y+(h//2))
        QuantityOfSquare = QuantityOfSquare + 1
        if CornerNum == 4 : objType = "Square"
        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(imgContour,str(x+(w//2)),(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
        cv2.putText(imgContour,str(y+(h//2)),(x+(w//2),y+(h//2)+10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)


def process(image):
    hs = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_threshold = np.array([0,120,100])
    high_threshold = np.array([200, 200, 160])
    # 根据阈值构建掩膜
    mask = cv2.inRange(hs, low_threshold, high_threshold)
    res = cv2.bitwise_and(image, image,mask=mask)
    return res


fcap = cv2.VideoCapture('/home/oem/Desktop/test1（复件）.mp4')
#src = cv2.imread('/home/oem/Downloads/1670739495913.jpg')
success = 1
t = 0
a = []
nowen = []
stage = 1
numa ,numc ,nume = 0,0,0
while success:
    success, src = fcap.read()
    
    tx = ' '
    if(success):
#cv2.imshow("input",src)
        result = process(src)
        imgContour = result.copy()
        imgGray = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY) #转灰度图
        kernel = np.ones((5, 5), dtype=np.uint8)
        imgGray = cv2.dilate(imgGray, kernel,2)
        imgGray = cv2.erode(imgGray, kernel, iterations=2)
        imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)  #高斯模糊
        imgCanny = cv2.Canny(imgBlur,60,60) #Canny 算子边缘检测
        cv2.imshow('result', imgBlur)
        QuantityOfSquare = 0
        Squarearray = {}
        detectsquare(imgCanny)
        maxy=0
        miny=12345
        for i in range(QuantityOfSquare):#判断是直线还是角
            print(Squarearray[i])
            if  maxy<Squarearray[i][1]:
                maxy=Squarearray[i][1]
            if  miny>Squarearray[i][1]:
                miny=Squarearray[i][1]
        if(maxy-miny<=100) : 
            print("alignment")
            tx = "alignment"
        else : 
            print("corner")
            tx = "corner"
        sp = imgContour.shape
        h=sp[0]
        w=sp[1]
        cv2.putText(imgContour,tx,((w//2),(h//2)+10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
        cv2.imshow('Contour',imgContour)# imgcontour 是 框完长方形后的选择
        cv2.waitKey(0)
        # check the block
    a.append(tx)
    if(tx == "corner"):
        numc = numc + 1
    if(tx == "alignment"):
        numa = numa + 1
    if(tx == "empty"):
        nume = nume + 1
    if(t>=10):
        if(a[t-10] == "corner"):
            numc = numc - 1
        if(a[t-10] == "alignment"):
              numa = numa - 1
        if(a[t-10] == "empty"):
            nume = nume - 1
        maxnum = max(numc,max(numa,nume))
        if(numc == maxnum):
            nowen.append("corner")
        if(numa == maxnum):
            nowen.append("alignment")
        if(nume == maxnum):
            nowen.append("empty")
    else:
        nowen.append("")
    # check the environment
    if(stage == 1):
        if(t>=10):
            if(nowen[t]=="corner" and nowen[t-10]=="empty"):
                state = "getin"
            if(nowen[t]=="empty" and nowen[t-10]=="corner"):
                state = "stop1"
            if(nowen[t]=="alignment" and nowen[t-10]=="empty"):
                state = "stop2"
            if(nowen[t]=="empty" and nowen[t-10]=="alignment"):
                state = "getout"
                stage = stage + 1
        #first corner 
    if(stage == 2):
        if(t>=10):
            if(nowen[t]=="alignment" and nowen[t-10]=="empty"):
                state = "getin"
            if(nowen[t]=="alignment" and nowen[t-10]=="alignment"):
                state = "stop"
            if(nowen[t]=="empty" and nowen[t-10]=="alignment"):
                state = "getout"
                stage = stage + 1
    t = t + 1
cv2.destroyAllWindows()