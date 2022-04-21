from libraries import *

#To calculate FPS
pTime = 0
cTime = 0
detector = handDetector()
cap = cv2.VideoCapture(0)
pr = preprocess()
model = LRmodel()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    if lmlist.__len__() != 0 and  pr.measureDiag(lmlist) > 100: #100 value is for my camera. You can calculate this value with function above: measureDiag()
        tag = model.predict(lmlist).tag
        detector.drawTag(tag[0], lmlist, img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    window = cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break