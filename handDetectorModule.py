from libraries import *


class handDetector:  # Handdetector class with Mediapipe
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):  # Draw hand lanmarks to indicate
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):  # Get landmarks of a hand

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

    def drawTag(self, tag, hand, img):  # Draw a rectangle with a tag which is the move's corresponding to
        hand = pd.DataFrame(hand)
        minX, minY = hand.iloc[:, 1].min(axis=0), hand.iloc[:, 2].min(axis=0)
        maxX, maxY = hand.iloc[:, 1].max(axis=0), hand.iloc[:, 2].max(axis=0)
        image = cv2.rectangle(img, (minX, minY), (maxX, maxY), (36, 255, 12), 1)
        if (minY - 10) > 0:
            cv2.putText(image, tag, (minX, minY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)



