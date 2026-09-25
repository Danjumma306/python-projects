import cv2
import mediapipe as mp
import time
import math


class handDetector:
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.9, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity  # FIX #7: now a proper class attribute
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.lnList = []  # FIX #2: initialise early so fingersUp() never crashes

    def fingersUp(self):
        # FIX #2: Guard against empty landmark list
        if not hasattr(self, 'lnList') or len(self.lnList) == 0:
            return [0, 0, 0, 0, 0]

        fingers = []

        # FIX #1: Thumb — compare tip (4) against pinky base (17) for
        # handedness-aware detection after horizontal flip.
        # If tip x > pinky-base x the thumb is extended on a right hand.
        if self.lnList[4][1] > self.lnList[17][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers — tip y < pip y means finger is up
        tipIds = [8, 12, 16, 20]
        for id in tipIds:
            if self.lnList[id][2] < self.lnList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lnList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)
                self.lnList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lnList

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lnList[p1][1], self.lnList[p1][2]
        x2, y2 = self.lnList[p2][1], self.lnList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lnList = detector.findPosition(img)
        if len(lnList) != 0:
            print(lnList[2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
