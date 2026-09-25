import cv2
import numpy as np
import mediapipe as mp
import os
import time
import handtTrackingModule as htm

# --- 1. Load Header Images ---
folderPath = "Virtual board/headerImages"

# FIX #4: Filter out system files (.DS_Store etc.) and sort for consistent order
myList = sorted([f for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

overlayList = []
for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    
    if img is not None:          # FIX #4: skip any file that failed to load
        
        overlayList.append(img)

if not overlayList:
    raise FileNotFoundError(f"No valid images found in '{folderPath}'. Check the path and filenames.")

header = overlayList[0]

# --- 2. Parameters ---
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 80
xp, yp = 0, 0
smooth_factor = 5
pTime = 0  # FIX #10: for FPS counter


# --- 3. Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.9)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Find Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]   # Index fingertip
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle fingertip

        fingers = detector.fingersUp()

        # --- A. Header / Color Selection (index tip in header zone) ---
        if y1 < 125:
            xp, yp = 0, 0  # reset to avoid jump lines when re-entering canvas
            if 150 < x1 < 350:
                header = overlayList[1]
                drawColor = (255, 255, 0)   # blue (BGR)
            elif 400 < x1 < 600:
                header = overlayList[3]
                drawColor = (0, 255, 0)     # green
            elif 650 < x1 < 850:
                header = overlayList[0]
                drawColor = (255, 0, 255)   # pink
            elif 950 < x1 < 1200:
                header = overlayList[2]
                drawColor = (0, 0, 0)       # eraser

        # --- B. Modes ---

        # FIX #3 & #5: Two fingers up = SELECTION ONLY (never draws or erases)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # always reset; no drawing in selection mode
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 255, 255), cv2.FILLED)

        # Eraser Mode — index only, black colour selected
        elif fingers[1] and not fingers[2] and drawColor == (0, 0, 0):
            cv2.circle(img, (x1, y1), eraserThickness // 2, (200, 200, 200), 2)
            if xp != 0 and yp != 0:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            xp, yp = x1, y1

# Drawing Mode — index only, any colour (smoothed)
        elif fingers[1] and not fingers[2]:
            gap = abs(y1 - y2)
            if gap < 30:  # middle finger too close — ambiguous, skip
                xp, yp = 0, 0
            else:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Exponential smoothing
                curr_x = xp + (x1 - xp) / smooth_factor
                curr_y = yp + (y1 - yp) / smooth_factor

                cv2.circle(img, (int(curr_x), int(curr_y)), 15, drawColor, cv2.FILLED)
                cv2.line(imgCanvas, (int(xp), int(yp)), (int(curr_x), int(curr_y)), drawColor, brushThickness)

                xp, yp = curr_x, curr_y

        else:
            xp, yp = 0, 0  # no recognised gesture → reset

        # --- FIX #9: On-screen mode label ---
        if drawColor == (0, 0, 0):
            mode_label = "Mode: Eraser"
            label_color = (180, 180, 180)
        elif fingers[1] and fingers[2]:
            mode_label = "Mode: Select"
            label_color = (255, 255, 255)
        elif fingers[1] and not fingers[2]:
            mode_label = "Mode: Draw"
            label_color = drawColor
        else:
            mode_label = ""
            label_color = (255, 255, 255)

        if mode_label:
            cv2.putText(img, mode_label, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

    # --- Merge Canvas onto Camera Feed ---
    img[0:125, 0:1280] = header
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # --- FIX #10: FPS Counter ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Virtual Writing Board", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # FIX #8: Press 'c' to clear the canvas 
    elif key == ord('c'):
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
