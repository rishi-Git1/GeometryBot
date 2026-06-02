import cv2
import mss
import numpy as np

# Capture the entire screen to find the bar
with mss.mss() as sct:
    img = np.array(sct.grab(sct.monitors[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# This function prints coordinates when you click
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: Left={x}, Top={y}")
        # Draw a little circle so you know where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Map your screen', img)

cv2.imshow('Map your screen', img)
cv2.setMouseCallback('Map your screen', click_event)

print("Click on the top-left and bottom-right of the progress bar.")
cv2.waitKey(0)
cv2.destroyAllWindows()