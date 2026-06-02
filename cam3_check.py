import mss
import numpy as np
import cv2
import time

# =========================================================================
# MATCH THESE TO YOUR BOT'S CONFIGURATION
# =========================================================================
MONITOR = {"top": 89, "left": 3, "width": 3000, "height": 2000}
# =========================================================================

print("📸 Diagnostic tool initialized.")
print("👉 Open your Geometry Dash window and make sure it's on screen.")

for i in range(3, 0, -1):
    print(f"Taking snapshot in {i}...")
    time.sleep(1)

with mss.mss() as sct:
    # 1. Capture the exact bounding box defined above
    raw_img = np.array(sct.grab(MONITOR))
    
    # 2. Convert to grayscale (matches the bot's pre-processing)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
    
    # 3. Resize to the AI's native input size
    small = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)

    print("\n✨ Snapshot captured successfully!")
    print("🔴 IMPORTANT: Click on the image windows and press ANY KEY to close them.")

    # Window 1: Shows the full bounding box area
    cv2.imshow("1. Raw Capture Box (Check if game is cut off)", gray)
    
    # Window 2: Shows the microscopic view the AI brain uses to make decisions
    # (Note: OpenCV will stretch this window so you can see it, but it represents 80x80 pixels)
    cv2.imshow("2. AI View (80x80 Pixel Processing)", small)
    
    # Keep windows open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("👋 Diagnostic windows closed.")