import cv2
import mss
import numpy as np
import time

print("Taking a picture of the bot's view in 15 seconds... Switch to Geometry Dash!")
time.sleep(15)

with mss.mss() as sct:
    monitor_info = sct.monitors[1]
    screen_w = monitor_info["width"]
    screen_h = monitor_info["height"]

    # This matches the box your bot is currently watching
    monitor = {
        "left": int(screen_w * 0.60), 
        "top": int(screen_h * 0.65),
        "width": int(screen_w * 0.05), 
        "height": int(screen_h * 0.08)
    }

    # Grab the screenshot and save it as an image file
    img = np.array(sct.grab(monitor))
    cv2.imwrite("bot_view.png", img)
    print("Picture saved as 'bot_view.png' in your folder!")