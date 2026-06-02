import cv2
import mss
import numpy as np
import pydirectinput
import time

print("Starting Predictive Engine in 5 seconds...")
time.sleep(5)

with mss.mss() as sct:
    monitor_info = sct.monitors[1]
    screen_w = monitor_info["width"]
    screen_h = monitor_info["height"]

# A generous vision box to see the entire upcoming obstacle group
monitor = {
    "left": int(screen_w * 0.54), 
    "top": int(screen_h * 0.56),  
    "width": int(screen_w * 0.30), 
    "height": int(screen_h * 0.14)
}

CUBE_SPEED = 10.3  # Approximate blocks per second (can be calibrated)
JUMP_DURATION = 0.48
is_mid_air = False
last_jump_time = 0

with mss.mss() as sct:
    while True:
        current_time = time.time()

        if is_mid_air:
            if current_time - last_jump_time >= JUMP_DURATION:
                is_mid_air = False
            else:
                continue

        # Capture and isolate outlines
        img = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        edges = cv2.Canny(gray, 150, 250)

        # Use contours to find distinct object shapes instead of raw pixels
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out tiny pixel noise
            if w > 5 and h > 5:
                obstacles.append((x, x + w, y)) # Store (Start_X, End_X, Height)

        if obstacles:
            # Sort obstacles from closest to furthest
            obstacles.sort(key=lambda item: item[0])
            
            # PATH PLANNING LOGIC:
            # Look at the closest obstacle group
            first_obstacle = obstacles[0]
            obs_start = first_obstacle[0]
            
            # Check if there's a second spike tightly packed right behind it (Double Spike Detector)
            total_hazard_width = first_obstacle[1] - first_obstacle[0]
            if len(obstacles) > 1:
                next_obstacle = obstacles[1]
                # If the second obstacle is less than 40 pixels away from the first, it's a multi-spike!
                if next_obstacle[0] - first_obstacle[1] < 40:
                    total_hazard_width = next_obstacle[1] - first_obstacle[0]
                    print(f"⚠️ MULTI-SPIKE DETECTED! Total width: {total_hazard_width}px")

            # --- PREDICTIVE SURVIVAL CALCULATION ---
            # If it's a wide multi-spike, we MUST wait until it's closer to jump,
            # otherwise our arc will land right on the second spike.
            if total_hazard_width > 35:
                # Target threshold for a double spike (Jump later)
                jump_threshold = 25 
            else:
                # Target threshold for a single spike/step (Jump earlier)
                jump_threshold = 55 

            # Trigger based on the intelligent path prediction
            if obs_start < jump_threshold:
                print(f"--- PATH VERIFIED SAFE: EXECUTE JUMP AT X={obs_start} ---")
                pydirectinput.press('space')
                last_jump_time = time.time()
                is_mid_air = True