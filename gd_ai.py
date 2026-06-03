import os
import cv2
import mss
import numpy as np
import pydirectinput
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

pydirectinput.PAUSE = 0.0 

# Data Analysis
class AIDebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(AIDebugCallback, self).__init__(verbose)
        self.analysis_count = 0

    def _on_rollout_end(self) -> None:
        # Triggers when the AI stops playing to update its neural network
        self.analysis_count += 1
        print("\n" + "="*50)
        print(f"🧠 [PHASE {self.analysis_count}] DATA ANALYSIS IN PROGRESS")
        print(f"The bot is pausing to review the last {self.model.n_steps} frames and update its brain.")
        print("="*50 + "\n")

    def _on_step(self) -> bool:
        return True


# Learning Rate
def linear_schedule(initial_value: float):
    """
    Drops the learning rate linearly from initial_value down to 0 over the training process.
    Will to Experiment Simulation
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# Environment
class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        self.last_prog = 0.0
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8)
        self.sct = mss.mss()
        self.monitor = {"top": 89, "left": 3, "width": 3000, "height": 2000}
        
        template_path = "stereo-death.png"
        if not os.path.exists(template_path):
            print(f"⚠️ WARNING: '{template_path}' not found!")
            self.template = None
        else:
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
        # Specific rewards
        self.death_penalty = -10.0
        #self.reward_per_second = 0.2
        #self.last_time = time.time()
        # In step():o

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Release the spacebar just in case it got stuck down
        pydirectinput.keyUp('space')
        self.last_prog = 0.0
        
        # Fix Double Deaths
        # Text Loop
        if hasattr(self, 'template') and self.template is not None:
            while True:
                raw_gray_frame = self._get_screenshot(return_both=True)[0]
                res = cv2.matchTemplate(raw_gray_frame, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val < 0.60: # The text is gone
                    break
                time.sleep(0.1) # Check again - 100ms
        
        
        time.sleep(0.5) 
        
        print("Respawn")
        #self.last_time = time.time() # Reset stopwatch
        
        return self._get_screenshot(), {}

    def step(self, action):
        if action == 1:
            pydirectinput.press('space')
        
        raw_gray_frame, obs = self._get_screenshot(return_both=True)
        terminated = False
        
        # Calculate how much time passed since the last frame
        #current_time = time.time()
        #delta_time = current_time - self.last_time
        #self.last_time = current_time
        
        # Cap delta_time just in case the computer lags -> NOT GREAT RN
        #delta_time = min(delta_time, 0.1) 
        
        # Strict Reward
        # reward = self.reward_per_second * delta_time 
        curr_prog = self._read_progress_bar(raw_gray_frame)
        prog_delta = curr_prog - self.last_prog
        reward = prog_delta * 100
        self.last_prog = curr_prog
        
        # Detect Death
        is_dead = False
        death_reason = ""
        
        # Check Screen
        if hasattr(self, 'template') and self.template is not None:
            res = cv2.matchTemplate(raw_gray_frame, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.80:
                is_dead = True
                death_reason = f"Screen Match ({max_val:.2f})"
        
        # Check Black/Blank
        screen_brightness = np.mean(raw_gray_frame)
        if screen_brightness < 15.0: 
            is_dead = True
            death_reason = f"Fade to Black (Brightness: {screen_brightness:.1f})"
            
        # Apply penalty
        if is_dead:
            reward = self.death_penalty
            terminated = True
            print(f"Death counted due to {death_reason} | Applied penalty: {self.death_penalty} pts")
            pydirectinput.press('space')
                
        return obs, reward, terminated, False, {}

    def _get_screenshot(self, return_both=False):
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        small = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
        obs = np.expand_dims(small, axis=-1)
        
        if return_both:
            return gray, obs
        return obs
    
    def _read_progress_bar(self, raw_gray_frame):
        bar_top, bar_bottom = 50, 70
        bar_left, bar_right = 1200, 1800
        bar_strip = raw_gray_frame[bar_top:bar_bottom, bar_left:bar_right]
        filled_mask = bar_strip > 200
        col_fill = filled_mask.mean(axis=0)
        filled_cols = np.sum(col_fill > 0.5)
        total_cols = bar_strip.shape[1]
        return filled_cols / total_cols

# Training Loop
if __name__ == "__main__":
    print("  GD AI Initializing  ")
    
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
        
    base_env = GeometryDashEnv()
    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=4) 
    
    save_filename = "gd_stereo_madness_model"
    
    if os.path.exists(f"{save_filename}.zip"):
        print(f"💾 Loading saved data from {save_filename}.zip.")
        model = PPO.load(save_filename, env=env)
    else:
        print("Creating new policy.")
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1, 
            # linear_schedule drops value
            learning_rate=linear_schedule(0.0003), 
            n_steps=256,  # Analyzes data every 256 frames (~30-40 seconds of gameplay)
            batch_size=64, 
            ent_coef=0.015  # Slightly higher starting entropy for strong early experimentation
        )
    
    print("Bot is live")
    
    # Instantiate tracker
    debug_callback = AIDebugCallback()
    
    try:
        # Changed to 500,000 timesteps to give it more room to learn
        model.learn(total_timesteps=500000, callback=debug_callback)
        model.save(save_filename)
        print("Progress saved successfully.")
    except KeyboardInterrupt:
        print("\nTraining safely interrupted by user. Saving model state.")
        model.save(save_filename)
        print("Saved")