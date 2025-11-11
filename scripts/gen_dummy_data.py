# Only English comments and print messages.
import os, cv2, numpy as np

VIDEO_ID = "test1"
mask_dir = f"data/interim/masks/{VIDEO_ID}"
frame_dir = f"data/interim/frames/{VIDEO_ID}"
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(frame_dir, exist_ok=True)

H, W, N = 240, 320, 12
for i in range(N):
    # frame: moving colored rectangle
    img = np.zeros((H, W, 3), np.uint8)
    x = 10 + i*5
    cv2.rectangle(img, (x, 60), (x+80, 160), (0, 180, 255), -1)
    cv2.imwrite(f"{frame_dir}/{i:06d}.png", img)

    # alpha: centered square (float in [0,1] saved as 8-bit)
    a = np.zeros((H, W), np.float32)
    cv2.rectangle(a, (90, 80), (230, 200), 1.0, -1)
    cv2.imwrite(f"{mask_dir}/{i:06d}.png", (a*255).astype(np.uint8))

print("Dummy data ready.")
