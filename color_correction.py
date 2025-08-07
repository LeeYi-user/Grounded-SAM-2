import cv2
import numpy as np
from matplotlib import pyplot as plt

# 讀取圖像
image = cv2.imread('vlcsnap-2025-08-07-08h23m25s110.png')

# 將圖像從BGR轉為RGB顯示用
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: 白平衡（簡單灰世界算法）
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

wb_image = white_balance(image)

# Step 2: CLAHE 增強對比
lab = cv2.cvtColor(wb_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
contrast_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Step 3: 銳化
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharpened = cv2.filter2D(contrast_image, -1, kernel_sharpening)

# 顯示結果（用 matplotlib）
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Enhanced Image')
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 可選：儲存結果
cv2.imwrite('enhanced_shrimp_detection.png', sharpened)
