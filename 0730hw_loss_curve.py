import pandas as pd
import matplotlib.pyplot as plt

# 讀取 results.csv 文件
results_path = 'runs/detect/e223_b64/results.csv'
df = pd.read_csv(results_path)

# 移除列名中的多餘空格
df.columns = df.columns.str.strip()

# 繪製 Loss 曲線
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Validation Class Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLOv8 Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
