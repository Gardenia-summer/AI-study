"""阶段1 第4天：Matplotlib 可视化线性回归的学习率对比。

安装：python -m pip install numpy matplotlib
运行：python day04_loss_visualization.py
输出：脚本所在目录下 outputs_day04 文件夹中的 PNG 图片和 CSV 训练记录。
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# 第 1 步：完全复用第 3 天的数据生成方式，保证实验可比
rng = np.random.default_rng(42)
n = 100
w_true, b_true = 2.0, 1.0
x = rng.uniform(-2.0, 2.0, size=n)
noise = rng.normal(0.0, 0.15, size=n)
y = w_true * x + b_true + noise


def predict(x_data, w, b):
    return w * x_data + b


def mse(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


# 第 2 步：训练函数。losses[0] 是训练前的损失，后续元素是每次更新后的损失
def train(lr, epochs=100):
    w, b = 0.0, 0.0
    losses = [mse(predict(x, w, b), y)]

    for _ in range(epochs):
        y_pred = predict(x, w, b)
        error = y_pred - y
        dw = 2 * np.mean(error * x)
        db = 2 * np.mean(error)
        w -= lr * dw
        b -= lr * db
        losses.append(mse(predict(x, w, b), y))

    return np.array(losses), w, b


# 第 3 步：改变学习率，保持数据、初始参数和轮数相同
learning_rates = [0.005, 0.02, 0.1]
epochs = 100
results = {}
for lr in learning_rates:
    losses, w, b = train(lr, epochs)
    results[lr] = losses
    print(f"lr={lr:<6}  initial_loss={losses[0]:.6f}  "
          f"final_loss={losses[-1]:.6f}  w={w:.4f}  b={b:.4f}")

steps = np.arange(epochs + 1)
out_dir = Path(__file__).resolve().parent / "outputs_day04"
out_dir.mkdir(parents=True, exist_ok=True)

# 第 4 步：同一张图绘制 3 条 Loss 曲线
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
for lr in learning_rates:
    ax.plot(steps, results[lr], linewidth=2, label=f"lr = {lr}")
ax.set_xlabel("Gradient descent steps")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_title("Linear Regression: Learning Rate Comparison")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(out_dir / "loss_curves_3_rates.png", dpi=180)

# 第 5 步（拓展）：极大学习率可能发散；使用对数纵轴看清数量级
loss_bad, w_bad, b_bad = train(1.0, epochs)
print(f"lr=1.0    initial_loss={loss_bad[0]:.6f}  "
      f"final_loss={loss_bad[-1]:.6e}  w={w_bad:.4e}  b={b_bad:.4e}")
fig2, ax2 = plt.subplots(figsize=(9, 5.5), layout="constrained")
ax2.semilogy(steps, results[0.1], linewidth=2, label="lr = 0.1 (converges)")
ax2.semilogy(steps, loss_bad, linewidth=2, label="lr = 1.0 (diverges)")
ax2.set_xlabel("Gradient descent steps")
ax2.set_ylabel("MSE (logarithmic scale)")
ax2.set_title("Convergence vs Divergence")
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.savefig(out_dir / "loss_divergence_log.png", dpi=180)

# 第 6 步（拓展）：可视化训练结束的拟合直线
_, w_fit, b_fit = train(0.1, epochs)
x_line = np.linspace(x.min(), x.max(), 100)
fig3, ax3 = plt.subplots(figsize=(9, 5.5), layout="constrained")
ax3.scatter(x, y, s=18, alpha=0.6, label="Noisy observations")
ax3.plot(x_line, w_true * x_line + b_true, linestyle="--", linewidth=2,
         label="True line")
ax3.plot(x_line, predict(x_line, w_fit, b_fit), linewidth=2,
         label="Fitted line")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("True Relationship vs Learned Linear Regression")
ax3.grid(True, alpha=0.3)
ax3.legend()
fig3.savefig(out_dir / "fitted_line.png", dpi=180)

# 第 7 步：把每次更新的损失记录导出为 CSV，便于以后汇总实验
with (out_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "lr_0.005", "lr_0.02", "lr_0.1", "lr_1.0"])
    for i in range(epochs + 1):
        writer.writerow([i, *(results[lr][i] for lr in learning_rates), loss_bad[i]])

assert len(results[0.1]) == epochs + 1
assert np.isclose(results[0.1][0], 5.5738681, atol=1e-5)
assert abs(w_fit - 2.0) < 0.1 and abs(b_fit - 1.0) < 0.1
assert results[0.1][-1] < results[0.1][0]
assert loss_bad[-1] > loss_bad[0]
print("\n所有图片和 CSV 已保存至：", out_dir)
print("提示：在 VS Code 文件侧边栏中打开 PNG 文件即可查看图像。")
plt.show()
