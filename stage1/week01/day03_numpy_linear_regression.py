"""阶段1 第3天：只用 NumPy 从零训练一元线性回归。

运行：python stage1/week01/day03_numpy_linear_regression.py
依赖：numpy
"""
import numpy as np


# 1. 构造可复现的带噪声数据：y = 2x + 1 + noise
rng = np.random.default_rng(42)
n = 100
w_true = 2.0
b_true = 1.0
x = rng.uniform(-2.0, 2.0, size=n)
noise = rng.normal(loc=0.0, scale=0.15, size=n)
y = w_true * x + b_true + noise

print("数据形状：x =", x.shape, ", y =", y.shape)
print(f"真实参数：w = {w_true:.4f}, b = {b_true:.4f}")


# 2. 模型预测和 MSE：与第2天的公式一致
def predict(x_data, weight, bias):
    return weight * x_data + bias


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# 3. 手工求梯度并更新参数：批量梯度下降
def train(x_data, y_data, lr=0.1, epochs=100, verbose=True):
    weight = 0.0
    bias = 0.0
    history = [(0, float(mse(predict(x_data, weight, bias), y_data)), weight, bias)]
    checkpoints = {1, 2, 5, 10, 20, 50, 100}

    if verbose:
        print("\n训练记录（每一行都是完成相应次数更新后的状态）：")
        print("step       loss             w           b")
        print(f"{0:4d}  {history[-1][1]:14.7f}  {weight:11.6f}  {bias:10.6f}")

    for step in range(1, epochs + 1):
        y_pred = predict(x_data, weight, bias)
        error = y_pred - y_data
        dw = 2 * np.mean(error * x_data)
        db = 2 * np.mean(error)

        weight -= lr * dw
        bias -= lr * db

        current_loss = float(mse(predict(x_data, weight, bias), y_data))
        history.append((step, current_loss, weight, bias))
        if verbose and step in checkpoints:
            print(f"{step:4d}  {current_loss:14.7f}  {weight:11.6f}  {bias:10.6f}")

    return weight, bias, history


w, b, history = train(x, y, lr=0.1, epochs=100)
print(f"\n训练参数：w = {w:.6f}, b = {b:.6f}")
print(f"真实参数：w = {w_true:.6f}, b = {b_true:.6f}")
print(f"参数误差：|Δw| = {abs(w-w_true):.6f}, |Δb| = {abs(b-b_true):.6f}")
print(f"训练 MSE：{history[-1][1]:.7f}")
print(f"当 x = 1.5，预测值 = {predict(1.5, w, b):.6f}")


# 4. 比较学习率：其他条件完全相同（尤其要保持同一份数据）
print("\n不同学习率的比较（各训练100次）：")
for eta in [0.005, 0.02, 0.1, 1.0]:
    w_eta, b_eta, hist_eta = train(x, y, lr=eta, epochs=100, verbose=False)
    print(f"lr = {eta:5.3f} | final_loss = {hist_eta[-1][1]:.7e} | w = {w_eta:.5f} | b = {b_eta:.5f}")

assert abs(w - w_true) < 0.1
assert abs(b - b_true) < 0.1
assert history[-1][1] < history[0][1]
print("\n自检通过：参数接近真实值，训练损失下降。")
