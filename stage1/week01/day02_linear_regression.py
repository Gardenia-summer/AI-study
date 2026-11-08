"""阶段1 第2天：手写线性回归、MSE 梯度和学习率实验。
运行：python day02_linear_regression.py
依赖：numpy
"""
import numpy as np


x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])


def predict(x_data, w, b):
    return w * x_data + b


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def gradients(x_data, y_true, w, b):
    error = predict(x_data, w, b) - y_true
    dw = 2 * np.mean(error * x_data)
    db = 2 * np.mean(error)
    return float(dw), float(db)


def loss_at(w, b):
    return float(mse(predict(x, w, b), y))


def train(learning_rate, steps):
    w, b = 0.0, 0.0
    records = [(0, w, b, loss_at(w, b))]
    for step in range(1, steps + 1):
        dw, db = gradients(x, y, w, b)
        next_w = w - learning_rate * dw
        next_b = b - learning_rate * db
        w, b = next_w, next_b
        records.append((step, w, b, loss_at(w, b)))
    return records


def main():
    w, b = 0.0, 0.0
    y_pred = predict(x, w, b)
    dw, db = gradients(x, y, w, b)
    print("=== 1. 手算核对 ===")
    print("x =", x, "y =", y)
    print("初始预测 =", y_pred)
    print(f"初始 MSE = {mse(y_pred, y):.6f} (应为 56/3)")
    print(f"dL/dw = {dw:.6f} (应为 -56/3)")
    print(f"dL/db = {db:.6f} (应为 -8)")
    lr = 0.01
    w_new = w - lr * dw
    b_new = b - lr * db
    print(f"学习率 {lr}，更新后 w = {w_new:.6f}, b = {b_new:.6f}")
    print(f"更新后 MSE = {loss_at(w_new, b_new):.6f}")

    eps = 1e-6
    numeric_dw = (loss_at(w + eps, b) - loss_at(w - eps, b)) / (2 * eps)
    numeric_db = (loss_at(w, b + eps) - loss_at(w, b - eps)) / (2 * eps)
    print("\n=== 2. 数值差分验证手推梯度 ===")
    print(f"数值 dL/dw = {numeric_dw:.6f}")
    print(f"数值 dL/db = {numeric_db:.6f}")
    assert np.allclose([dw, db], [numeric_dw, numeric_db], atol=1e-6)
    print("核对通过：解析梯度与数值近似梯度一致")

    print("\n=== 3. 比较学习率（同样训练 30 次） ===")
    for learning_rate in [0.001, 0.01, 0.1, 0.3]:
        history = train(learning_rate, 30)
        print(f"学习率 η = {learning_rate}")
        for step in [0, 1, 2, 5, 10, 30]:
            _, w_step, b_step, loss_step = history[step]
            print(f"  第{step:>2}步: w={w_step:>12.6f} b={b_step:>12.6f} MSE={loss_step:.6g}")
    print("\n思考：哪种学习率下降太慢？哪一种开始发散？")
    print("目标参数是 w=2, b=0，但有限步的拟合结果不一定完全等于目标值。")


if __name__ == "__main__":
    main()
