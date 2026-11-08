"""阶段 1 · 第 1 天：Python 与 NumPy 基础练习。
运行：python day01_numpy.py
"""

import numpy as np


def main():
    print("=== 1. ndarray、shape、reshape ===")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print("a =\n", a)
    print("a.shape =", a.shape)
    print("a.ndim =", a.ndim)

    numbers = np.arange(12)
    matrix = numbers.reshape(3, 4)
    print("arange(12).reshape(3, 4) =\n", matrix)

    print("\n=== 2. 切片 ===")
    print("第一行：", matrix[0, :])
    print("第二列：", matrix[:, 1])
    print("前两行：\n", matrix[:2, :])

    print("\n=== 3. 实践任务：创建 (100, 2) 矩阵并进行矩阵乘法 ===")
    x = np.arange(200).reshape(100, 2)
    w = np.array([[1, 2, 3], [4, 5, 6]])
    y = x @ w
    print("x.shape =", x.shape)
    print("w.shape =", w.shape)
    print("y.shape =", y.shape)
    print("x 的前三行：\n", x[:3])
    print("y 的前三行：\n", y[:3])

    print("\n=== 4. 广播 ===")
    offset = np.array([10, 20])
    shifted = x + offset
    print("offset.shape =", offset.shape)
    print("x + offset 的前三行：\n", shifted[:3])

    print("\n=== 5. sum 与 mean ===")
    print("x.sum(axis=0) =", x.sum(axis=0))
    print("x.mean(axis=0) =", x.mean(axis=0))
    print("x.mean(axis=1) 的前三项 =", x.mean(axis=1)[:3])

    print("\n=== 6. 连接机器学习：线性层的计算 ===")
    b = np.array([1, 0, -1])
    z = x @ w + b
    print("b.shape =", b.shape)
    print("z.shape =", z.shape)
    print("z 的前三行：\n", z[:3])


if __name__ == "__main__":
    main()
