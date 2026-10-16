# AI Research Study

本仓库用于记录人工智能与大语言模型的学习、
代码实验、论文复现和科研探索。

## 开发环境

- Windows 11 + WSL2 Ubuntu
- Python 3.14
- VS Code
- PyTorch

## 学习目标

1. 掌握 PyTorch 和深度学习基础
2. 理解 Transformer 与大语言模型
3. 完成经典论文复现
4. 尝试独立科研实验

## 仓库定位

`ai-study` 是我的人工智能与大语言模型（LLM）科研学习仓库，用于持续积累学习笔记、代码实验、论文阅读与复现，以及科研探索记录。

## 当前文件

- `debug_test.py`：简单运算与调试练习。
- `demo.ipynb`：交互式 Python 实验 Notebook。
- `environment_test.py`、`test.py`：Python 环境与解释器检查。
- `torch_test.py`：PyTorch 张量运算与 CUDA 可用性检查。

后续将在已有实验基础上逐步补充深度学习、Transformer、大语言模型训练与推理等学习内容。

## 文件管理约定

- 提交学习笔记、实验源码和适合分享的 Notebook；分享前检查 Notebook 输出是否包含敏感信息。
- `.venv/`、`.idea/`、缓存、环境变量文件和凭据仅保存在本地，由 `.gitignore` 排除。
- 数据集放在 `data/` 或 `datasets/`，模型与权重放在 `models/`、`weights/` 或 `checkpoints/`；这些目录及常见模型、二进制数据文件不上传到 GitHub。
- 新增大文件前检查 `git status`，并在笔记中记录数据集、模型的来源与获取方式。
