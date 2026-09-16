# Transformer 学习与复现

本项目参考论文 **Attention Is All You Need** 和 **The Annotated Transformer**，
使用 PyTorch 实现 Transformer，并完成简单的 Copy Task 训练与推理。

## 环境配置

本项目开发环境：

- Windows
- Conda
- Python 3.11
- PyTorch

创建并进入 Conda 环境：

```bash
conda create -n transformer python=3.11
conda activate transformer
## 项目结构

```text
transformer/
├── model.py          # Transformer 模型结构
├── train.py          # 模型训练
├── inference.py      # 模型推理
├── test_model.py     # 模型测试
├── README.md         # 项目说明
└── transformer_copy_model.pt   # 训练后的模型参数
```

## 主要实现内容

项目主要实现了：

- Encoder 和 Decoder
- Multi-Head Attention
- Feed Forward Network
- Positional Encoding
- Padding Mask 和 Subsequent Mask
- Label Smoothing
- Transformer 训练
- Greedy Decoding 推理

## 训练

运行：

```bash
python train.py
```

训练使用简单的 Copy Task，即让 Transformer 学习：

```text
输入：[1, 3, 4, 5, 2]
输出：[1, 3, 4, 5, 2]
```

训练完成后保存模型：

```text
transformer_copy_model.pt
```

## 推理

运行：

```bash
python inference.py
```

测试结果：

```text
Generated sequence:
tensor([[1, 3, 4, 5, 2]])

Expected sequence:
tensor([[1, 3, 4, 5, 2]])

Inference SUCCESS!
```

说明训练后的 Transformer 已经能够完成当前 Copy Task。

## 参考资料

- Attention Is All You Need
- The Annotated Transformer