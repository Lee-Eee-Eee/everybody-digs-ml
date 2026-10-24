# 参考地图

## 本地 NTU GenAI/ML 作业

课程设计已读取 `../../NTU_HYLee/GenAI_ML/hw/` 的十份学习本。

| 原材料 | 本课程的连接位置 | 先完成的基础 |
| --- | --- | --- |
| HW01 tokens、prompt、多轮对话 | 12–15 | 词表、序列轴、下一 token 概率 |
| HW02 RAG | 完成 03、12–15 后的后续项目 | 向量相似度、检索与生成的接口 |
| HW03 Logit Lens / Function Vector | 15 | Block 输出、hook、隐藏状态投影与干预 |
| HW04 指令防御 | 完成生成模型后的扩展 | 区分输入文本、模型行为与评价数据 |
| HW05 Regression | 06–07 | Dataset、标准化、MLP、训练/验证划分 |
| HW06 CNN / augmentation / fine-tuning | 08–09 | 卷积、NCHW、分类训练、残差 |
| HW07 SFT / DPO | 后续进阶 | 先读懂 14–15 的训练目标与数据流 |
| HW08 遗忘 | 18 的对照实验思路 | 固定评价任务，观察前后变化 |
| HW09 Diffusion | 后续生成模型分支 | 概率、噪声回归、CNN 与训练循环 |
| HW10 Speech | 后续序列模型分支 | token、时序表示、重建与生成 |

## minGPT

- 项目：[karpathy/minGPT](https://github.com/karpathy/minGPT)
- 本地固定版本：`37baab71b9abea1b76ab957409a1cc2fbfba8a26`
- [模型文件](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py)：CausalSelfAttention → Block → GPT.forward → generate。
- [训练器](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/trainer.py)：取 batch、前向、梯度清零、反传、裁剪、更新。
- 第 13 课逐头实现便于看清每一头；第 15 课用带编号的张量解释原实现的合并投影与换轴。

本地副本保留上游 MIT 许可证。课程中的微型 decoder 代码与讲解单独保存在 `models/tiny_gpt.py` 和 12–15 课。

## 主要一手资料

| 材料 | 阅读任务 |
| --- | --- |
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | 将 tensor、autograd、module、DataLoader 对应到 05–07 的代码 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 从式中的矩阵乘法、缩放与归一化重建 12–13 课 |
| [GCN](https://arxiv.org/abs/1609.02907) | 从自环、度归一化读到节点特征更新与实验划分 |
| [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) | 找出动力学方程、有效时间常数与数值更新 |
| [作者 ncps 实现](https://github.com/mlech26l/ncps) | 对照 LTC 单元的参数约束、分子与分母；进一步阅读 CfC |
| [Neural ODE](https://arxiv.org/abs/1806.07366) | 比较离散残差与连续动力学，区分模型和数值求解器 |

每课在符号与代码旁给出必要解释；末尾的问题用于把已经做过的小实验连接到论文。
