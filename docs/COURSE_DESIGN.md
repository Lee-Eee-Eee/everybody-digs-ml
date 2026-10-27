# 课程设计与参考｜Course design and references

[课程首页](../README.md) · [English guide](../README.en.md) · [项目任务](PROJECTS.md)

本课程围绕「看懂一个计算 → 自己写出它 → 通过实验解释结果」组织。各节的公式、代码、图和练习服务于同一个具体问题；末尾阅读材料帮助你把小实验连接到原论文或开源实现。

The course connects a concrete calculation, an implementation, and an experiment. References offer context and further reading, while each lesson develops its own examples and exercises.

## 四条设计线索 / Four design principles

| 线索 / Principle | 在本课程中的体现 / In this course |
| --- | --- |
| 从数值到概念 / From numbers to concepts | 先计算一个小例子，再讨论形状、公式与实现。例如手算一个卷积窗口，再写出共享核。 |
| 从基线到模型 / From baseline to model | 手写数字问题贯穿距离、线性分类与神经网络；比较条件时记录数据划分与指标。 |
| 从训练到评价 / From training to evaluation | 将预处理、验证选择和留出测试放在完整实验中，记录错例与多次运行。 |
| 从代码到研究 / From code to research | 先读本课自己组装的模块，再对照 minGPT 等源码或原论文，追问设计选择。 |

## 数据与任务 / Data and tasks

- **`digits`：** scikit-learn 自带的 8×8 手写数字数据，用于离线观察像素、训练分类器和快速比较。它与 28×28 的 MNIST 是不同的数据集。
- **MNIST：** 可作为扩展数据集，通过 OpenML 获取；首次下载需要联网和更多时间。比较时应分别记录数据来源、样本划分、预处理和评价设置。
- **文本：** 微型 GPT 使用课程短语料，预测下一个字符；训练结果用于理解流程和张量，不代表通用语言模型的能力。
- **合成实验：** 回归、时序、图、液体网络和扩散入门使用可控的小例子，以便观察变化与验证计算。

## 参考与本课程的调整 / References and adaptations

| 参考 / Reference | 启发 / Inspiration | 本课程的具体安排 / Our approach |
| --- | --- | --- |
| [Build Your Own X: Machine Learning](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning) | 通过具体算法或模型动手学习。 | 用同一手写数字问题连接算法、数据划分、评价和错例；将不同任务所需的回归数据单独说明。 |
| [Dive into Deep Learning](https://github.com/d2l-ai/d2l-zh) | 将文字、公式和可运行代码放在一起。 | 将推导、代码和可检查练习分成小步骤；从一个数或一条计算路径出发。 |
| [micrograd](https://github.com/karpathy/micrograd) | 通过小计算图理解链式法则。 | 在 NumPy 手写反传后，用有限差分与 PyTorch autograd 对照。 |
| [minGPT](https://github.com/karpathy/minGPT) | 阅读紧凑的 GPT 实现。 | 先在课程中构造 Q/K/V 和 decoder，再沿本地快照追踪原实现的张量和训练器；第三方许可证保留在原目录。 |
| [Happy-LLM](https://github.com/datawhalechina/happy-llm) | 可作为系统性大模型学习的后续阅读。 | 本课程聚焦小型可运行实例和计算路径；更广泛的预训练、微调与应用可在此之后继续探索。 |
| [DDPM](https://arxiv.org/abs/2006.11239) | 前向扩散、预测噪声和反向生成。 | 先可视化手写数字加噪，再用一维可训练模型展示完整训练和反向采样。 |

课程文件、更多论文与出处见[参考地图](../references/SOURCES.md)；包含的第三方代码与许可证见[第三方说明](../THIRD_PARTY.md)。
