# 教学组织参考

## dive-into-llms

阅读版本：[`f84c04268794ef94f8949808bbc14ab8636763a0`](https://github.com/Lordog/dive-into-llms/tree/f84c04268794ef94f8949808bbc14ab8636763a0)。阅读了下列三个 Notebook 的完整正文与内嵌代码。

| 教程 | 可借用的组织办法 | 在本课程中的用法 |
| --- | --- | --- |
| [微调与部署：dive-tuning.ipynb](https://github.com/Lordog/dive-into-llms/blob/f84c04268794ef94f8949808bbc14ab8636763a0/documents/chapter1/dive-tuning.ipynb) | 用同一个分类任务串联数据、模型、训练与输出，各个模块的作用有具体落点 | 时序课用同一条滤波轨迹连接状态递推、BPTT、训练与预测曲线 |
| [提示学习：dive-prompting.ipynb](https://github.com/Lordog/dive-into-llms/blob/f84c04268794ef94f8949808bbc14ab8636763a0/documents/chapter2/dive-prompting.ipynb) | 固定任务与输入，改变示例或采样条件，再比较具体输出 | 保留一组数值，改变一个权重、门值或时间间隔，先预测影响，再核对中间量 |
| [知识编辑：dive_edit.ipynb](https://github.com/Lordog/dive-into-llms/blob/f84c04268794ef94f8949808bbc14ab8636763a0/documents/chapter3/dive_edit.ipynb) | 从一条明确的修改目标，接到操作、修改前后输出与邻域评价 | 每个实验先给出要解释的变化；图后将变化接回公式，并安排能区分原因的对照 |

## 具体写法

先让读者遇到需要计算的问题，再引入解决它的量。一个标量或一行算清后，再扩展到矩阵。新符号出现时说明对象、形状与操作；推导中的每个乘数都能指回一条计算路径。

图前说明要对照的数值，图后说明可见变化来自哪一项。例如，规则环上的局部平均逐步压低节点间差异，同时保留全图通道均值；LTC 中同一输入持续更久，状态有更多时间响应。末尾实验只改变一个相关因素，并明确要记录的量。

这些组织方法用于本课程已有的数值、合成数据与推导。上游正文、代码和图片不收入本课程。
