"""Course entry point and a complete small research protocol."""
from coursekit import Lesson


def build():
    result=[]
    l=Lesson('00_start_and_environment','00 从这里开始：运行、预测、修改、解释',
             ['确认当前解释器属于本课程','做一个可检查的小练习','找到基础课、模型课与研究实验的学习路线'])
    l.md('''
## 一张数字图片，一段尚未写完的话

电脑看到一张手写数字时，收到的是一张灰度数值表。让它接着写 `the ` 时，收到的是几个字符编号。它怎样从这些数里学出一个判断，又怎样知道该改动哪一条连接？

这套课从能亲手改动的小程序出发。前几课把数字组织成列表与数组，第04课从一个小网络算出全部梯度，第09课训练手写数字分类器，第14课让自己的字符模型续写文本。一路上遇到的新工具，都用来解决当时具体遇到的困难。

也可以先打开 [交互学习室](../../interactive/index.html#network)：点一条连接，改变它的权重，看输出怎样变化。第04课会把这个过程完整写成 Python。这里从最短的一行代码开始。
''')
    l.md('## 1 先运行一段最小程序\n\n先点击下面代码格，再运行。`print` 将括号中的结果显示出来。读输出之前，先猜这两个数分别是什么。')
    l.code('print(2+3)\nprint(2*3)')
    l.ex('start01','自己改一个计算','让 Python 算出 3×4+2，把结果保存为 answer。',
         'answer=None', '先乘法，再加法。', 'answer=3*4+2', "check('计算结果',answer,14)")
    l.md('## 2 找到课程自己的 Python\n\nPowerShell 中先进入课程目录，再用 `uv run`。uv 根据本目录的 `pyproject.toml` 与 `uv.lock` 使用 `.venv`。编辑器的 notebook 内核也选择这份 `.venv/Scripts/python.exe`。')
    l.code(r'''
import sys
print('课程目录：',ROOT)
print('当前解释器：',sys.executable)
print('Python版本：',sys.version.split()[0])
''')
    l.md(r'''
在 PowerShell 中：

```powershell
# 在下载的课程文件夹中运行
uv sync --locked
uv run python -c "import sys; print(sys.executable)"
uv run jupyter lab
```

若习惯先激活环境，可运行 `.\.venv\Scripts\Activate.ps1`。`uv run` 可以直接指定课程环境运行程序。
''')
    l.code(r'''
from importlib.metadata import version
for package in ['numpy','torch','matplotlib']:
    print(package,version(package))
''')
    l.md('## 3 把代码格变成自己的草稿纸\n\n学习本里的代码格可以单独修改和运行。输入保存为变量之后，后面的代码还可以继续使用它；重新启动内核，就相当于换了一张空白草稿纸。\n\n练习格中的 `None` 留给自己填写。相邻一格会比较结果；思路卡住时，可以展开提示。下面这题只需要从三个数字中取出一个，后面取样本、挑一段文本也会用到同样的下标。')
    l.ex('start02','区分位置与数值','列表 `[9,2,7]` 的第 1 个位置存着什么？下标从零开始。',
         'start_values=[9,2,7]\nstart_value=None', '位置0是9，位置1是下一项。',
         'start_value=start_values[1]', "check('位置1的值',start_value,2)")
    l.md('''
## 4 学习路线

| 阶段 | 课号 | 这一段的独立成果 |
| --- | --- | --- |
| Python 与数组 | 01–03 | 把数字表写成函数，读写文件，解释每个数组轴 |
| 导数与学习 | 04–07 | 手写反传，核对 autograd，训练与评价 MLP |
| 传统机器学习 | 20–21 | 用可复现实验理解距离、回归、分类与聚类 |
| 图像 | 08–09、23 | 手算卷积，训练 CNN 和小型残差网络 |
| 时间与记忆 | 10–11 | 展开 RNN，计算 LSTM 各门与状态 |
| 注意力与生成 | 12–15 | 从 QKV 到训练微型 GPT，再读 minGPT |
| 图与连续时间 | 16–17 | 手写 GCN 与液体时间常数单元 |
| 扩散模型 | 22 | 从噪声与去噪理解生成过程 |
| 小研究 | 18 | 写问题、设对照、固定划分、报告多次实验 |

01–07 连成一条基础线；20–21 连接传统机器学习方法。之后可以按图像、序列、图、连续动力学或扩散模型分支推进。学习本按主题放在 `lessons/` 的子目录中，每课的变量在课内逐步建立，从头运行便能恢复这个过程。

交互学习室穿插在04、05、07、08、09、13课中：看前向数值怎样变，逐步跟着反传，再回到 notebook 写出相同计算。实验室使用两个输出的小网络，04课使用一个输出的小网络，权重与节点数各有对应；沿边相乘、在汇合处相加的规则贯穿两者。
''')
    l.md('''
## 5 从自己的小实验读到论文

当一个词只有名字时，论文很难读。亲手算过卷积的一个窗口，再看“权重共享”，它就对应到同一组乘数反复出现在不同位置；训练过字符模型，再看“自监督”，它就对应到原文错开一位形成的标签。

各课末尾会带着刚做过的计算，阅读一段原论文或官方源码。第18课再把实验连起来：比较两个模型，观察不同初始化，解释训练区间外的失败。这些经历会逐渐帮助我们分清，一个新方法改的是模型表示、学习目标，还是更新参数的方式。
''')
    l.ex('start03','先保存再检查','将 5 的平方保存在 result 中，运行后面的检查。',
         'result=None','幂运算写 **。','result=5**2',"check('平方',result,25)")
    l.md('下一节打开 `01_python_values_control.ipynb`。从一份测量数字表开始，学习变量、容器、条件和循环。')
    result.append(l)

    l=Lesson('18_research_experiments','18 从会训练模型，到做一项能解释的小研究',
             ['把问题写成可比较的实验','按训练/验证/测试安排模型选择','保存配置、结果和失败样例并解释证据'], '07；完成任一模型分支后可回到本课')
    l.code(r'''
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
torch.set_num_threads(1)
style()
''')
    l.md(r'''
## 1 曲线贴住了样本，往外延长会怎样

一个网络可以把训练点拟合得很好。现在将横轴继续向右延伸：它会顺着真实曲线走，还是沿着自己的方向离开？这个小实验会把训练、模型选择与外推放到同一张图里。

真实规律取 $y=\sin(3x)$。网络只见到 [-1,1] 内的带噪声观测，我们却知道 [1,2] 内的真实曲线，因此可以检查它向未见区域延伸时的表现。下面的浅色区域就是那段将来才交给模型的输入。

我们会训练宽度为4和16的网络，用训练区间内另一组位置比较它们，再查看选定宽度在区间内外各自的误差。验证与测试用无噪声规律，衡量的是模型恢复这条曲线的程度。
''')
    l.code(r'''
data_rng=np.random.default_rng(12)
x_train=np.sort(data_rng.uniform(-1,1,60)).astype('float32')[:,None]
y_train=np.sin(3*x_train)+data_rng.normal(0,.05,x_train.shape).astype('float32')
x_val=np.linspace(-.98,.98,41,dtype='float32')[:,None]
x_test=np.linspace(-.99,.99,80,dtype='float32')[:,None]
x_out=np.linspace(1.02,2,80,dtype='float32')[:,None]
''')
    l.code(r'''
fig,ax=plt.subplots(figsize=(7.2,3.6))
axis=np.linspace(-1,2,200)
ax.plot(axis,np.sin(3*axis),color='black',label='Generating function')
ax.scatter(x_train,y_train,s=16,label='Training observations')
ax.axvspan(1,2,color='#B66A47',alpha=.12,label='Extrapolation interval')
ax.set(xlabel='Input x',ylabel='Response y');ax.legend(frameon=False)
show(fig)
''')
    l.ex('r01','计算一个集合的误差','真实值 `[1,2]`，预测 `[1.5,1.5]`，算均方误差。',
         'ex_r01_mse=None','先逐项相减、平方，再对两个样本平均。',
         'ex_r01_mse=np.mean((np.array([1.5,1.5])-np.array([1.,2.]))**2)',
         "check('MSE',ex_r01_mse,.25)")
    l.md('## 2 给同一个任务两种容量\n\n宽度4和16意味着隐藏层有不同数量的可调曲线成分。更宽的网络可以表示更多变化，但最终学到的曲线还受初始权重与优化过程影响。每种宽度从三个不同的初始化开始，便能看见一次运行以外的变化。\n\n两种网络使用同一份数据、Tanh 激活和200次更新。宽模型每步处理的参数更多，输出中也会列出参数量；这次比较的是相同更新次数下的结果。')
    l.code(r'''
train_x=torch.from_numpy(x_train);train_y=torch.from_numpy(y_train)
validation_x=torch.from_numpy(x_val)
configurations=[dict(width=4),dict(width=16)]
seeds=[0,1,2]
print('训练输入、目标：',train_x.shape,train_y.shape)
''')
    l.md('把一次运行写成透明的小函数：新建模型、更新200次、返回模型与训练曲线。评价函数只接收已有模型，不更新参数。')
    l.code(r'''
def fit_once(width,seed):
    torch.manual_seed(seed)
    model=nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    losses=[]
    for step in range(200):
        loss=((model(train_x)-train_y)**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward();optimizer.step()
        losses.append(loss.item())
    return model,losses
@torch.no_grad()
def error(model,x):
    model.eval()
    target=torch.sin(3*torch.from_numpy(x))
    return ((model(torch.from_numpy(x))-target)**2).mean().item()
''')
    l.ex('r02','每个配置应该跑几次','列出 width 为 4、16 且 seed 为 0、1、2 的全部组合。',
         'ex_r02_pairs=None', '外层循环配置，内层循环种子。',
         'ex_r02_pairs=[(width,seed) for width in [4,16] for seed in [0,1,2]]',
         "check('实验组合',ex_r02_pairs,[[4,0],[4,1],[4,2],[16,0],[16,1],[16,2]])")
    l.code(r'''
records=[];trained={}
for cfg in configurations:
    for seed in seeds:
        model,losses=fit_once(cfg['width'],seed)
        validation=error(model,x_val)
        key=(cfg['width'],seed);trained[key]=model
        records.append(dict(width=cfg['width'],seed=seed,val_mse=validation,
                            parameters=sum(p.numel() for p in model.parameters()),
                            first_loss=losses[0],last_loss=losses[-1]))
print(*records,sep='\n')
''')
    l.md('## 3 从六次结果里选择一个宽度\n\n现在每种宽度都有三份验证误差。它们来自同一组验证位置，差别在于初始权重。把每组的三个误差取平均，选均值较低的宽度；旁边的样本标准差描述这三次初始化带来的散布。\n\n验证位置没有参与梯度更新，却参与了这次选择。稍后另外一组测试位置会评价已经选好的宽度，这样便能分开“据此作选择”和“查看选择后的结果”两个动作。')
    l.code(r'''
validation_means={}
for width in [4,16]:
    values=np.array([r['val_mse'] for r in records if r['width']==width])
    validation_means[width]=float(values.mean())
    print('width=',width,'values=',values,'mean=',values.mean(),'std=',values.std(ddof=1))
chosen_width=min(validation_means,key=validation_means.get)
print('验证集选定宽度：',chosen_width)
''')
    l.ex('r03','根据验证误差选模型','候选均值 `{4:0.04,16:0.02}`，返回选定宽度。',
         'ex_r03_width=None', '比较字典的值，返回对应的键。',
         'ex_r03_scores={4:.04,16:.02}\nex_r03_width=min(ex_r03_scores,key=ex_r03_scores.get)',
         "check('选定配置',ex_r03_width,16)")
    l.md('## 4 配置确定后查看测试结果\n\n对选定宽度的三个种子统一计算插值和外推误差。外推区域的输入分布已经改变；分别报告两个区域，才能看见模型在哪种问题上有效。')
    l.code(r'''
test_records=[]
for seed in seeds:
    model=trained[(chosen_width,seed)]
    test_records.append(dict(width=chosen_width,seed=seed,
                             interpolation_mse=error(model,x_test),extrapolation_mse=error(model,x_out)))
print(*test_records,sep='\n')
''')
    l.code(r'''
fig,ax=plt.subplots(figsize=(7.2,3.8))
ax.plot(axis,np.sin(3*axis),color='black',label='Generating function')
with torch.no_grad():
    for seed in seeds:
        prediction=trained[(chosen_width,seed)](torch.tensor(axis[:,None],dtype=torch.float32)).numpy()
        ax.plot(axis,prediction[:,0],label=f'Seed {seed}',alpha=.75)
ax.axvline(1,color='#758A95',ls='--');ax.scatter(x_train,y_train,s=12,color='#758A95')
ax.set(xlabel='Input x',ylabel='Prediction and response');ax.legend(frameon=False)
show(fig)
''')
    l.ex('r04','解释多次运行的散布','对 `[0.02,0.03,0.04]` 计算均值与样本标准差。',
         'ex_r04_mean=None\nex_r04_std=None', '标准差使用 ddof=1。',
         'ex_r04_mean=np.mean([.02,.03,.04])\nex_r04_std=np.std([.02,.03,.04],ddof=1)',
         "check('三次均值',ex_r04_mean,.03)\ncheck('样本标准差',ex_r04_std,.01)")
    l.md('## 5 把问题、实现与观察连起来\n\n沿图检查三个种子在训练区间内是否接近；再看区间外是否分开。用实测误差回答当前问题。模型在训练数据上损失下降，说明优化确实找到了能拟合这些样本的参数；外推能力还取决于结构、数据覆盖与生成规律。\n\n下一轮可以保持宽度不变，只把 Tanh 改为 ReLU；或者保持模型不变，只扩大训练区间。分别提出预期，再运行对照。')
    l.ex('r05','为一个测试样本标记区域','输入 `[0.2,0.8,1.4]`，以 x>1 标记外推区域。',
         'ex_r05_out=None','逐元素比较会产生布尔数组。',
         'ex_r05_out=np.array([.2,.8,1.4])>1',
         "check('外推区域',ex_r05_out,[False,False,True])")
    l.md('## 6 保存可重新解释的记录\n\n记录包括生成规律、固定数据种子、候选配置、优化种子、更新次数、验证选择和测试结果。别人读到一个数字时，应能找到它对应的运行。')
    l.code(r'''
experiment=dict(problem='sin(3x), noisy training observations',data_seed=12,
                widths=[4,16],optimization_seeds=seeds,updates=200,
                chosen_width=chosen_width,validation=records,test=test_records)
destination=ROOT/'.build'/'artifacts';destination.mkdir(parents=True,exist_ok=True)
(destination/'research_experiment.json').write_text(json.dumps(experiment,indent=2),encoding='utf-8')
print('记录位置：',destination/'research_experiment.json')
''')
    l.ex('r06','写下下一轮的控制变量','将保持不变的量写进字典：width=16、seed=0、updates=200。',
         'ex_r06_controls=None','每个键记录一项设置。',
         "ex_r06_controls=dict(width=16,seed=0,updates=200)",
         "check('固定设置',ex_r06_controls,dict(width=16,seed=0,updates=200))")
    l.md('''
## 7 按研究对象选择下一步

| 对象 | 可以检验的结构假设 | 一个具体对照 |
| --- | --- | --- |
| 标定响应 | MLP 能表示非线性 | 与线性模型比较独立测量上的误差 |
| 图像/波形局部模式 | 卷积共享局部检测器 | 固定参数量，改变平移与位置 |
| 时序记忆 | 隐状态保留过去信息 | 改变依赖间隔并检查早期梯度 |
| 序列关系 | 注意力读取相关位置 | 对固定输入遮挡特定位置，测量损失变化 |
| 探测通道/图结构 | 邻接关系辅助信息聚合 | 与打乱边的图、无边 MLP 对照 |
| 不规则采样过程 | 连续时间状态使用实际 Δt | 固定观测值，改变采样间隔检验响应 |

论文阅读顺序：先写任务和实验对象，再找模型方程、训练目标、基线、数据划分与主要结果表。将其中一个结论缩小到自己能复现的小实验，检查它需要哪些条件。
''')
    l.md(r'''
## 8 把这些网络放回机器学习的版图

数字分类与字符生成看起来不同，却都要连接六件事：

| 环节 | 第09课的数字分类 | 第14课的下一字符预测 |
| --- | --- | --- |
| 任务 | 根据图像预测类别 | 根据前文预测后一个字符 |
| 数据表示 | 样本、通道、高、宽 | 样本、位置、嵌入特征 |
| 模型结构 | CNN 的局部连接与共享卷积核 | Transformer 的位置与因果注意力 |
| 训练信号 | 人给图像标出的数字 | 原文中本来就有的后一个字符 |
| 损失 | 正确类别的负对数概率 | 真实后续字符的负对数概率 |
| 参数更新 | 对损失反传，再更新权重 | 对损失反传，再更新权重 |
| 优化器 | Adam | AdamW |

第一种训练信号是**监督学习**中的人工标签。第二种从数据自身构造目标，称为**自监督学习**。同一种网络结构可以换任务，同一个任务也可以换结构；先写清输入、目标与评价量，再决定用哪种网络。

把模型记为 $f_\theta$，θ 是全部可训练参数。第 i 个样本为 $(x_i,y_i)$，共 N 个样本，单个样本的损失记为 ℓ。训练时计算的是有限样本上的平均：
$$\widehat R(\theta)=\frac1N\sum_{i=1}^N\ell(f_\theta(x_i),y_i).$$
其中 ℓ 可以是本课的平方误差，也可以是分类中的交叉熵。**优化**研究如何降低这个训练目标；**泛化**关心新样本上的损失。若新样本来自分布 p，其平均损失写成
$$R_p(\theta)=\mathbb E_{(x,y)\sim p}[\ell(f_\theta(x),y)].$$
期望符号表示按 p 的出现频率对所有可能样本求平均。实验里只能用有限的独立样本估计它。这里的插值与外推使用不同输入区域，因此对应不同的 p；训练损失相近的两个网络，在区域外可以产生很不一样的曲线。

**表示**关心模型能构成什么函数。卷积的局部连接、图网络的邻居聚合、LTC 的连续时间状态，都把某些关系提前写进结构；这种偏好称为归纳偏置。它是否适合任务，可以通过平移、打乱边、改变采样间隔等对照来检验。

读论文时，先把新方法放在这三处之一：它改变了表示、优化，还是训练信号与数据？再找出支持相应主张的对照实验。沿这条线阅读，网络名称就能和具体的问题、方程与证据对应起来。
''')
    l.reading('Neural Ordinary Differential Equations','https://arxiv.org/abs/1806.07366',
              '读从离散残差更新走到连续时间微分方程的定义。对照第09、17课：改变的是模型参数化、积分步长还是训练目标？分别设计只改变其中一项的实验。')
    result.append(l)
    return result
