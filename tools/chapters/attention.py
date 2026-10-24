"""From a hand-computed weighted sum to a trained decoder and minGPT."""
from pathlib import Path
import ast
from coursekit import Lesson

ROOT = Path(__file__).resolve().parents[2]


def setup(l):
    l.code(r'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)
torch.manual_seed(7)
style()
''')


def class_source(name):
    source = (ROOT / 'models/tiny_gpt.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    text = ast.get_source_segment(source, node)
    return text.split('    @torch.no_grad()')[0].rstrip() if name == 'TinyGPT' else text


def build():
    lessons = []
    l = Lesson('12_attention_qkv', '12 从一次加权平均理解 Q、K、V',
               ['手算一个 query 的权重与输出', '画出每个位置允许读取哪些位置', '检查注意力的输入梯度与因果性'], '05–07，10 的序列形状')
    setup(l)
    l.md(r'''
## 1 从三条记录里，取出此刻有用的信息

RNN 把过去读过的内容逐步压进一个状态。如果每个位置的表示都保留下来，当前位置就可以回头选择：这一次更需要哪几处的信息？注意力从这个动作开始。

把问题缩小到三条记录。每条记录有两份向量：一份用来判断它是否相关，叫键 k；一份保存被读取的内容，叫值 v。当前的查询 q 与各个键做点积，得到三个匹配分数。这里取 q=[1,0]，所以只看键的第一个坐标；后面的代码会算出分数 [1,0,1]。

第一条和第三条匹配程度相同，但它们提供的内容不同。把“按什么寻找”与“实际取回什么”分开，正是 K 与 V 各自承担的工作。这些小向量暂由我们指定，之后会让网络学出来。记键与查询的长度为 d，值向量的长度为 $d_v$。
''')
    l.code(r'''
q = torch.tensor([1., 0.])
keys = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
values = torch.tensor([[2., 0.], [0., 4.], [6., 2.]])
scores = torch.stack([(q * key).sum() for key in keys])
print('三个点积：', scores)
''')
    l.md(r'''
现在需要把三个匹配分数变成三个取用比例。Softmax 给出正权重 $p_j=e^{z_j}/\sum_m e^{z_m}$，它们加起来为1。较高分数分到更多比例，同分则分到相同比例。这里先将分数除以 $\sqrt2$，第3节会解释这个缩放。

每条记录贡献 $p_jv_j$，相加就是输出 $y=\sum_jp_jv_j$。读下面的“逐项贡献”表：每一行来自一条记录，每一列是内容向量的一个坐标。代码先减最大分数再取指数，Softmax 的比值不变，计算中的数却小了许多。
''')
    l.code(r'''
z = scores / np.sqrt(2)
exp_z = torch.exp(z - z.max())
weights = exp_z / exp_z.sum()
weighted_values = weights[:, None] * values
answer = weighted_values.sum(dim=0)
print('权重：', weights, '总和：', weights.sum())
print('逐项贡献：\n', weighted_values, '\n输出：', answer)
''')
    l.ex('a01', '先不用矩阵乘法', '分数改为 `[0,0]`，值为 `[2,0]` 与 `[0,4]`。计算权重和输出。',
         'ex_a01_weights = None\nex_a01_answer = None',
         '两个分数相等，权重相等；先乘每个值向量，再沿候选位置方向求和。',
         'ex_a01_weights = torch.softmax(torch.tensor([0.,0.]), dim=0)\nex_a01_answer = (ex_a01_weights[:,None]*torch.tensor([[2.,0.],[0.,4.]])).sum(dim=0)',
         "check('等分权重',ex_a01_weights,[.5,.5])\ncheck('加权输出',ex_a01_answer,[1.,2.])")
    l.md(r'''
## 2 三个位置一起查询

刚才只有一个查询，得到一个向量。现在让三个位置各自查询一次：它们共用同一组键和值，却可以给出不同的取用比例。

把查询按行排为 $Q\in\mathbb R^{T\times d}$，键和值分别为 $K\in\mathbb R^{T\times d}$、$V\in\mathbb R^{T\times d_v}$。T 是同一条序列的位置数，这里为3。

$(QK^\top)_{ij}=q_i^\top k_j$，所以分数表的第 i 行表示位置 i 对全部位置的查询。沿最后一轴 Softmax，让**每一行**的权重和为 1。

$$A=\operatorname{softmax}_{\rm row}(QK^\top/\sqrt d),\qquad Y=AV.$$
''')
    l.code(r'''
Q = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
K, V = keys, values
table = Q @ K.T / (Q.shape[1] ** .5)
A = torch.softmax(table, dim=-1)
Y = A @ V
print('分数表：\n',table,'\n权重表：\n',A)
print('每行权重之和：',A.sum(dim=-1),'\n全部输出：\n',Y)
''')
    l.code(r'''
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), layout='constrained')
for ax, matrix, title in zip(axes, [table,A], ['Scores: query row, key column','Weights per query']):
    im = ax.imshow(matrix.detach(), cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='black')
    ax.set(xticks=range(3),yticks=range(3),xlabel='Key position',ylabel='Query position',title=title)
show(fig)
''')
    l.ex('a02','判断 Softmax 的轴','自己计算 Q 对全部 K 的分数表，并让每个查询对应一行概率。',
         'ex_a02_scores = None\nex_a02_A = None', 'Q @ K.T 后，最后一轴枚举可读的位置。',
         'ex_a02_scores = Q @ K.T / np.sqrt(2)\nex_a02_A = torch.softmax(ex_a02_scores,dim=-1)',
         "check('分数表',ex_a02_scores,table)\ncheck('权重',ex_a02_A,A)")
    l.md(r'''
## 3 缩放因子从哪里来

若 q、k 的各分量独立、均值为 0、方差为 1，则乘积每项方差为 1，d 项相加的方差为 d。除以 $\sqrt d$ 后方差回到 1。这个估计解释缩放的出发点；实际训练得到的分量有自己的相关性。

维数增加时，未缩放的点积波动变大；Softmax 可能因此把比例分得很悬殊。下面将 d 从4增加到64，用实际标准差观察这件事。缩放控制的是进入 Softmax 的分数尺度。
''')
    l.code(r'''
torch.manual_seed(3)
for d in [4,64]:
    random_q, random_k = torch.randn(2000,d), torch.randn(2000,d)
    dots = (random_q*random_k).sum(dim=-1)
    print('d=',d,'点积std=',round(dots.std().item(),2),
          '缩放后std=',round((dots/d**.5).std().item(),2))
''')
    l.md(r'''
## 4 预测下一个字符时，只读取当前位置及过去

生成一句话时，后面的字符尚未出现。训练时虽然整句已经在文件里，模型也应练习从已有前文作预测；否则它可以直接读取要预测的答案。

位置 i 预测第 i+1 个字符，于是只允许读取 $j\le i$。用 $-\infty$ 替换未来位置的分数，Softmax 中对应的指数就为零。下面的下三角表把“此刻可读到哪里”逐行画了出来。
''')
    l.code(r'''
allowed = torch.tril(torch.ones(3,3)).bool()
masked_scores = table.masked_fill(~allowed,float('-inf'))
causal_A = torch.softmax(masked_scores,dim=-1)
print('允许读取的位置：\n',allowed)
print('加mask后的权重：\n',causal_A)
check('未来位置权重为0',causal_A[~allowed],torch.zeros(3))
''')
    l.ex('a03','亲手制作因果掩码','制作长度为 4 的因果掩码，保留对角线。',
         'ex_a03_mask = None', '`torch.tril` 保留主对角线及下方，再转 bool。',
         'ex_a03_mask = torch.tril(torch.ones(4,4)).bool()',
         "check('4位置掩码',ex_a03_mask,np.tril(np.ones((4,4))).astype(bool))")
    l.md(r'''
## 5 Q、K、V 都由同一个输入投影出来

目前查询、键和值是分别填写的数组。实际网络收到的是同一份输入 X：每行代表一个位置。对它做三种可训练的线性变换，便得到 $Q=XW_Q$、$K=XW_K$、$V=XW_V$。三份权重让同一份原始特征分别承担查询、匹配与传递内容的工作。

下面把这几行接成函数，再改变最后一个输入位置。前两个输出保持相同，说明它们确实没有读取后来发生的变化。
''')
    l.code(r'''
torch.manual_seed(5)
Wq,Wk,Wv = [torch.randn(2,2) for _ in range(3)]
def causal_read(X):
    q,k,v = X@Wq, X@Wk, X@Wv
    scores = (q@k.T)/np.sqrt(2)
    mask = torch.tril(torch.ones(len(X),len(X))).bool()
    return torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)@v
X = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
print(causal_read(X))
''')
    l.code(r'''
changed = X.clone()
changed[-1] += torch.tensor([4.,-3.])
check('未来输入改变，早期输出相同',causal_read(changed)[:2],causal_read(X)[:2])
''')
    l.ex('a04','写一次查询输出','只算第 1 个位置，它可以读取位置 0 和 1。用相应两行 K、V 计算。',
         'ex_a04_answer = None', '先做 Q[1] @ K[:2].T，Softmax，再乘 V[:2]。',
         'ex_a04_answer = torch.softmax(Q[1]@K[:2].T/np.sqrt(2),dim=-1)@V[:2]',
         "check('第1位置的输出',ex_a04_answer,(causal_A@V)[1])")
    l.md(r'''
## 6 反向梯度同时经过三条投影路径

三份投影来自同一个 X。改变 X 的一个元素，会同时改变它提出的查询、供别人匹配的键和被别人取走的值。回传到这个元素的梯度，因而需要收齐三条路的贡献。

沿用上一节的具体数值，把这段依赖展开：
$$X\longrightarrow(Q,K,V)\longrightarrow S=QK^\top/\sqrt d
\longrightarrow A=\operatorname{softmax}_{\rm row}(\operatorname{mask}(S))
\longrightarrow Y=AV\longrightarrow L.$$

记 $D_Z=\partial L/\partial Z$，它与 Z 的形状相同。对本例，X、Q、K、V、Y 都为 3×2，S、A 为 3×3，$d=2$。
先为数值核对新建 float64 的输入和权重副本，并让三份权重也参与求导。
''')
    l.code(r'''
backward_X = X.detach().double().clone().requires_grad_()
backward_Wq, backward_Wk, backward_Wv = [
    weight.detach().double().clone().requires_grad_() for weight in (Wq, Wk, Wv)]
backward_q = backward_X @ backward_Wq
backward_k = backward_X @ backward_Wk
backward_v = backward_X @ backward_Wv
backward_scale = backward_q.shape[-1] ** .5
''')
    l.md('先重现正向的每个中间量。`retain_grad()` 让反传后可以读取这些中间节点收到的梯度，逐步与手算结果比较。')
    l.code(r'''
backward_allowed = torch.tril(torch.ones(len(X), len(X))).bool()
backward_scores = backward_q @ backward_k.T / backward_scale
backward_A = torch.softmax(backward_scores.masked_fill(~backward_allowed, float('-inf')), dim=-1)
backward_Y = backward_A @ backward_v
for intermediate in (backward_q, backward_k, backward_v, backward_scores, backward_A, backward_Y):
    intermediate.retain_grad()
print('因果权重 A：\n', backward_A.detach(), '\n输出 Y：\n', backward_Y.detach())
''')
    l.md(r'''
### 6.1 从输出梯度拆开 Y=AV

这里取 $L=\sum_{i,r}Y_{ir}$，所以输出收到的梯度 $G=D_Y$ 每项都是 1。
由 $Y_{ir}=\sum_jA_{ij}V_{jr}$ 得
$$D_V=A^\top G\quad(3\times2),\qquad D_A=GV^\top\quad(3\times3).$$
一个值向量被多个查询读取，$D_V$ 沿这些查询累加；一个权重影响输出向量的所有坐标，$D_A$ 沿特征坐标累加。
下面取出正向数值，显式写出反向计算。
''')
    l.code(r'''
backward_Q_value, backward_K_value, backward_V_value = [
    value.detach() for value in (backward_q, backward_k, backward_v)]
backward_A_value = backward_A.detach()
backward_G = torch.ones_like(backward_Y)
backward_dV = backward_A_value.T @ backward_G
backward_dA = backward_G @ backward_V_value.T
print('输出梯度 G：\n', backward_G)
print('D_V：\n', backward_dV, '\nD_A：\n', backward_dA)
''')
    l.md(r'''
### 6.2 把整行 Softmax 的偏导收集起来

同一查询行内，$\partial A_{ij}/\partial S_{im}=A_{ij}(\delta_{jm}-A_{im})$，
其中 $\delta_{jm}$ 在 j=m 时为 1，否则为 0。应用链式法则并整理求和：
$$D_{S,ij}=\sum_m D_{A,im}\frac{\partial A_{im}}{\partial S_{ij}}
=A_{ij}\left(D_{A,ij}-\sum_m A_{im}D_{A,im}\right).$$
这就是行 Softmax 的反向向量乘积：先算每行的概率加权平均，再逐项相减、乘概率。
mask 把未来位置替换成常数，因此这些位置对原分数 S 的导数为 0。
''')
    l.code(r'''
backward_row_average = (backward_A_value * backward_dA).sum(dim=-1, keepdim=True)
backward_dS = backward_A_value * (backward_dA - backward_row_average)
backward_dS = backward_dS.masked_fill(~backward_allowed, 0.)
print('各行加权平均：\n', backward_row_average, '\nD_S：\n', backward_dS)
check('mask位置不传梯度', backward_dS[~backward_allowed], torch.zeros(3, dtype=torch.float64))
check('每行分数梯度和为0', backward_dS.sum(dim=-1), torch.zeros(3, dtype=torch.float64))
''')
    l.md(r'''
### 6.3 从匹配分数回到查询与键

$S_{ij}=\sum_rQ_{ir}K_{jr}/\sqrt d$，所以局部导数分别为
$\partial S_{ij}/\partial Q_{ir}=K_{jr}/\sqrt d$、
$\partial S_{ij}/\partial K_{jr}=Q_{ir}/\sqrt d$。
把查询行、键列收到的贡献分别相加：
$$D_Q=D_SK/\sqrt d,\qquad D_K=D_S^\top Q/\sqrt d.$$
查询梯度收集该查询对所有键的匹配变化，键梯度收集所有查询对该键的匹配变化。
''')
    l.code(r'''
backward_dQ = backward_dS @ backward_K_value / backward_scale
backward_dK = backward_dS.T @ backward_Q_value / backward_scale
print('D_Q：\n', backward_dQ)
print('D_K：\n', backward_dK)
''')
    l.md(r'''
### 6.4 三条投影路径在输入处汇合

对于行向量约定的 $Q=XW_Q$，有 $D_X^{(Q)}=D_QW_Q^\top$；另外两条路径同理。
输入 X 在图中被使用三次，所以必须把三份梯度相加：
$$D_X=D_QW_Q^\top+D_KW_K^\top+D_VW_V^\top.$$
打印三张 3×2 表，分别观察查询、键、值路径给每个输入元素的贡献。
''')
    l.code(r'''
backward_dX_q = backward_dQ @ backward_Wq.detach().T
backward_dX_k = backward_dK @ backward_Wk.detach().T
backward_dX_v = backward_dV @ backward_Wv.detach().T
backward_dX = backward_dX_q + backward_dX_k + backward_dX_v
for name, contribution in [('Q路径', backward_dX_q), ('K路径', backward_dX_k), ('V路径', backward_dX_v)]:
    print(name, '\n', contribution)
print('汇合后的输入梯度：\n', backward_dX)
''')
    l.md('横轴为输入特征，纵轴为序列位置。四张图使用同一颜色尺度；在同一位置、同一特征处，Q、K、V 三张图的数值相加，得到总梯度图的对应格。\n\n位置0只能读取自己，只有一个分数的 Softmax 恒为1，因此该位置的 Q 路径梯度为0。它的 K 还会被后续位置查询，所以 K 路径可以有非零梯度。')
    l.code(r'''
parts = [backward_dX_q, backward_dX_k, backward_dX_v, backward_dX]
limit = max(part.abs().max().item() for part in parts)
fig, axes = plt.subplots(1, 4, figsize=(9.8, 3.2), layout='constrained')
for ax, part, title in zip(axes, parts, ['Q path', 'K path', 'V path', 'Total dL/dX']):
    heat = ax.imshow(part, cmap='coolwarm', vmin=-limit, vmax=limit)
    for row, col in np.ndindex(tuple(part.shape)):
        ax.text(col, row, f'{part[row,col]:.3f}', ha='center', va='center', fontsize=11,
                color='white' if abs(part[row,col].item()) > .6 * limit else 'black')
    ax.set(xticks=[0,1], yticks=[0,1,2], xlabel='Input feature', title=title)
axes[0].set_ylabel('Sequence position'); fig.colorbar(heat, ax=list(axes), shrink=.8)
show(fig)
''')
    l.md(r'''
每份投影权重也收到梯度。一个权重被三个输入位置共用，因此沿位置求和：
$$D_{W_Q}=X^\top D_Q,\qquad D_{W_K}=X^\top D_K,\qquad D_{W_V}=X^\top D_V.$$
这三份矩阵才是优化器更新投影参数时使用的量。
''')
    l.code(r'''
backward_dWq = backward_X.detach().T @ backward_dQ
backward_dWk = backward_X.detach().T @ backward_dK
backward_dWv = backward_X.detach().T @ backward_dV
for name, gradient in [('W_Q', backward_dWq), ('W_K', backward_dWk), ('W_V', backward_dWv)]:
    print(name, '的梯度：\n', gradient)
''')
    l.md('### 6.5 逐个中间量与 autograd 核对\n\n现在才对同一正向图调用 `backward()`。不仅比较最终输入梯度，也检查输出乘法、Softmax、匹配分数和三条投影的每一步。')
    l.code(r'''
backward_Y.sum().backward()
for name, manual, automatic in [
    ('G', backward_G, backward_Y.grad), ('D_V', backward_dV, backward_v.grad),
    ('D_A', backward_dA, backward_A.grad), ('D_S', backward_dS, backward_scores.grad),
    ('D_Q', backward_dQ, backward_q.grad), ('D_K', backward_dK, backward_k.grad)]:
    check(name + '逐步反传', manual, automatic, atol=1e-9)
''')
    l.code(r'''
check('三路径输入梯度之和', backward_dX, backward_X.grad, atol=1e-9)
check('查询投影权重梯度', backward_dWq, backward_Wq.grad, atol=1e-9)
check('键投影权重梯度', backward_dWk, backward_Wk.grad, atol=1e-9)
check('值投影权重梯度', backward_dWv, backward_Wv.grad, atol=1e-9)
''')
    l.md(r'''
### 6.6 再用一次正向扰动核对输入偏导

中心差分只改变 X[1,0]，比较损失在 $+\varepsilon$、$-\varepsilon$ 两个输入处的差。
函数使用本节的 float64 权重数值重新计算正向，与沿计算图逐边传播的结果比较。
''')
    l.code(r'''
def backward_forward(input_value):
    q_value = input_value @ backward_Wq.detach()
    k_value = input_value @ backward_Wk.detach()
    v_value = input_value @ backward_Wv.detach()
    score_value = q_value @ k_value.T / backward_scale
    a_value = torch.softmax(score_value.masked_fill(~backward_allowed, float('-inf')), dim=-1)
    return a_value @ v_value
''')
    l.code(r'''
eps = 1e-5
plus,minus = backward_X.detach().clone(),backward_X.detach().clone()
plus[1,0] += eps; minus[1,0] -= eps
finite_difference = (backward_forward(plus).sum()-backward_forward(minus).sum())/(2*eps)
print('输入X[1,0]：手写', backward_dX[1,0].item(), '差分', finite_difference.item())
check('输入偏导差分核对', backward_dX[1,0], finite_difference, atol=1e-7)
''')
    l.ex('a05','Softmax 的非对角导数','权重 `[0.2,0.3,0.5]` 中，求第 0 个概率对第 1 个分数的导数。',
         'ex_a05_derivative = None', 'i≠j 时 δij=0，所以结果是 -p_i*p_j。',
         'ex_a05_derivative = -.2*.3', "check('交叉偏导',ex_a05_derivative,-.06)")
    l.ex('a06','从空函数写出一次因果注意力','输入 q,k,v 已经是按行排列的二维表，完整写出分数、掩码、Softmax、加权和。',
         'def ex_a06_attention(q,k,v):\n    return None', '形状依次为 T×d、T×T、T×T、T×dv。',
         "def ex_a06_attention(q,k,v):\n    scores=q@k.T/q.shape[-1]**.5\n    mask=torch.tril(torch.ones(len(q),len(k))).bool()\n    return torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)@v",
         "check('整段因果输出',ex_a06_attention(Q,K,V),causal_A@V)")
    l.reading('Attention Is All You Need §3.2','https://arxiv.org/abs/1706.03762',
              '在论文公式中找出 Q、K、V、缩放、Softmax 对应的五行代码。独立实验：保持输入和投影相同，只移除 mask，再改变最后一个位置，测量第 0 个位置输出的变化。')
    lessons.append(l)

    l = Lesson('13_transformer_block','13 把注意力、残差与归一化接成 Transformer',
               ['逐层追踪 B、T、C', '写出多头注意力及 Pre-LN 残差块', '用因果性实验检查完整 decoder'], '09，12')
    setup(l)
    l.md(r'''
## 1 给字符一个可以学习的表示

上一课的注意力已经能读取一列向量。文本却由字符组成：怎样把字符交给它，又怎样把取回的信息接成下一层？这一课会把这些部件接成一个小型 Transformer。

可以先给每个字符编号，再为每个编号准备一行可训练的数。`nn.Embedding` 就是这张表。例如编号1取第1行，字符本身没有被当作数值大小参与乘法；被训练的是那一行的向量。

下面一次放入两条序列，每条有三个位置。B 表示序列条数，T 表示位置数，C 表示向量长度；于是整数编号从 `(B,T)` 变为 `(B,T,C)`。整张查找表为 `(V,C)`，V 是可用编号的总数。
''')
    l.code(r'''
tokens = torch.tensor([[0,1,2],[2,1,0]])
embedding = nn.Embedding(4,6)
embedded = embedding(tokens)
print('IDs:',tokens.shape,'权重表:',embedding.weight.shape,'向量:',embedded.shape)
check('查表等于取行',embedded[0,1],embedding.weight[1])
''')
    l.ex('t01','取出一个 token 的向量','从 embedded 取 batch 1、位置 2 的向量。',
         'ex_t01_vector = None', '两个下标先选序列与位置，最后一个轴全部保留。',
         'ex_t01_vector = embedded[1,2]', "check('该位置的向量',ex_t01_vector,embedding.weight[0])")
    l.md(r'''
## 2 位置信息与内容相加

“我喜欢你”和“你喜欢我”由同样的字组成，顺序却改变了关系。只按内容匹配的无 mask 注意力，在输入重排时也随之重排；位置还需要自己的表示。这里为位置建立另一张可学习的表 $P\in\mathbb R^{T_{\max}\times C}$。
$T_{\max}$ 是位置表允许的最大序列长度，对应模型配置的 context 上限。

位置向量形状 `(T,C)`，相加时广播到 B 条序列：$H_{b,t}=E_{\mathrm{id}_{b,t}}+P_t$。
''')
    l.code(r'''
positions = nn.Embedding(8,6)
position_vectors = positions(torch.arange(tokens.shape[1]))
hidden = embedded + position_vectors
print('位置表输出：',position_vectors.shape,'相加后：',hidden.shape)
''')
    l.ex('t02','看清广播的两个轴','取 batch 0、位置 1，自己把内容向量和位置向量相加。',
         'ex_t02_hidden = None', '两个加数都应是长度为 C 的向量。',
         'ex_t02_hidden = embedded[0,1]+position_vectors[1]', "check('位置加内容',ex_t02_hidden,hidden[0,1])")
    l.md(r'''
## 3 先将上一课的一头装进 Module

线性投影作用于最后一个轴，前面的 B、T 都保留。缓存的三角 mask 不是要训练的参数，使用 `register_buffer`，它会跟随模型保存与迁移设备。
''')
    l.code(class_source('AttentionHead'))
    l.code(r'''
head = AttentionHead(width=6,head_size=3,context=8)
one_head = head(hidden)
print('一头输出：',one_head.shape)
''')
    l.md(r'''
## 4 多头让同一个位置有多套读取方式

一套 Q、K、V 只给每个位置形成一种读取结果。多头注意力给它多套独立投影，于是同一位置可以同时取得几种不同的组合；这些组合如何有用，由训练目标决定。

每头输出 `(B,T,D)`，H 头沿最后一轴拼接为 `(B,T,H×D)`，取 H×D=C。拼接保留各头的结果，随后的一次线性投影再将它们混合。

`ModuleList` 注册一组可训练子模块；普通 Python 列表不会自动完成参数注册。先显式逐头计算，后面阅读 minGPT 时再把 H 合进 batch 维。
''')
    l.code(class_source('MultiHead'))
    l.code(r'''
multi = MultiHead(width=6,heads=2,context=8)
print('两头各自输出：',[h(hidden).shape for h in multi.heads])
print('拼接与投影后：',multi(hidden).shape)
''')
    l.ex('t03','两头怎样拼接','把形状相同的两个头沿特征轴拼接，保留 B 与 T。',
         'ex_t03_joined = None', '`torch.cat([...],dim=-1)`。',
         'ex_t03_joined = torch.cat([one_head,one_head],dim=-1)',
         "check('拼接形状',None if ex_t03_joined is None else ex_t03_joined.shape,(2,3,6))\ncheck('第一头仍保留',None if ex_t03_joined is None else ex_t03_joined[...,:3],one_head)")
    l.md(r'''
## 5 LayerNorm 在每个位置内部归一化

多层运算会不断改变向量的尺度。LayerNorm 在一个位置内部，把各坐标减去均值，再除以共同的标准差；随后用可学习的尺度和偏移调整输出。

对一个 C 维向量，$m=C^{-1}\sum_c x_c$，$v=C^{-1}\sum_c(x_c-m)^2$，输出 $\gamma_c(x_c-m)/\sqrt{v+\epsilon}+\beta_c$。γ、β 各有 C 项。$\epsilon$ 是防止分母为零的正小常数，这里为 $10^{-5}$。

下面只计算一个位置。方差按 C 除，`unbiased=False` 对应这一公式。
''')
    l.code(r'''
sample = torch.tensor([1.,2.,3.])
mean = sample.mean()
variance = ((sample-mean)**2).mean()
manual_norm = (sample-mean)/torch.sqrt(variance+1e-5)
norm = nn.LayerNorm(3)
check('手算 LayerNorm',norm(sample),manual_norm)
''')
    l.ex('t04','给归一化增加尺度与偏移','γ 全为 2、β 全为 1，计算归一化后的向量。',
         'ex_t04_affine = None', '先用上面的 manual_norm，再逐坐标乘 γ、加 β。',
         'ex_t04_affine = 2*manual_norm+1', "check('尺度和偏移',ex_t04_affine,2*manual_norm+1)")
    l.md(r'''
## 6 残差上的导数包含两条路径

注意力读出的向量作为对原表示的一次修正，加回输入：$y=x+F(x)$。于是输入可以沿直连路径继续传递，也可以经过 F 参与修正。

反向时，下游梯度分别进入两条支路，回到 x 时相加：$\partial L/\partial x=(I+J_F)^\top\partial L/\partial y$。$J_F$ 是 F 对 x 的 Jacobian。对标量 $F(x)=2x$，两条贡献为1和2，合起来是3。

[打开残差实验](../interactive/index.html#residual)，逐步回传同一个梯度，再切换到两路抵消的例子。直连贡献存在时，总梯度仍取决于两路相加的结果。
''')
    l.code(r'''
residual_input = torch.tensor(2.,requires_grad=True)
residual_output = residual_input+2*residual_input
residual_output.backward()
check('两条路径的导数相加',residual_input.grad,3.)
''')
    l.md('每个位置的前馈网络先把 C 扩成 4C，经过非线性后再投回 C。所有位置共享这组权重；位置间的信息交换发生在注意力中。')
    l.code(class_source('FeedForward'))
    l.md('接成 Pre-LN 块：先归一化、做注意力、加回输入，再归一化、做前馈、加回上一状态。每次残差相加的两项形状相同。')
    l.code(class_source('Block'))
    l.md('沿图先走直连，再走下方的计算分支；每个 Σ 把两项相加。反向时，下游梯度从 Σ 同时进入两条输入路径，回到分支起点后再求和。图中的全部支路对应上面 forward 的两行更新。')
    l.code('from support import block_diagram\nshow(block_diagram())')
    l.code(r'''
block = Block(width=6,heads=2,context=8)
block_output = block(hidden)
check('块保持B,T,C',block_output.shape,hidden.shape)
''')
    l.ex('t05','独立写一个残差块的 forward','已有 attention、ff、norm1、norm2，按顺序写出两次残差。',
         'def ex_t05_forward(module,x):\n    return None', '第二次加回的是第一次更新后的 x。',
         'def ex_t05_forward(module,x):\n    x=x+module.attention(module.norm1(x))\n    return x+module.ff(module.norm2(x))',
         "check('两次残差的顺序',ex_t05_forward(block,hidden),block_output)")
    l.md('最后堆叠若干块，接最终归一化与词表投影。每个位置输出 V 个 logits，形状 `(B,T,V)`。这里把已经写过的组件组装成完整模型。')
    l.code(class_source('TinyGPT'))
    l.code(r'''
model = TinyGPT(vocab_size=4,context=8,width=6,heads=2,depth=2)
logits = model(tokens)
print('最终logits：',logits.shape)
changed_tokens = tokens.clone(); changed_tokens[:, -1] = 3
check('完整模型不会读取未来token',model(changed_tokens)[:,:2],logits[:,:2])
print('参数总数：',sum(p.numel() for p in model.parameters()))
''')
    l.ex('t06','取下一 token 的预测分数','取第一条序列最后一个位置的全部词表分数。',
         'ex_t06_last = None', '先选 batch 0，再选位置 -1；保留词表轴。',
         'ex_t06_last = logits[0,-1,:]', "check('下一个token的分数',ex_t06_last,logits[0,-1])")
    l.reading('minGPT: CausalSelfAttention 与 Block','https://github.com/karpathy/minGPT/blob/master/mingpt/model.py',
              '将这课的逐头实现对照其 view(B,T,H,D).transpose(1,2)。画出两个实现的形状表。独立实验：保留参数和输入，去掉位置信息并同时重排位置，先在无mask的一头上检验置换等变性。')
    lessons.append(l)

    l = Lesson('14_small_gpt_training','14 自己训练一个微型字符 GPT',
               ['构造错开一位的训练目标', '训练、验证与保存真实的decoder', '从下一个字符概率逐步生成文本'], '06–07，12–13')
    setup(l)
    l.md(r'''
## 1 让小模型接着写一句话

给出 `the `，让模型继续写。刚开始它只有随机权重，字母往往拼不成词；训练之后，我们希望它能接出语料中出现过的词和句子结构。这一课就用上一课搭好的网络完成这件事。

语料由主语、动词与时间描述组合成27句，例如 `the sun glows slowly.`。这些句子短，规律也看得见，便于判断模型究竟学到了字符搭配、完整词，还是常常重复的片段。

18句用于学习权重，5句用于比较后续配置，另外4句留作测试。按完整句子拆开，便能观察模型面对未用于训练的句子组合时的表现。下一格生成并打印其中的一段。
''')
    l.code(r'''
sentences = [f'{subject} {verb} {ending}.\n'
             for subject in ['the sun','a star','the moon']
             for verb in ['moves','glows','rises']
             for ending in ['slowly','quietly','today']]
order = np.random.default_rng(7).permutation(len(sentences))
train_text = ''.join(sentences[i] for i in order[:18])
val_text = ''.join(sentences[i] for i in order[18:23])
test_text = ''.join(sentences[i] for i in order[23:])
print('训练句子示例：',repr(train_text[:90]))
''')
    l.md('字符词表事先固定为换行、空格、小写字母与句点。字典把字符映射为整数；解码按相反方向查表。一个字符就是这一课的一个 token。')
    l.code(r'''
alphabet = '\n abcdefghijklmnopqrstuvwxyz.'
stoi = {character:i for i,character in enumerate(alphabet)}
def encode(text):
    return torch.tensor([stoi[c] for c in text],dtype=torch.long)
def decode(ids):
    return ''.join(alphabet[int(i)] for i in ids)
train_ids,val_ids,test_ids = map(encode,[train_text,val_text,test_text])
print(encode('the sun.'),decode(encode('the sun.')))
''')
    l.ex('g01','实现一段编码','用 stoi 将字符串 moon 变成整数张量。',
         'ex_g01_ids = None', '逐字符查字典；Embedding 的索引类型应为 long。',
         "ex_g01_ids = torch.tensor([stoi[c] for c in 'moon'],dtype=torch.long)",
         "check('moon的编码',ex_g01_ids,encode('moon'))")
    l.md(r'''
## 2 一份窗口同时提供输入和标签

一段原文自己就提供了练习题和答案。例如读到 `the`，下一字符是空格；读到 `the s`，下一字符是 `u`。把原文复制一份、错开一个位置，就能一次取得许多这样的预测任务。

窗口起点为 s、长度为 T：输入取 `[s:s+T]`，目标取 `[s+1:s+T+1]`。下面打印的两行正好差一位。目标数组虽然整行存着，因果 mask 仍让位置 t 只读取它自己的前缀。
''')
    l.code(r'''
context = 16
x0,y0 = train_ids[:context],train_ids[1:context+1]
print('输入：',repr(decode(x0)))
print('标签：',repr(decode(y0)))
print('逐个检查：',[(decode([a]),decode([b])) for a,b in zip(x0[:5],y0[:5])])
''')
    l.code(r'''
def get_batch(data,batch_size=16):
    starts = torch.randint(len(data)-context,(batch_size,))
    x = torch.stack([data[s:s+context] for s in starts])
    y = torch.stack([data[s+1:s+context+1] for s in starts])
    return x,y
xb,yb = get_batch(train_ids)
print('输入和标签：',xb.shape,yb.shape)
''')
    l.ex('g02','写出错开一位的目标','数组为 `[0,1,2,3,4,5]`，起点 1、窗口长度 3。',
         'ex_g02_x = None\nex_g02_y = None', '输入是位置1、2、3；目标是2、3、4。',
         'ex_g02_data=torch.arange(6)\nex_g02_x=ex_g02_data[1:4]\nex_g02_y=ex_g02_data[2:5]',
         "check('窗口输入',ex_g02_x,[1,2,3])\ncheck('下一位标签',ex_g02_y,[2,3,4])")
    l.md(r'''
## 3 每个位置都是一次词表分类

TinyGPT 为每个位置给出一张“下个字符分数表”。第13课写过的定义保存在 `models/tiny_gpt.py`，这里将它用于真正的训练。输出 `(B,T,V)` 的最后一个轴列出所有字符的分数。

每个位置都有一个真实的后续字符，所以可以把 B、T 合成一轴，得到许多个词表分类任务。交叉熵接收 `(B×T,V)` 分数和 `(B×T,)` 整数目标：

$$L=-\frac1{BT}\sum_{b,t}\log p_\theta(y_{b,t}\mid x_{b,\le t}).$$

如果真实字符的概率很小，负对数损失就大；提高它的概率，损失便下降。这个信号会穿过输出层、Transformer 块，一直回到字符与位置嵌入。
''')
    l.code(r'''
from models.tiny_gpt import TinyGPT
torch.manual_seed(7)
config = dict(vocab_size=len(alphabet),context=context,width=24,heads=2,depth=1)
model = TinyGPT(**config)
logits = model(xb)
loss = F.cross_entropy(logits.reshape(-1,len(alphabet)),yb.reshape(-1))
print('logits:',logits.shape,'loss:',loss.item())
''')
    l.ex('g03','正确展开位置','把 logits 和 yb 排成交叉熵需要的两个形状。',
         'ex_g03_scores = None\nex_g03_targets = None', '只合并 B、T 两轴，保留完整词表轴。',
         'ex_g03_scores=logits.reshape(-1,len(alphabet))\nex_g03_targets=yb.reshape(-1)',
         "check('分数形状',None if ex_g03_scores is None else ex_g03_scores.shape,(256,len(alphabet)))\ncheck('标签形状',None if ex_g03_targets is None else ex_g03_targets.shape,(256,))")
    l.md('每次从训练文本取一批窗口，网络为所有位置给分，损失再把修正方向传回参数。这个循环与第06课相同；变化的是输入、目标和模型。AdamW 的学习率控制更新尺度，weight_decay 在更新中对权重施加衰减。')
    l.code(r'''
optimizer = torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.01)
history = []
for step in range(140):
    model.train()
    xb,yb = get_batch(train_ids)
    scores = model(xb)
    loss = F.cross_entropy(scores.reshape(-1,len(alphabet)),yb.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    history.append(loss.item())
print('前10步平均loss：',np.mean(history[:10]),'后10步：',np.mean(history[-10:]))
''')
    l.md('前10步与后10步的平均损失，给出了这140次更新的两个截面。下面的整条曲线还能看见中间的波动：不同窗口的难度不同，一次较高的损失不等于训练倒退。随后用完整的验证与测试文本，查看未用于更新权重的字符预测。')
    l.code(r'''
fig,ax = plt.subplots(figsize=(6,3.2))
ax.plot(history,color='#37798A')
ax.set(xlabel='Update step',ylabel='Training cross entropy')
show(fig)
''')
    l.code(r'''
@torch.no_grad()
def text_loss(data):
    model.eval()
    total,targets=0.,0
    for start in range(0,len(data)-1,context):
        length=min(context,len(data)-start-1)
        x=data[start:start+length][None,:]
        y=data[start+1:start+length+1]
        total+=F.cross_entropy(model(x).reshape(-1,len(alphabet)),y,reduction='sum').item()
        targets+=length
    return total/targets
print('validation CE:',text_loss(val_ids),'test CE:',text_loss(test_ids))
''')
    l.md(r'''
## 4 给结果找一个容易解释的基线

Unigram 基线只数训练字符频率，不读取上下文。给每个字符加 1 个计数，概率为 $(n_c+1)/(N+V)$，测试损失是对应负对数概率的平均。
''')
    l.code(r'''
counts = torch.bincount(train_ids,minlength=len(alphabet)).float()+1
unigram = counts/counts.sum()
baseline_loss = -unigram[test_ids[1:]].log().mean().item()
print('unigram test CE:',baseline_loss,'decoder test CE:',text_loss(test_ids))
''')
    l.ex('g04','从一个概率得到负对数损失','正确 token 概率为 0.25，计算单位置的交叉熵。',
         'ex_g04_loss = None', '用自然对数：-log(p)。',
         'ex_g04_loss=-np.log(.25)', "check('单位置loss',ex_g04_loss,np.log(4))")
    l.md(r'''
## 5 生成就是反复追加一个 token

训练时答案来自原文，生成时却要把自己的选择接回输入。给定 `the `，读取最后一个位置的词表分数，转成概率、抽取一个字符、接到末尾；重复这个动作，就从一次预测变成了一段文本。

模型每轮读取当前文本最后至多 T 个 token。温度 τ 是分数除以的正数：温度低时，高分字符占比更大；温度高时，抽到其他字符的机会增加。下一格只追加一次，后面再连续生成70个字符。
''')
    l.code(r'''
prefix = encode('the ')[None,:]
model.eval()
with torch.no_grad():
    last_logits=model(prefix)[:,-1,:]
    probabilities=torch.softmax(last_logits/.8,dim=-1)
    next_id=torch.multinomial(probabilities,num_samples=1)
    extended=torch.cat([prefix,next_id],dim=1)
print('追加一个字符：',repr(decode(extended[0])))
''')
    l.code(r'''
torch.manual_seed(8)
generated=model.generate(prefix,steps=70,temperature=.8)
print(decode(generated[0]))
''')
    l.ex('g05','把采样改为贪心选择','从 last_logits 选分数最高的 ID，形状保留为 `(1,1)` 以便追加。',
         'ex_g05_id = None', '`argmax(dim=-1,keepdim=True)`。',
         'ex_g05_id=last_logits.argmax(dim=-1,keepdim=True)',
         "check('贪心选择',ex_g05_id,last_logits.argmax(dim=-1,keepdim=True))")
    l.md('保存参数时同时保存构建配置，重新加载后核对同一个输入的 logits。训练状态与数据划分记录也应跟着实验保存。')
    l.code(r'''
artifact = ROOT/'.build'/'artifacts'
artifact.mkdir(parents=True,exist_ok=True)
torch.save(dict(config=config,state_dict=model.state_dict()),artifact/'tiny_gpt.pt')
saved=torch.load(artifact/'tiny_gpt.pt',map_location='cpu',weights_only=True)
restored=TinyGPT(**saved['config'])
restored.load_state_dict(saved['state_dict']); restored.eval()
check('保存后重建同一模型',restored(prefix),model(prefix))
''')
    l.ex('g06','数一数更新的参数','使用 model.parameters()，计算所有可训练参数的标量总数。',
         'ex_g06_count = None', '每个 p.numel() 是该张量的元素数，把这些数相加。',
         'ex_g06_count=sum(p.numel() for p in model.parameters())',
         "check('参数总数',ex_g06_count,sum(p.numel() for p in model.parameters()))")
    l.md('## 独立实验\n\n固定句子划分，比较 width=12 与 width=24，各运行三个种子。记录参数量、训练/验证损失和更新次数。用验证结果选择配置后，统一查看测试结果。保留生成文本中的重复、断词和语法错误，逐类判断模型已经学到了哪些局部规律。')
    l.reading('minGPT 的字符模型项目','https://github.com/karpathy/minGPT/tree/master/projects/chargpt',
              '找出 Dataset 返回的 x、y 和这里错开一位的切片。沿一条 batch 追踪到 model.forward、交叉熵、backward、optimizer.step。')
    lessons.append(l)

    l=Lesson('15_mingpt_reading_and_generation','15 打开 minGPT，把每个组件认出来',
             ['读懂真实项目的模型/数据/训练边界','核验多头 reshape 与因果性','用一个微型任务调用原项目模型'], '12–14')
    setup(l)
    l.md('## 1 在真实项目里认出自己写过的计算\n\n上一课已经生成了一段文本。现在打开 minGPT，目标是看懂：同样的 QKV、残差和下一字符预测，在一个真实项目里怎样组织。遇到更短、更快的写法时，可以拿自己的逐头实现逐项对照。\n\n`mingpt/model.py` 定义注意力、Block 和 GPT；`trainer.py` 接训练循环；`projects/chargpt` 将文本变成窗口。本地副本在 `references/minGPT`，下面先列出模型中的类与前向函数，找到熟悉的入口。')
    l.code(r'''
reference=ROOT/'references'/'minGPT'
model_source=(reference/'mingpt/model.py').read_text(encoding='utf-8')
for number,line in enumerate(model_source.splitlines(),1):
    if line.startswith('class ') or 'def forward(' in line:
        print(number,line.strip())
''')
    l.md(r'''
## 2 用带编号的张量理解多头换轴

原项目把所有头的投影合在一次 Linear 中。投影后的 C=H×D 先拆成 `(B,T,H,D)`，再交换 T、H 成 `(B,H,T,D)`。矩阵乘法会分别处理每个 B、H。

下面填入编号，不进行训练，只检查换轴前后“哪个数属于哪个位置、哪一头”。
''')
    l.code(r'''
B,T,H,D=1,3,2,2
numbered=torch.arange(B*T*H*D).reshape(B,T,H,D)
heads=numbered.transpose(1,2)
print('原本B,T,H,D：\n',numbered)
print('变成B,H,T,D：\n',heads)
restored=heads.transpose(1,2).contiguous().view(B,T,H*D)
check('换回来没有丢数',restored,numbered.reshape(B,T,H*D))
''')
    l.ex('m01','取第二头的第二个位置','从 heads 取 batch 0、head 1、time 1 的向量。',
         'ex_m01_vector=None', 'heads 的轴顺序已经是 B,H,T,D。',
         'ex_m01_vector=heads[0,1,1]', "check('头与位置对应',ex_m01_vector,[6,7])")
    l.md('`contiguous()` 按新的逻辑顺序准备连续存储，接着 `view(B,T,C)` 合并头与头内特征。`transpose` 改的是轴的解释，不能用直接 reshape 来替代这次换轴。')
    l.ex('m02','合并所有头','把 heads 换回 B,T,H,D，再合并最后两个轴。',
         'ex_m02_joined=None','先 transpose(1,2)，再 contiguous().view(...)。',
         'ex_m02_joined=heads.transpose(1,2).contiguous().view(B,T,H*D)',
         "check('合并头',ex_m02_joined,restored)")
    l.md('## 3 用原项目创建一个小模型\n\n配置里选择一种方式指定结构：这里直接给层数、头数、宽度，所以 model_type 设为 None。将 Dropout 设为 0，以便做数值对照。')
    l.code(r'''
import sys
sys.path.insert(0,str(reference))
from mingpt.model import GPT
configuration=GPT.get_default_config()
configuration.model_type=None
configuration.n_layer,configuration.n_head,configuration.n_embd=1,2,16
configuration.vocab_size,configuration.block_size=7,8
configuration.embd_pdrop=configuration.resid_pdrop=configuration.attn_pdrop=0.
upstream=GPT(configuration)
''')
    l.code(r'''
input_ids=torch.tensor([[0,1,2,3],[3,2,1,0]])
target_ids=torch.tensor([[1,2,3,4],[2,1,0,1]])
upstream_logits,upstream_loss=upstream(input_ids,target_ids)
print('输出：',upstream_logits.shape,'loss：',upstream_loss.item())
check('原项目交叉熵',upstream_loss,F.cross_entropy(upstream_logits.reshape(-1,7),target_ids.reshape(-1)))
''')
    l.ex('m03','从 logits 得到概率','对 upstream_logits 的最后一个轴做 Softmax。',
         'ex_m03_probs=None','最后一轴是词表。',
         'ex_m03_probs=torch.softmax(upstream_logits,dim=-1)',
         "check('每位置概率和',None if ex_m03_probs is None else ex_m03_probs.sum(dim=-1),torch.ones(2,4))")
    l.md('## 4 从结果判断模型有没有读到未来\n\n三角形 mask 写在源码里，还可以从模型的实际输出验证它。下面只改变最后一个输入位置；第0–2位置面对的过去没有变，它们的输出也应相同。这个小干预把“因果”变成一个可以观察的结果。')
    l.code(r'''
upstream.eval()
original,_=upstream(input_ids)
perturbed=input_ids.clone(); perturbed[:,-1]=6
after,_=upstream(perturbed)
check('真实minGPT的因果性',after[:,:-1],original[:,:-1])
''')
    l.ex('m04','挑选可比较的位置','比较最后一个输入之前的全部 logits。返回它们的最大绝对差。',
         'ex_m04_difference=None', '用 after[:,:-1]-original[:,:-1]，再 abs().max()。',
         'ex_m04_difference=(after[:,:-1]-original[:,:-1]).abs().max()',
         "check('前缀分数不受未来影响',ex_m04_difference,0.)")
    l.md('## 5 打开训练器后，自己写出它的一步\n\n原项目 Trainer 的核心仍是前向、清梯度、反传、更新。这里用同一小 batch 重复 20 步，观察实际损失；原项目还在这几步外围添加 DataLoader、回调和梯度裁剪。')
    l.code(r'''
upstream.train()
upstream_optimizer=torch.optim.AdamW(upstream.parameters(),lr=.01)
upstream_history=[]
for step in range(20):
    _,loss=upstream(input_ids,target_ids)
    upstream_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    upstream_optimizer.step()
    upstream_history.append(loss.item())
print('同一batch的loss：',upstream_history[0],'->',upstream_history[-1])
''')
    l.ex('m05','重新算训练目标损失','用原项目返回的 logits 自己计算一次平均交叉熵。',
         'ex_m05_loss=None', '合并batch、位置，保留7个词表分数。',
         'ex_m05_loss=F.cross_entropy(upstream_logits.reshape(-1,7),target_ids.reshape(-1))',
         "check('交叉熵接口一致',ex_m05_loss,upstream_loss)")
    l.md('## 6 观察中间层，再提出可检验的问题\n\nForward hook 在指定模块计算后保存输出。先核对这个输出的形状；随后才能研究某层保留了什么信息。hook 的三项参数是模块、输入元组、输出。')
    l.code(r'''
cache={}
def record_hidden(module,inputs,output):
    cache['hidden']=output.detach()
handle=upstream.transformer.h[0].register_forward_hook(record_hidden)
upstream.eval()
_ = upstream(input_ids)
handle.remove()
print('Block输出：',cache['hidden'].shape)
''')
    l.ex('m06','把隐藏状态投影到词表','使用最终归一化与输出投影，把 cache 中的状态转成词表分数。',
         'ex_m06_lens=None','依次使用 upstream.transformer.ln_f 和 upstream.lm_head。',
         "ex_m06_lens=upstream.lm_head(upstream.transformer.ln_f(cache['hidden']))",
         "check('隐藏状态投影形状',None if ex_m06_lens is None else ex_m06_lens.shape,(2,4,7))")
    l.md('## 接到 NTU 作业\n\nHW01 的 token、prompt 和多轮对话建立在模型接口之上；先用本课的整数输入把接口走通。HW03 的 Logit Lens 与 Function Vector 涉及中间表示和干预：可先在第14课训练的模型上记录隐藏状态，固定输入、只改变一层的某个向量，再观察损失变化。\n\n独立任务：在 minGPT 找出 mask、LayerNorm、两次残差、目标错位、训练器更新五处源码；每处写一段数值检查。只训练同一小batch测的是拟合能力；再构造独立序列检验泛化。')
    l.reading('minGPT 原项目','https://github.com/karpathy/minGPT',
              '依次读 model.py 的 CausalSelfAttention、Block、GPT.forward，再读 trainer.py 的 run。比较这里逐头实现与原项目合并投影实现，哪些是数学结构，哪些是计算效率选择？')
    lessons.append(l)
    return lessons
