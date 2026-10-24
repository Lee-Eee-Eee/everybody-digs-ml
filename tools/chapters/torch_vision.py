"""Five independent lessons: PyTorch, training, convolution, and residuals."""
from coursekit import Lesson


def section(lesson, title, explanation, source):
    lesson.md('## ' + title + '\n\n' + explanation)
    lesson.code(source)


def tensors():
    L = Lesson('05_torch_tensors_autograd', '05｜从数组到可求导的计算图',
               ['判断 Tensor 的形状、类型、存储与梯度归属',
                '沿着分支计算图手算并验证输入、权重的梯度',
                '独立完成一次反向传播与参数更新'],
               '已会 NumPy 数组、矩阵乘法和标量链式法则')
    L.md(r'''
    这次让程序接手一件已经能手算的事。一个小模型输出 $q=xw+x$，输入给定为 $x=2$，
    权重暂取 $w=3$，于是输出是8。把目标定为0，用 $L=\tfrac12q^2$ 记录误差，
    当前损失就是32。调整哪一个数、朝哪个方向走，才能让这个误差下降？

    后面会亲手算出 $\partial L/\partial w=16$，并让 PyTorch 得到同一结果。
    输入 x 还被用了两次，它的梯度需要收齐两条路径。这个例子足够小，程序的每一个答案都有地方核对。
    ''')
    section(L, '1. 同一组数，多了哪些信息', r'''
    输入、权重和中间结果都要存起来。NumPy 用数组，PyTorch 用 Tensor，标量也是一个零维 Tensor。
    下格的2×2表还没有求导，只用来认清一份数据随身带着的信息：`shape` 是形状，
    `dtype` 是数值类型，`device` 是存放数据的设备。输出中的 `cpu` 对应这次计算的位置。

    训练小网络使用 `float32`；本章核对微小扰动时改用舍入误差更小的 `float64`。
    `.to(dtype=..., device=...)` 返回所需类型、设备上的 Tensor。类别编号会用整数类型 `torch.long`，
    它和这些可微的浮点数承担不同的角色。
    ''', r'''
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    torch.manual_seed(5)
    style()
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    print(a, '\nshape:', tuple(a.shape), 'dtype:', a.dtype, 'device:', a.device)
    print('转换类型:', a.to(dtype=torch.float64, device='cpu').dtype)
    ''')
    section(L, '2. 改了数组，Tensor 也跟着变了', r'''
    同一份 NumPy 数据用两种方式交给 PyTorch，随后把原数组首项从1改成9。
    `shared` 会显示9，`copied` 仍显示1：`from_numpy` 与原数组共用 CPU 内存，
    `torch.tensor` 另外复制了一份。两个变量名字不同，并不意味着它们背后存着两份数。

    保存一份将来还要对照的 Tensor 时，用 `.clone()` 取得独立副本。
    后面比较更新前后的参数，会再次用到这件事。
    ''', r'''
    original = np.array([1., 2., 3.], dtype=np.float32)
    shared = torch.from_numpy(original)
    copied = torch.tensor(original)
    original[0] = 9.
    print('NumPy:', original)
    print('from_numpy:', shared, 'tensor:', copied)
    ''')
    L.ex('5.1', '类型与独立副本',
         '把整数列表转成 float64 Tensor，复制后把副本首项改为 8；返回原 Tensor。',
         r'''ex_t51_result = None  # TODO：建立 Tensor，并修改其独立副本''',
         '`torch.tensor(..., dtype=...)` 建立数据；用 `.clone()` 得到另一份存储。',
         r'''
         ex_t51_result = torch.tensor([1, 2, 3], dtype=torch.float64)
         ex_t51_copy = ex_t51_result.clone()
         ex_t51_copy[0] = 8.
         ''', r'''
         check('原数据保持独立', ex_t51_result, [1., 2., 3.])
         check('数据类型', None if ex_t51_result is None else str(ex_t51_result.dtype), 'torch.float64')
         ''')
    section(L, '3. shape、索引与广播', r'''
    把两份样本排在一起，每份含三项数值，就得到下面的2×3表。
    记样本数为 $B$、每份样本的特征数为 $D$，于是 `X.shape=(B,D)`。
    `X[0]` 取出第一份样本 `[1,2,3]`，`X[:,1]` 取出所有样本的第二项 `[2,5]`。

    如果每份样本都要加同一组偏置 `[10,20,30]`，直接写 `X+b` 即可。
    PyTorch 从末维对齐形状，将长度3的 b 用于每一行；这叫广播。
    输出第二行是 `[14,25,36]`，可以逐项检查对应关系。
    ''', r'''
    X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    b = torch.tensor([10., 20., 30.])
    print('B,D =', X.shape)
    print('第一个样本:', X[0], '第二列:', X[:, 1])
    print('广播相加:\n', X + b)
    ''')
    section(L, '4. reshape 与 transpose 改变的东西', r'''
    把 `[0,1,2,3,4,5]` 排成2×3表，第一行是 `[0,1,2]`。
    转置后第一行却是 `[0,3]`；沿新行读取，得到的次序为 `[0,3,1,4,2,5]`。
    `reshape` 重新安排形状，`transpose` 交换轴，二者在这组数上做的事情已经不同。

    `stride()` 表示沿某个轴走一格，需要在底层存储中跨过多少个元素。
    转置把步幅从 `(3,1)` 变为 `(1,3)`，通常仍引用原存储。
    `.contiguous()` 按新的逻辑次序准备连续存储，接着 `.view(-1)` 才能直接读成一行。
    `.reshape()` 也能完成变形，必要时会自行复制。
    ''', r'''
    grid = torch.arange(6).reshape(2, 3)
    transposed = grid.T
    print('原矩阵:\n', grid, '\n转置:\n', transposed)
    print('步幅:', grid.stride(), transposed.stride())
    print('连续:', grid.is_contiguous(), transposed.is_contiguous())
    print('转置后展平:', transposed.contiguous().view(-1))
    ''')
    L.ex('5.2', '从样本矩阵得到线性输出',
         r'给定 $X\in\mathbb R^{2\times3}$、$w\in\mathbb R^3$、标量 $b=1$，算 $Xw+b$，输出应为形状 `(2,)`。',
         r'''
         ex_t52_X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
         ex_t52_w = torch.tensor([1., 0., -1.])
         ex_t52_result = None
         ''', '矩阵乘法使用 `@`；标量偏置会广播到两个样本。',
         r'''
         ex_t52_X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
         ex_t52_w = torch.tensor([1., 0., -1.])
         ex_t52_result = ex_t52_X @ ex_t52_w + 1.
         ''', r'''check('线性输出', ex_t52_result, [-1., -1.])''')
    section(L, '5. 把开头的小模型交给 PyTorch', r'''
    现在计算 $a=xw$、$q=a+x$、$L=\tfrac12q^2$，输出依次为6、8、32。
    建立 x、w 时设置 `requires_grad=True`，PyTorch 就会记录后续运算怎样依赖它们。
    x、w 是直接建立的叶节点，a、q、loss 是运算得到的中间结果。

    `a.retain_grad()` 让中间结果 a 也保留收到的梯度，便于沿图核对。
    输入 x 一路进入乘法，一路直接进入加法；这两处使用必须都留在记录里。
    ''', r'''
    x = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    w = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    a = x * w
    a.retain_grad()
    q = a + x
    loss = 0.5 * q.square()
    print('x → a=xw → q=a+x → L:', x.item(), a.item(), q.item(), loss.item())
    ''')
    L.md(r'''
    图上的每个框都对应刚才的一项数值。顺着箭头可以从2和3算到32，
    反过来则要追踪：末端的损失变化率沿哪条依赖关系传到 x、w。
    x 通往 q 的两条路线在图上都看得见。
    ''')
    L.code(r'''
    fig, ax = plt.subplots(figsize=(8, 2.7))
    positions = {'x': (.08, .7), 'w': (.08, .15), 'a': (.37, .42), 'q': (.67, .42), 'L': (.94, .42)}
    labels = {'x': 'x = 2', 'w': 'w = 3', 'a': 'a = xw\n6', 'q': 'q = a + x\n8', 'L': 'L = q² / 2\n32'}
    for key, (px, py) in positions.items():
        ax.text(px, py, labels[key], ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=.45', fc='white', ec='tab:blue'))
    for source, target in [('x', 'a'), ('w', 'a'), ('a', 'q'), ('q', 'L'), ('x', 'q')]:
        ax.annotate('', xy=positions[target], xytext=positions[source],
                    arrowprops=dict(arrowstyle='->', color='tab:blue', shrinkA=30, shrinkB=34,
                                    connectionstyle='arc3,rad=' + ('-.45' if (source, target) == ('x', 'q') else '0')))
    ax.set(xlim=(-.05, 1.08), ylim=(-.1, 1.12)); ax.axis('off'); show(fig)
    ''')
    section(L, '6. 逐边相乘，在分叉处相加', r'''
    输出 q 增加一个很小的量时，损失约增加它的8倍，即 $\partial L/\partial q=q=8$。
    沿乘法这一路，$\partial q/\partial a=1$，
    $\partial a/\partial x=w=3$。乘法分支给 $x$ 的贡献为 $8\times1\times3=24$；
    直接分支给出 $8\times1=8$，所以 $\partial L/\partial x=32$。
    同理 $\partial L/\partial w=8\times x=16$。

    32描述改变输入的影响，16描述改变权重的影响。训练时更新 w，使用的就是后者。
    `.backward()` 把这些导数算到相关叶节点的 `.grad` 中；中间节点 a 的梯度由上一格的
    `.retain_grad()` 留下来，数值应为8。
    ''', r'''
    loss.backward()
    print('乘法路径:', q.item() * w.item(), '直接路径:', q.item())
    print('x.grad:', x.grad.item(), 'w.grad:', w.grad.item(), 'a.grad:', a.grad.item())
    check('输入两路累加', x.grad, 32.)
    check('权重梯度', w.grad, 16.)
    ''')
    L.ex('5.3', '另一张分支图',
         r'新建标量 $u=2$，计算 $J=u^2+3u$，反传后取 `u.grad`。先手算再运行。',
         r'''ex_t53_result = None''', r'$d(u^2)/du=2u$，$d(3u)/du=3$，相同叶节点处相加。',
         r'''
         ex_t53_u = torch.tensor(2., requires_grad=True)
         ex_t53_loss = ex_t53_u.square() + 3 * ex_t53_u
         ex_t53_loss.backward()
         ex_t53_result = ex_t53_u.grad.clone()
         ''', r'''check('分支梯度', ex_t53_result, 7.)''')
    section(L, '7. 同一个导数，第二次却得到14', r'''
    对 $u^2+3u$ 在 $u=2$ 处求导，答案是7。下格重新计算这个损失并反传两次，
    `.grad` 却依次显示7、14。参数与公式都没变，变化来自 `.grad` 会收集并累加新的贡献。

    令 `u.grad=None` 后重算一次，结果回到7。普通训练希望每一步使用当前这一批数据的梯度，
    就在这一轮反传前清掉旧值。优化器的 `zero_grad(set_to_none=True)` 做同样的清理。
    ''', r'''
    u = torch.tensor(2., requires_grad=True)
    (u.square() + 3 * u).backward()
    print('第一次:', u.grad.item())
    (u.square() + 3 * u).backward()
    print('第二次累加:', u.grad.item())
    u.grad = None
    (u.square() + 3 * u).backward()
    print('清零后:', u.grad.item())
    ''')
    section(L, '8. 用有限差分从外部核对', r'''
    刚才的32和16还可以从输入输出的变化得到。将 x 从2分别改为 $2+\varepsilon$、
    $2-\varepsilon$，两次损失之差除以输入间隔 $2\varepsilon$，得到中心差分：
    $$f'(x)\approx\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}.$$
    它只调用正向函数，因此可以独立核对沿图求导的结果。下格也用同样方法扰动 w。
    $\varepsilon$ 过大会有截断误差，过小会有相减消去误差；这里取 $10^{-5}$、float64。
    ''', r'''
    def scalar_loss(x_value, w_value):
        q_value = x_value * w_value + x_value
        return 0.5 * q_value ** 2
    eps = 1e-5
    fd_x = (scalar_loss(2 + eps, 3) - scalar_loss(2 - eps, 3)) / (2 * eps)
    fd_w = (scalar_loss(2, 3 + eps) - scalar_loss(2, 3 - eps)) / (2 * eps)
    print('差分:', fd_x, fd_w)
    check('差分与反传', [fd_x, fd_w], [x.grad.item(), w.grad.item()])
    ''')
    L.ex('5.4', '两次反传只保留第二次',
         r'令 $u=3$。先对 $u^2$ 反传，再清空梯度，对 $4u$ 重新反传，返回最终梯度。',
         r'''ex_t54_result = None''', '每次重新写出损失；在第二个损失前设置 `u.grad=None`。',
         r'''
         ex_t54_u = torch.tensor(3., requires_grad=True)
         ex_t54_u.square().backward()
         ex_t54_u.grad = None
         (4 * ex_t54_u).backward()
         ex_t54_result = ex_t54_u.grad.clone()
         ''', r'''check('第二次独立梯度', ex_t54_result, 4.)''')
    section(L, '9. 一批样本的梯度怎样合起来', r'''
    把刚才的单个输入换成三组数据：输入 `[1,2,3]`，目标 `[2,4,6]`。
    同一组 w、b 为全部样本计算 $\hat y_i=wx_i+b$。用一批 $B$ 个样本的平均损失
    $$L=\frac1{2B}\sum_{i=1}^B(\hat y_i-y_i)^2.$$
    每个样本的残差记为 $e_i=\hat y_i-y_i$。对 w 求导时，第 i 项留下 $e_ix_i$，
    对 b 求导留下 $e_i$，所以 $\partial L/\partial w=B^{-1}\sum_i e_ix_i$，
    $\partial L/\partial b=B^{-1}\sum_i e_i$。
    代码中的 `mean()` 除以 B；打印的“手算”和“反传”应逐项相等。
    ''', r'''
    bx = torch.tensor([1., 2., 3.])
    by = torch.tensor([2., 4., 6.])
    bw = torch.tensor(1., requires_grad=True)
    bb = torch.tensor(0., requires_grad=True)
    errors = bw * bx + bb - by
    batch_loss = 0.5 * errors.square().mean()
    batch_loss.backward()
    print('手算:', (errors.detach() * bx).mean(), errors.detach().mean())
    print('反传:', bw.grad, bb.grad)
    ''')
    L.ex('5.5', '手写批梯度',
         r'$x=[1,2],y=[3,5],w=1,b=0$。采用上一节的 $1/(2B)$ 损失，返回 `[dL/dw,dL/db]`。',
         r'''ex_t55_result = None''', '先算残差 `w*x+b-y`，然后分别求残差乘输入的均值、残差均值。',
         r'''
         ex_t55_x = torch.tensor([1., 2.])
         ex_t55_e = ex_t55_x - torch.tensor([3., 5.])
         ex_t55_result = torch.stack([(ex_t55_e * ex_t55_x).mean(), ex_t55_e.mean()])
         ''', r'''check('批平均梯度', ex_t55_result, [-4., -2.5])''')
    section(L, '10. 更新数值，然后重新计算图', r'''
    这组直线数据的初值是 w=1、b=0，预测偏低。得到的两个梯度都为负，
    按 $w\leftarrow w-\eta\,\partial L/\partial w$ 更新，w 就会增大，b 也会增大。
    下格取学习率 $\eta=0.1$，对照更新前后的平均损失，验证这一步是否沿着期望方向移动。

    数值修改放在 `torch.no_grad()` 中，下一轮用新参数重新计算损失。
    记录损失时，`.item()` 取出一个 Python 数字；记录整份数组时，常用 `.detach()`
    取出不再追踪梯度的 Tensor。
    ''', r'''
    before = batch_loss.item()
    with torch.no_grad():
        bw -= 0.1 * bw.grad
        bb -= 0.1 * bb.grad
    after = (0.5 * (bw * bx + bb - by).square().mean()).item()
    print('更新后的参数:', bw.item(), bb.item())
    print('更新前后损失:', before, after)
    ''')
    L.ex('5.6', '完成一次数值更新',
         r'参数 $w=2$，已有梯度 $g=3$，学习率 $\eta=0.1$。在 `no_grad` 中更新参数并返回它。',
         r'''ex_t56_result = None''', '参数仍设置 `requires_grad=True`，只在更新这一步关闭记录。',
         r'''
         ex_t56_w = torch.tensor(2., requires_grad=True)
         with torch.no_grad():
             ex_t56_w -= 0.1 * 3.
         ex_t56_result = ex_t56_w.detach().clone()
         ''', r'''check('梯度下降一步', ex_t56_result, 1.7)''')
    section(L, '11. 回到分支图，把一步更新放到曲线上', r'''
    这里回到开头的 $q=2w+2$，画出 $L=\tfrac12q^2$ 随 w 的变化。
    w=3处的切线斜率是16，梯度下降向左移动；步长过大时，会越过低点到曲线另一侧。
    导数给出当前位置附近的变化率，学习率决定实际走多远。
    ''', r'''
    weights = np.linspace(-2, 4, 200)
    values = scalar_loss(2., weights)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(weights, values, label='Loss')
    ax.plot(weights, scalar_loss(2., 3.) + 16 * (weights - 3.), '--', label='Tangent at w=3')
    ax.scatter([3.], [scalar_loss(2., 3.)], zorder=3)
    ax.set(xlabel='Weight w', ylabel='Loss', ylim=(-10, 60))
    ax.legend(); show(fig)
    ''')
    L.md(r'''
    [打开交互反向传播实验](../interactive/index.html#backprop)：用步进按钮展开反向计算，
    观察一个节点收到第二条路径时，梯度怎样在旧值上累加。把同一个节点的各路贡献加起来，
    与显示的总梯度核对；随后只改一个权重，看受影响的前向数值和反向贡献怎样一起变化。
    ''')
    L.reading('PyTorch：Autograd mechanics', 'https://docs.pytorch.org/docs/stable/notes/autograd',
              '阅读 “How autograd encodes the history”。在本课分支图中，找出叶节点、运算节点和累加位置。\n\n'
              '**单变量实验：**只改变学习率为 0.01、0.1、1，记录一次更新后的损失；用曲线解释变化。')
    return L


def modules():
    L = Lesson('06_modules_data_training', '06｜把参数、数据和训练循环接起来',
               ['写出可注册参数的 nn.Module 与 Dataset',
                '追踪一个 batch 从取样到损失的形状',
                '独立写出并解释清梯度、反传和更新三步'],
               '05：Tensor、计算图与梯度累加')
    L.md(r'''
    上一课已经能更新两个数。现在让同一个模型看完40组 $y=2x+1$ 数据，
    一次读取8组，看一批就更新一次，最终把整条直线学出来。
    w、b 必须在这些更新之间保留下来，输入和目标必须始终配成同一份样本。

    这会用到三个对象：模型保管参数并做预测，数据集按编号返回样本，
    训练循环把一批数据的误差变成一次更新。下面围绕这条直线把它们接起来。
    ''')
    section(L, '1. class 把状态与操作放到一起', r'''
    模型每处理一批新输入，都要使用当前的 w、b。把它们存在 `self.w`、`self.b` 中，
    就能由同一个对象持续保管；`self` 指向当前这一份模型。
    `__init__` 建立这份状态，`forward` 描述拿到输入后怎样得到预测。

    `ScalarLine(nn.Module)` 继承 PyTorch 的模型接口。
    `super().__init__()` 初始化父类维护的参数记录，随后赋给属性的 `nn.Parameter`
    就能被登记。调用 `model(x)` 时，PyTorch 会执行这份模型的 `forward(x)`。
    ''', r'''
    import numpy as np
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    torch.set_num_threads(1); torch.manual_seed(6); style()
    class ScalarLine(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(0.))
            self.b = nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            return self.w * x + self.b
    ''')
    section(L, '2. 哪些数会交给优化器', r'''
    一个模型属性可以存很多数，优化器需要知道其中哪些要学习。
    `nn.Parameter` 让 w、b 出现在 `model.parameters()` 中；优化器稍后接收的正是这份列表。
    单纯给对象加一个 Tensor 属性，不会把它加入可学习参数列表。

    下格特意放入 `note` 和 buffer `scale`，对照两种输出：参数名单只有 w、b，
    `state_dict()` 还保存 scale。buffer 适合随模型保存、迁移的固定状态。
    这里的 forward 只读取 w、b，初值都是0，因此两个输入对应的预测也都是0。
    ''', r'''
    model = ScalarLine()
    model.note = torch.tensor(7.)
    model.register_buffer('scale', torch.tensor(1.))
    print('可学习参数:', [(name, tuple(p.shape)) for name, p in model.named_parameters()])
    print('保存的状态:', list(model.state_dict()))
    print('模型输出:', model(torch.tensor([1., 2.])))
    ''')
    L.ex('6.1', '判断参数登记',
         '写一个仅含可学习标量 `gain=2` 和普通 Tensor `offset=1` 的模块，返回可学习参数总元素数。',
         r'''ex_t61_result = None''', '继承 `nn.Module`，初始化父类，然后只给 `gain` 包装 `nn.Parameter`。',
         r'''
         class ExT61Model(nn.Module):
             def __init__(self):
                 super().__init__()
                 self.gain = nn.Parameter(torch.tensor(2.))
                 self.offset = torch.tensor(1.)
         ex_t61_result = sum(p.numel() for p in ExT61Model().parameters())
         ''', r'''check('登记的参数元素数', ex_t61_result, 1)''')
    section(L, '3. Dataset：输入与目标按编号配对', r'''
    对直线数据，输入2的目标应该是5。只要读错配对关系，模型就会学另一件事。
    `Dataset` 将取样过程放在 `__getitem__` 中：同一个 index 同时取 x 和 y。
    Python 的 `dataset[2]` 会调用这个方法，`len(dataset)` 会调用 `__len__`。

    `LineDataset` 只保存已有 Tensor，并按行返回。下面建立8份样本，
    打印编号2时应看到 `(tensor([2.]), tensor([5.]))`。
    ''', r'''
    from torch.utils.data import Dataset, DataLoader
    class LineDataset(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __len__(self):
            return len(self.x)
        def __getitem__(self, index):
            return self.x[index], self.y[index]
    x_data = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    dataset = LineDataset(x_data, 2 * x_data + 1)
    print('总数:', len(dataset), '编号2:', dataset[2])
    ''')
    section(L, '4. DataLoader：四个样本成为一批', r'''
    一次处理四份输入，可以把四次预测合为一次张量运算。
    DataLoader 从 Dataset 取出样本，将四个 `(1,)` 输入堆成 `(4,1)`，目标也排成 `(4,1)`。
    `iter(loader)` 建立迭代器，`next(...)` 就取出当前的下一批。

    `shuffle=True` 改变每轮的取样顺序，固定生成器种子可复现这次顺序。
    打印出的输入虽然经过打乱，每一列对应的目标仍满足 $y=2x+1$。
    `num_workers=0` 表示由当前进程读取这组小数据。
    ''', r'''
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(6), num_workers=0)
    batch_x, batch_y = next(iter(loader))
    print('输入:', batch_x.T, '目标:', batch_y.T)
    print('batch形状:', tuple(batch_x.shape), tuple(batch_y.shape))
    print('逐个样本编号:', list(range(len(dataset))))
    ''')
    L.ex('6.2', '最后一批有几个样本',
         '把 10 个样本装入 batch_size=4、shuffle=False 的 DataLoader，返回每批样本数。',
         r'''ex_t62_result = None''', '`len(batch)` 读出这一批的第一维；默认保留最后不足一批的数据。',
         r'''
         ex_t62_data = torch.arange(10)
         ex_t62_loader = DataLoader(ex_t62_data, batch_size=4, shuffle=False)
         ex_t62_result = [len(batch) for batch in ex_t62_loader]
         ''', r'''check('三批的大小', ex_t62_result, [4, 4, 2])''')
    section(L, '5. nn.Linear 的行列约定', r'''
    已经有了一批输入，预测仍然是乘权重、加偏置。`nn.Linear` 将这组运算和参数封装在一起。
    输入 `X` 为 $(B,D_{in})$，`nn.Linear(Din,Dout)` 的 `weight` 为
    $(D_{out},D_{in})$，`bias` 为 $(D_{out},)$。它计算
    $$\hat Y=XW^\top+b\quad\in\mathbb R^{B\times D_{out}}.$$
    本例 $D_{in}=D_{out}=1$。把权重设为2、偏置设为1，预测就应逐项等于 batch_y。
    输出与目标都为 `(B,1)`，相减时每个位置对应同一份样本。
    ''', r'''
    affine = nn.Linear(1, 1)
    with torch.no_grad():
        affine.weight.fill_(2.)
        affine.bias.fill_(1.)
    direct = batch_x @ affine.weight.T + affine.bias
    print('W,b:', affine.weight.shape, affine.bias.shape)
    print('预测与目标:', affine(batch_x).shape, batch_y.shape)
    check('Linear等于矩阵运算', affine(batch_x), direct)
    ''')
    section(L, '6. 同一起点比较手动 SGD 与优化器', r'''
    优化器即将替代手写的 `parameter -= lr * parameter.grad`。
    用两份从相同参数出发的模型，各走一步，就能检查它实际做了什么。
    两次 `nn.Linear` 会各自抽取随机初值，`load_state_dict` 将第二份参数改成第一份的数值。

    两边共同使用输入 `[1,2]`、目标 `[3,5]` 和学习率0.1。
    `nn.MSELoss()` 对应 $L=B^{-1}\sum_i(\hat y_i-y_i)^2$；残差对预测求导时得到 $2e_i/B$。
    ''', r'''
    manual = nn.Linear(1, 1)
    automatic = nn.Linear(1, 1)
    automatic.load_state_dict(manual.state_dict())
    step_x = torch.tensor([[1.], [2.]])
    step_y = torch.tensor([[3.], [5.]])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(automatic.parameters(), lr=0.1)
    print('相同起点:', manual.state_dict(), automatic.state_dict())
    ''')
    L.ex('6.3', '自己完成矩阵正向',
         r'输入 $(2,2)$、权重 $(1,2)$、偏置 $(1,)$，按 $XW^\top+b$ 计算。',
         r'''
         ex_t63_X = torch.tensor([[1., 2.], [3., 4.]])
         ex_t63_W = torch.tensor([[2., -1.]])
         ex_t63_b = torch.tensor([0.5])
         ex_t63_result = None
         ''', '`W.T` 先把输出优先的权重矩阵转置。',
         r'''
         ex_t63_X = torch.tensor([[1., 2.], [3., 4.]])
         ex_t63_W = torch.tensor([[2., -1.]])
         ex_t63_b = torch.tensor([0.5])
         ex_t63_result = ex_t63_X @ ex_t63_W.T + ex_t63_b
         ''', r'''check('矩阵正向', ex_t63_result, [[0.5], [2.5]])''')
    section(L, '7. 看优化器能否复现手写的一步', r'''
    第一份模型清除旧梯度、计算损失并反传，随后逐个修改参数。
    第二份模型经历同样的正向和反向计算，由 `optimizer.step()` 读取 `.grad` 并更新。
    最后两项检查分别比较 weight、bias，二者都应相等。

    `zero_grad` 准备这一批的梯度收集位置，`backward` 填入梯度，`step` 使用它们。
    这三个调用对应的是三个不同动作，后面的多批训练只会重复这段过程。
    ''', r'''
    manual.zero_grad(set_to_none=True)
    manual_loss = loss_fn(manual(step_x), step_y)
    manual_loss.backward()
    with torch.no_grad():
        for parameter in manual.parameters():
            parameter -= 0.1 * parameter.grad
    optimizer.zero_grad(set_to_none=True)
    loss_fn(automatic(step_x), step_y).backward()
    optimizer.step()
    for name, parameter in manual.named_parameters():
        check('SGD一致：' + name, parameter, automatic.state_dict()[name])
    ''')
    section(L, '8. train、eval 与 no_grad 分别控制什么', r'''
    预测新样本时，通常只需要数值。下格把模型切到 `eval()` 后，输出的
    `requires_grad` 仍为 True；进入 `torch.no_grad()` 后，同一预测才不再记录求导过程。

    `.train()` 与 `.eval()` 切换模块运行模式，Dropout、BatchNorm 会据此改变行为，
    本例的 Linear 在两种模式下使用同一公式。`no_grad()` 控制这次运算是否记录梯度，
    所以评估循环会同时使用 `eval()` 和 `no_grad()`。
    ''', r'''
    automatic.eval()
    prediction_with_graph = automatic(step_x)
    with torch.no_grad():
        prediction_without_graph = automatic(step_x)
    print('eval仍求导:', prediction_with_graph.requires_grad)
    print('no_grad:', prediction_without_graph.requires_grad)
    print('模型training标记:', automatic.training)
    automatic.train()
    ''')
    L.ex('6.4', 'MSE 的批平均梯度',
         r'预测 `[1,2]`，目标 `[3,3]`。用 `nn.MSELoss` 反传，返回预测的梯度。',
         r'''ex_t64_result = None''', r'对两个元素平均：每项导数是 $2(\hat y_i-y_i)/2$。',
         r'''
         ex_t64_pred = torch.tensor([1., 2.], requires_grad=True)
         nn.MSELoss()(ex_t64_pred, torch.tensor([3., 3.])).backward()
         ex_t64_result = ex_t64_pred.grad.clone()
         ''', r'''check('MSE预测梯度', ex_t64_result, [-2., -1.])''')
    section(L, '9. 逐批训练，逐轮记录', r'''
    一步更新已经核对过，现在把它放回本章的40组数据中。
    输入均匀分布在 $[-1,1]$，目标为 $y=2x+1$；每批8个样本，一轮需要5次更新。
    完整遍历一遍训练集称为一轮 epoch。

    下面建立数据读取器、模型和优化器。它们都放在训练循环外：
    这样下一批读到新样本时，模型仍保存上一批刚学到的参数。
    ''', r'''
    train_x = torch.linspace(-1, 1, 40).reshape(-1, 1)
    train_y = 2 * train_x + 1
    train_loader = DataLoader(LineDataset(train_x, train_y), batch_size=8,
                              shuffle=True, generator=torch.Generator().manual_seed(6))
    trained = nn.Linear(1, 1)
    train_optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
    history = []
    ''')
    section(L, '10. 一份可以从头重写的完整循环', r'''
    `for inputs, targets in train_loader` 每次取得一批配对数据；
    预测与目标都为 `(B,1)`，相减、平方、平均后得到一个标量损失。
    `loss.backward()` 从这个标量反传，随后优化器更新现有模型。

    `loss.item()` 只取数值来记账，反传仍使用 loss Tensor。
    整轮记账时乘 `len(inputs)`，将批均值还原成这一批的损失和，最后除以数据集总数。
    即使末批不足8个样本，每个样本也会获得相同权重。
    ''', r'''
    for epoch in range(40):
        trained.train()
        total_loss = 0.
        for inputs, targets in train_loader:
            train_optimizer.zero_grad(set_to_none=True)
            predictions = trained(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            train_optimizer.step()
            total_loss += loss.item() * len(inputs)
        history.append(total_loss / len(train_loader.dataset))
    print('学习结果:', trained.weight.item(), trained.bias.item(), '末轮MSE:', history[-1])
    ''')
    L.md('输出中的两个参数应分别接近2和1。它们与末轮 MSE 一起说明：这份持续更新的模型已经从各批数据中恢复了同一条直线。')
    L.ex('6.5', '不等大小 batch 的损失',
         '两个 batch 分别有 4、2 个样本，批平均损失分别是 1、4，返回整组样本的平均损失。',
         r'''ex_t65_result = None''', '每批均值乘这一批样本数，合计后除以 6。',
         r'''ex_t65_result = (4 * 1. + 2 * 4.) / 6''',
         r'''check('整轮加权平均', ex_t65_result, 2.)''')
    section(L, '11. 评估与画图', r'''
    左图是每轮训练损失，右图把最终预测与40个目标点画到同一坐标中。
    参数接近2、1时，预测线应该穿过这些点；如果损失下降而图中配对错乱，
    就要回到输入、目标的行号检查。这里的评估使用刚才那一份 trained 模型，
    参数会从训练阶段一直保留到画图阶段。
    ''', r'''
    trained.eval()
    with torch.no_grad():
        fitted = trained(train_x)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(history); axes[0].set(xlabel='Epoch', ylabel='Mean squared error')
    axes[1].scatter(train_x[:, 0], train_y[:, 0], s=14, label='Data')
    axes[1].plot(train_x[:, 0], fitted[:, 0], color='tab:red', label='Learned line')
    axes[1].set(xlabel='Input x', ylabel='Target / prediction'); axes[1].legend()
    fig.tight_layout(); show(fig)
    ''')
    L.ex('6.6', '验证 eval 与求导记录',
         '建立 `nn.Linear(1,1)`，调用 `eval()`，在 `no_grad` 中预测。返回 `[model.training, output.requires_grad]`。',
         r'''ex_t66_result = None''', '这两个布尔值分别表示模块模式和这次输出是否记录求导。',
         r'''
         ex_t66_model = nn.Linear(1, 1)
         ex_t66_model.eval()
         with torch.no_grad():
             ex_t66_output = ex_t66_model(torch.ones(2, 1))
         ex_t66_result = [ex_t66_model.training, ex_t66_output.requires_grad]
         ''', r'''check('评估模式与记录状态', ex_t66_result, [False, False])''')
    L.reading('PyTorch：优化模型参数（官方源码）',
              'https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/optimization_tutorial.py',
              '找到训练函数的 `zero_grad`、`backward`、`step`，并追踪评估函数的 `no_grad`。'
              '指出评估集的样本在哪一行进入模型、是否会更新参数。\n\n'
              '**单变量实验：**只把 batch_size 从 8 改成 40，保留 epoch 数与学习率。比较更新次数及损失曲线。')
    return L


def mlps():
    L = Lesson('07_mlp_regression_classification', '07｜从一条直线到非线性回归与分类',
               ['把回归、分类的损失导数写成张量代码',
                '用训练集统计量标准化，并区分训练、验证、测试',
                '训练线性基线与 MLP，读懂拟合曲线和决策边界'],
               '06：模块、batch 与完整训练循环')
    L.md(r'''
    上一课的目标是一条直线，Linear 足以把它学好。这次让目标随着 x 上下弯曲：
    $$y=0.5x+0.8\sin(3x)+\text{噪声}.$$
    仍然训练一条直线，它会给出一条折中的斜线，曲折的部分留在残差里。
    增加训练步数只能调整这条直线的位置和斜率，模型需要有表达弯曲关系的能力。

    这就是本章 MLP 的用途。我们把直线模型与一个小型 MLP 放到同一份数据上，
    看它们学出的函数；随后用同样的训练过程处理两类二维点，观察输出和损失怎样改变。
    ''')
    section(L, '1. 生成一条带噪声的弯曲关系', r'''
    下格在 $[-2,2]$ 取240个输入点，按上面的函数生成目标，再加入标准差为0.08的高斯噪声。
    输入和目标都是一个连续数，这样的任务叫回归；每批使用 `(B,1)` 形状。
    `generator` 固定这次数据中的随机噪声，模型比较时会遇到同一组目标。
    ''', r'''
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    torch.set_num_threads(1); torch.manual_seed(7); style()
    generator = torch.Generator().manual_seed(7)
    rx = torch.linspace(-2, 2, 240).reshape(-1, 1)
    ry = 0.5 * rx + 0.8 * torch.sin(3 * rx) + 0.08 * torch.randn(240, 1, generator=generator)
    ''')
    section(L, '2. 留出尚未用于更新的样本', r'''
    训练时要反复调整参数，也要知道调整后的曲线在其他点上是否合适。
    这里随机分出160个训练点、40个验证点和40个测试点。
    训练点参与梯度更新，验证点用来选参数快照，最后用测试点计算误差。

    将输入写成 $\tilde x=(x-\mu_{train})/s_{train}$，可把训练输入移到均值0、
    标准差1附近。之后模型收到新输入，也沿用训练时定下的这一变换。
    `correction=0` 对应 $s_{train}^2=N^{-1}\sum_i(x_i-\mu_{train})^2$，
    `keepdim=True` 保留 `(1,1)`，让同一组统计量广播到每份输入。
    ''', r'''
    order = torch.randperm(len(rx), generator=generator)
    rtrain, rval, rtest = order[:160], order[160:200], order[200:]
    rmean = rx[rtrain].mean(dim=0, keepdim=True)
    rstd = rx[rtrain].std(dim=0, keepdim=True, correction=0)
    rxs = (rx - rmean) / rstd
    print('样本数:', len(rtrain), len(rval), len(rtest))
    print('训练均值、标准差:', rmean.item(), rstd.item())
    check('训练标准化均值', rxs[rtrain].mean(), 0.)
    ''')
    L.ex('7.1', '只用训练集标准化',
         '训练输入 `[1,3]`，测试输入 `[5]`。用训练总体均值和标准差变换测试输入。',
         r'''ex_t71_result = None''', '训练均值为2、总体标准差为1；不要把测试值加入统计。',
         r'''
         ex_t71_train = torch.tensor([1., 3.])
         ex_t71_test = torch.tensor([5.])
         ex_t71_result = (ex_t71_test - ex_t71_train.mean()) / ex_t71_train.std(correction=0)
         ''', r'''check('测试集沿用训练统计', ex_t71_result, [3.])''')
    section(L, '3. 线性模型的表达能力', r'''
    `nn.Linear(1,1)` 始终画出 $\hat y=wx+b$ 这条直线。
    即使相接两个无激活的线性层，展开后仍是仿射函数：
    $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$。
    中间加入 Tanh 后，每个隐藏单元会得到一条经过平滑弯折的响应曲线，
    输出层再将这些响应加权组合。这里的宽度16表示有16份这样的中间响应。
    下格打印出49个 MLP 参数，包含两层的权重与偏置。
    ''', r'''
    torch.manual_seed(7)
    regression_models = {
        'Linear': nn.Linear(1, 1),
        'MLP': nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)),
    }
    for name, net in regression_models.items():
        print(name, '输入输出:', (4, 1), tuple(net(rxs[:4]).shape),
              '参数量:', sum(p.numel() for p in net.parameters()))
    ''')
    section(L, '4. MSE 从目标写到导数', r'''
    用一个数评价整条预测曲线，可以将每个点的预测误差平方后平均：
    $$L=\frac1B\sum_i(\hat y_i-y_i)^2,\qquad
      \frac{\partial L}{\partial\hat y_i}=\frac2B(\hat y_i-y_i).$$
    一项误差为正，梯度下降就推动该预测减小；误差为负，推动预测增大。
    若 $\hat y=Xw+b$，继续乘上线性层的局部导数，得到 $\nabla_wL=X^\top[2(\hat y-y)/B]$，
    $\nabla_bL=\sum_i2(\hat y_i-y_i)/B$。
    下格两个残差是−1、2，平均平方误差为2.5，预测梯度为−1、2。
    `F.mse_loss` 默认对所有输出元素求均值，本课每个样本只有一个输出，正好对应这份推导。
    ''', r'''
    demo_pred = torch.tensor([[1.], [3.]], requires_grad=True)
    demo_y = torch.tensor([[2.], [1.]])
    mse = F.mse_loss(demo_pred, demo_y)
    mse.backward()
    analytic = 2 * (demo_pred.detach() - demo_y) / len(demo_y)
    print('MSE:', mse.item(), '预测梯度:', demo_pred.grad.T)
    check('MSE导数', demo_pred.grad, analytic)
    ''')
    L.ex('7.2', '数一遍 MLP 参数',
         '网络是 Linear(3,4)→Tanh→Linear(4,2)，返回可学习参数总数。也写出各层权重、偏置形状。',
         r'''ex_t72_result = None''', '第一层4×3+4，第二层2×4+2；Tanh无可学习参数。',
         r'''
         ex_t72_net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
         ex_t72_result = sum(p.numel() for p in ex_t72_net.parameters())
         print([(name, tuple(p.shape)) for name, p in ex_t72_net.named_parameters()])
         ''', r'''check('MLP参数量', ex_t72_result, 26)''')
    section(L, '5. Adam 怎样利用梯度历史', r'''
    训练 MLP 时，不同参数会收到不同大小、不断变化的梯度。
    这里采用 Adam：为每个参数保留梯度方向的平滑记录 m，以及梯度平方的平滑记录 v，
    用它们确定更新。令当前梯度为 $g_t$，从 $m_0=v_0=0$ 开始，更新
    $$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
      v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$
    初值为零会压低前几步的平均值；按已累计的权重做偏差校正，得到
    $\hat m_t=m_t/(1-\beta_1^t)$、$\hat v_t=v_t/(1-\beta_2^t)$，再更新
    $$\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.$$
    PyTorch 默认 $\beta_1=0.9,\beta_2=0.999,\varepsilon=10^{-8}$。
    下格只看第一步：梯度为0.2时，校正后的 m 是0.2、v 是0.04，
    归一后的更新量接近学习率0.01，参数从1变为约0.99。整网训练对每个参数重复这一计算。
    ''', r'''
    adam_parameter = nn.Parameter(torch.tensor(1., dtype=torch.float64))
    adam_optimizer = torch.optim.Adam([adam_parameter], lr=.01)
    adam_parameter.grad = torch.tensor(.2, dtype=torch.float64)
    first_m = (1 - .9) * .2
    first_v = (1 - .999) * .2 ** 2
    expected_adam = 1 - .01 * (first_m / (1 - .9)) / (np.sqrt(first_v / (1 - .999)) + 1e-8)
    adam_optimizer.step()
    check('Adam第一次更新', adam_parameter, expected_adam)
    ''')
    section(L, '6. 用验证集选择保存的参数', r'''
    训练开始后，参数每一步都在变化，其中某一步可能对留出的验证点拟合得最好。
    下面让两个模型各更新300次，每次使用全部160个训练点，随后计算40个验证点的 MSE。
    验证误差更低时，将当前参数存入 best_state，结束后恢复这份快照。

    `.clone()` 在这里有实际作用：快照要保存当时的数值，后续更新继续修改模型本身。
    训练循环还保留每一步的训练、验证损失，下一格会把它们画成曲线。
    ''', r'''
    regression_history = {}
    for name, net in regression_models.items():
        optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
        records, best_value, best_state = [], float('inf'), None
        for epoch in range(300):
            net.train(); optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(rxs[rtrain]), ry[rtrain])
            loss.backward(); optimizer.step()
            net.eval()
            with torch.no_grad():
                valid_loss = F.mse_loss(net(rxs[rval]), ry[rval]).item()
            records.append((loss.item(), valid_loss))
            if valid_loss < best_value:
                best_value = valid_loss
                best_state = {key: value.clone() for key, value in net.state_dict().items()}
        net.load_state_dict(best_state)
        regression_history[name] = np.asarray(records)
    ''')
    section(L, '7. 看误差，也看学出的函数', r'''
    左图比较验证 MSE，右图直接画最终预测函数。在这份固定种子和划分下，
    测试 MSE 约为线性模型0.29、MLP 0.02，图中可以找到误差差异的来源：
    直线穿过了数据的大致趋势，MLP 还能跟随中间的起伏。

    这组输出让表达能力有了具体含义。若直线的损失已稳定，再重复同一更新，
    仍只能得到另一条直线；增加非线性隐藏响应，才改变了可画出的函数形状。
    ''', r'''
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[1].scatter(rx[:, 0], ry[:, 0], s=9, alpha=.35, label='Data')
    for name, net in regression_models.items():
        with torch.no_grad():
            predictions = net(rxs)
            test_mse = F.mse_loss(predictions[rtest], ry[rtest]).item()
        print(name, '测试MSE:', round(test_mse, 5))
        axes[0].plot(regression_history[name][:, 1], label=name)
        axes[1].plot(rx[:, 0], predictions[:, 0], label=name)
    axes[0].set(xlabel='Step', ylabel='Validation MSE'); axes[0].legend()
    axes[1].set(xlabel='Input x', ylabel='Target / prediction'); axes[1].legend()
    fig.tight_layout(); show(fig)
    ''')
    L.ex('7.3', 'MSE 通过线性层回传',
         r'$X=[[1,2],[3,4]],w=[1,0],y=[2,2]$，无偏置，采用均方误差。手算返回 $\nabla_wL$。',
         r'''ex_t73_result = None''', r'先算预测 `[1,3]`，再用 `X.T @ (2*error/B)`。',
         r'''
         ex_t73_X = torch.tensor([[1., 2.], [3., 4.]])
         ex_t73_w = torch.tensor([1., 0.])
         ex_t73_error = ex_t73_X @ ex_t73_w - torch.tensor([2., 2.])
         ex_t73_result = ex_t73_X.T @ (2 * ex_t73_error / 2)
         ''', r'''check('回传到权重', ex_t73_result, [2., 2.])''')
    section(L, '8. 分类从 logits 得到概率', r'''
    把目标换成类别后，单个连续预测值就不够用了。
    网络为每个样本输出 $C$ 个实数 $z_1,\ldots,z_C$，每类一个分数，称为 logits。
    用 Softmax 将这组相对分数转换成总和为1的概率：
    $$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.$$
    下格的 `[1,2,-1]` 变成约 `[0.259,0.705,0.035]`，第二类概率最高。
    分母由全部类别共同决定，提高一类的分数也会降低其他类的概率。
    计算时同时减去最大 logit，分子分母都约去同一因子，概率保持相同，指数数值则更稳。
    ''', r'''
    logits_demo = torch.tensor([[1., 2., -1.]], dtype=torch.float64)
    shifted = logits_demo - logits_demo.max(dim=1, keepdim=True).values
    probabilities = shifted.exp() / shifted.exp().sum(dim=1, keepdim=True)
    print('logits:', logits_demo, '概率:', probabilities)
    check('概率归一', probabilities.sum(dim=1), [1.])
    check('手写与torch.softmax', probabilities, torch.softmax(logits_demo, dim=1))
    ''')
    section(L, '9. 将正确类别的概率写成损失', r'''
    如果真实类别是第二类，刚才的0.705就是模型分给正确答案的概率。
    希望这个概率增大，可以最小化它的负对数。
    真类别编号记为 $t$，用 $y_k=\mathbf1[k=t]$ 表示对应的 one-hot 向量，单样本交叉熵为
    $$L=-\log p_t=-z_t+\log\sum_j e^{z_j}.$$
    对 $z_k$ 求导：$-z_t$ 给出 $-y_k$；对数项给出
    $e^{z_k}/\sum_j e^{z_j}=p_k$，所以
    $\partial L/\partial z_k=p_k-y_k$。批平均后再除以 $B$。
    正确类的梯度为 $p_t-1<0$，其他类的梯度为正。
    沿 logits 的负梯度移动，方向就是提高正确类的分数、压低其余分数。
    `F.cross_entropy(logits, targets)` 把这些运算合在内部，输入直接用 logits，
    目标则是一批类别编号，形状 `(B,)`、类型 `long`。
    ''', r'''
    ce_logits = torch.tensor([[1., 2., -1.], [0., 1., 0.]], requires_grad=True)
    ce_targets = torch.tensor([1, 0], dtype=torch.long)
    ce = F.cross_entropy(ce_logits, ce_targets)
    ce.backward()
    ce_expected = (ce_logits.detach().softmax(dim=1) - F.one_hot(ce_targets, 3)) / 2
    print('CE:', ce.item(), '\nlogit梯度:\n', ce_logits.grad)
    check('批平均交叉熵梯度', ce_logits.grad, ce_expected)
    ''')
    L.ex('7.4', '相同 logits 的交叉熵',
         '两个样本、三个类别，所有 logits 都是0，真类别为 `[0,2]`。求批平均 CE，并返回 logits 的梯度。',
         r'''ex_t74_loss, ex_t74_grad = None, None''', r'每类概率为1/3；每行梯度为 `(p-one_hot)/2`。',
         r'''
         ex_t74_z = torch.zeros(2, 3, requires_grad=True)
         ex_t74_loss = F.cross_entropy(ex_t74_z, torch.tensor([0, 2]))
         ex_t74_loss.backward()
         ex_t74_grad = ex_t74_z.grad.clone()
         ''', r'''
         check('均匀概率CE', ex_t74_loss, np.log(3.))
         check('CE梯度', ex_t74_grad, [[-1/3, 1/6, 1/6], [1/6, 1/6, -1/3]])
         ''')
    section(L, '10. 二维分类：线性边界与非线性边界', r'''
    现在让两组相对的象限属于同一类：对二维点 $x=(x_1,x_2)$，
    同号的两个坐标归为类别1，异号归为类别0。这是连续样本上的 XOR 结构。
    用一条直线切开平面，总有相对的一组象限分不对。

    线性模型的两个 logits 相减仍是仿射函数，预测由 $z_1-z_0$ 的正负决定，
    因此类别边界仍是一条直线。下面的 MLP 使用16个 Tanh 隐藏响应，
    将输入变为新特征后再分类，允许边界弯曲。
    ''', r'''
    cg = torch.Generator().manual_seed(71)
    cx = torch.rand(320, 2, generator=cg) * 2 - 1
    cy = (cx[:, 0] * cx[:, 1] > 0).long()
    ci = torch.randperm(len(cx), generator=cg)
    ctrain, cval, ctest = ci[:220], ci[220:270], ci[270:]
    torch.manual_seed(71)
    classifiers = {'Linear': nn.Linear(2, 2),
                   'MLP': nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))}
    print('输入:', cx.shape, '标签:', cy.shape, cy.dtype)
    ''')
    section(L, '11. 把回归循环用于类别预测', r'''
    从清梯度到参数更新，这个循环与刚才的回归循环相同。
    变化发生在模型末端：输出是 `(B,2)` logits，交叉熵将它们与 `(B,)` 类别编号比较。
    `argmax(dim=1)` 取每行分数最大的类别，便得到可与标签直接比较的 `(B,)` 预测。

    两个模型各更新250次，并按同一验证集上的 CE 保存参数。
    测试准确率表示50个留出测试点中预测正确的比例；下一幅边界图会展示这些点落在哪里。
    ''', r'''
    for name, net in classifiers.items():
        optimizer = torch.optim.Adam(net.parameters(), lr=.03)
        best_value, best_state = float('inf'), None
        for epoch in range(250):
            net.train(); optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(cx[ctrain]), cy[ctrain])
            loss.backward(); optimizer.step()
            net.eval()
            with torch.no_grad():
                value = F.cross_entropy(net(cx[cval]), cy[cval]).item()
            if value < best_value:
                best_value = value
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        with torch.no_grad():
            accuracy = (net(cx[ctest]).argmax(1) == cy[ctest]).float().mean().item()
        print(name, '验证CE:', round(best_value, 4), '测试准确率:', round(accuracy, 3))
    ''')
    L.ex('7.5', '取出预测类别',
         'logits 是 `[[2,1,-1],[0,3,4]]`，真类别 `[0,1]`，返回分类准确率。',
         r'''ex_t75_result = None''', '`argmax(dim=1)` 后与目标逐项比较，转float求均值。',
         r'''
         ex_t75_z = torch.tensor([[2., 1., -1.], [0., 3., 4.]])
         ex_t75_targets = torch.tensor([0, 1])
         ex_t75_result = (ex_t75_z.argmax(1) == ex_t75_targets).float().mean()
         ''', r'''check('准确率', ex_t75_result, 0.5)''')
    section(L, '12. 画决策边界，检查模型学到的形状', r'''
    下格在平面上铺一张90×90网格，将坐标排成8100行二维输入。
    模型为每行预测类别，再将结果排回方格，背景色就显出了模型的分类区域。
    散点颜色是真实类别；散点与所在背景颜色不同，就是一个错例。

    同样的训练步骤，线性模型画出直线边界，MLP 则能分出相对的象限。
    把本章两幅结果图并排看，回归中的弯曲拟合和分类中的弯曲边界来自同一个改变：
    在两层线性变换之间加入非线性激活。
    ''', r'''
    axis = torch.linspace(-1.1, 1.1, 90)
    gx, gy = torch.meshgrid(axis, axis, indexing='xy')
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5))
    for ax, (name, net) in zip(axes, classifiers.items()):
        with torch.no_grad():
            labels = net(grid).argmax(1).reshape(gx.shape)
        ax.contourf(gx, gy, labels, levels=[-.5, .5, 1.5], cmap='coolwarm', alpha=.25)
        ax.scatter(cx[ctest, 0], cx[ctest, 1], c=cy[ctest], cmap='coolwarm', edgecolors='k', s=22)
        ax.set(title=name, xlabel='x1', ylabel='x2', aspect='equal')
    fig.tight_layout(); show(fig)
    ''')
    L.md(r'''
    [打开实时训练实验](../interactive/index.html#training)：保留 XOR 数据，把隐藏层数设为0，
    训练并观察边界；随后改为带 Tanh 的隐藏层，重新初始化并训练。
    比较验证点、边界形状和损失曲线。固定结构后，只改变学习率，观察同样20步内损失怎样移动。
    ''')
    L.ex('7.6', '验证集选择参数',
         '三个 epoch 的验证损失依次是 `[.7,.4,.5]`，相应参数快照是 `[1,2,3]`。返回应选的参数值。',
         r'''ex_t76_result = None''', '选择验证损失最小的下标，再从同一下标读取参数快照。',
         r'''
         ex_t76_val = np.array([.7, .4, .5])
         ex_t76_snapshots = np.array([1., 2., 3.])
         ex_t76_result = ex_t76_snapshots[ex_t76_val.argmin()]
         ''', r'''check('验证集选模', ex_t76_result, 2.)''')
    L.md(r'''
    ## 13. 接到 NTU 的实际回归作业

    本地 [HW05_Regression.ipynb](../../NTU_HYLee/GenAI_ML/hw/HW05_Regression.ipynb)
    用游戏特征预测 Steam 游戏的 Metacritic 分数。输入从本章的一列 x 变成多列游戏特征，
    输出仍是一个连续评分，主干形状因此变成 `(B,D)→(B,1)`。
    Dataset 的 `__getitem__` 决定一份游戏数据究竟含哪些数值，模型第一层必须接得上这个 D。

    阅读时沿着一份游戏记录追到评分损失，标出中途的形状。
    类别特征编码、缺失值处理与数值缩放都发生在这条数据流上，
    而本章的正向、MSE、反传和更新可以直接成为理解训练部分的起点。
    ''')
    L.reading('PyTorch：CrossEntropyLoss',
              'https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',
              '找到输入和类别编号目标的形状。把文档中的负对数似然展开成 `-z_t+logsumexp(z)`，'
              '对照本课推导，说明批平均怎样进入梯度。\n\n'
              '**单变量实验：**只把隐藏宽度从16改成4，保留划分和训练设置；同时比较验证误差与边界形状。')
    return L


def convolution():
    L = Lesson('08_convolution_from_windows', '08｜从一个滑动窗口写出卷积',
               ['手算窗口乘加，并写出可调步幅和补零的双循环',
                '追踪 NCHW、多输入通道与多输出通道的形状',
                '手算共享卷积核和重叠输入像素的梯度'],
               '05—07：Tensor、矩阵形状、损失与反传')
    L.md(r'''
    在一张图中找同一种局部变化，规则不该到了另一个位置就全部重学。
    例如“左边亮、右边暗”可以用一组左右异号的权重检测；把这组权重移到图像各处，
    就能得到各处的响应。卷积将这个做法写成了一项可训练的计算。

    本章用1到16排成的小图，逐个算出四个响应，再追踪损失怎样回到共享核权重。
    这九个数既会影响多个输出，也会从多个位置收回梯度。
    ''')
    section(L, '1. 用一张小图测试局部规则', r'''
    灰度图写成 $H\times W$ 数表，`[r,c]` 对应第 r 行、第 c 列像素。
    下格用4×4表代替真实灰度值，用3×3卷积核 K 表示局部规则。
    核左列为1、右列为−1，中间为0；它会把三行的“左边减右边”加起来。
    ''', r'''
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    import matplotlib.pyplot as plt
    torch.set_num_threads(1); torch.manual_seed(8); style()
    image = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    kernel = np.array([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
    print('图像:\n', image, '\n卷积核:\n', kernel)
    ''')
    section(L, '2. 左上窗口的九个数怎样变成−6', r'''
    `image[0:3,0:3]` 取左上角3×3窗口。第一行 `[1,2,3]` 与核第一行相乘，
    贡献 $1-3=-2$；另外两行也各贡献−2，合起来就是−6。
    代码的 `*` 完成逐项乘法，`.sum()` 将九个乘积汇成一个输出。

    这里窗口与核保持原方向相乘，数学上称互相关；PyTorch 的 Conv2d 使用这个约定。
    后面与框架对照时，同一张核可以直接传入。
    ''', r'''
    window = image[0:3, 0:3]
    products = window * kernel
    first_output = products.sum()
    print('窗口:\n', window, '\n九个乘积:\n', products)
    print('输出:', first_output)
    check('第一个输出', first_output, -6.)
    ''')
    L.ex('8.1', '取窗口再乘加',
         '用 `np.arange(1,17).reshape(4,4)`，取行1:4、列1:4的3×3窗口；与全1卷积核乘加，返回标量。',
         r'''ex_t81_result = None''', 'NumPy切片右端不包含；这个窗口从像素6开始，最后一个像素16。',
         r'''
         ex_t81_image = np.arange(1, 17).reshape(4, 4)
         ex_t81_window = ex_t81_image[1:4, 1:4]
         ex_t81_result = (ex_t81_window * np.ones((3, 3))).sum()
         ''', r'''check('右下窗口求和', ex_t81_result, 99.)''')
    section(L, '3. 两个循环把窗口移遍图像', r'''
    窗口从左上移动一格，又遇到一份3×3数据，仍用刚才的核乘加。
    左上角可放在第0或第1行、第0或第1列，共四种位置，所以输出是2×2。
    外循环列举行，内循环列举列，`output[row,col]` 存当前位置的响应。

    四个响应都会是−6：这张小图的每个窗口中，同一行的右端都比左端大2。
    同一规则在四处检测到了相同的局部变化，九个核权重只存了一份。
    ''', r'''
    output = np.zeros((2, 2), dtype=np.float64)
    for row in range(2):
        for col in range(2):
            patch = image[row:row + 3, col:col + 3]
            output[row, col] = np.sum(patch * kernel)
            print('位置', (row, col), '→', output[row, col])
    print('完整输出:\n', output)
    ''')
    section(L, '4. 让窗口在图中可见', r'''
    左图的方框圈出第一次乘加使用的九个像素，中图是那份反复使用的核，
    右图的四格对应四个窗口。核中1与−1的总和为0，
    因而常数窗口的响应为0；像素从左向右增大时，这份核给出负响应。
    权重符号、图像变化方向和响应符号可以在这三张图中逐项对上。
    ''', r'''
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, table, title in zip(axes, [image, kernel, output], ['Image + first window', 'Shared kernel', 'Output']):
        ax.imshow(table, cmap='coolwarm')
        for row, col in np.ndindex(table.shape):
            ax.text(col, row, f'{table[row, col]:g}', ha='center', va='center', fontsize=12)
        ax.set(title=title, xticks=[], yticks=[])
    axes[0].add_patch(Rectangle((-.5, -.5), 3, 3, fill=False, edgecolor='black', linewidth=2))
    fig.tight_layout(); show(fig)
    ''')
    L.ex('8.2', '把2×2核滑遍3×3图',
         '图像是1到9的3×3表，核为全1的2×2表。自己用两个循环返回2×2输出。',
         r'''ex_t82_result = None''', '先建立2×2零输出，再逐位置切片 `image[r:r+2,c:c+2]`。',
         r'''
         ex_t82_image = np.arange(1, 10).reshape(3, 3)
         ex_t82_result = np.zeros((2, 2))
         for ex_t82_r in range(2):
             for ex_t82_c in range(2):
                 ex_t82_result[ex_t82_r, ex_t82_c] = ex_t82_image[ex_t82_r:ex_t82_r+2, ex_t82_c:ex_t82_c+2].sum()
         ''', r'''check('窗口滑动', ex_t82_result, [[12., 16.], [24., 28.]])''')
    section(L, '5. stride 与 padding 决定窗口位置', r'''
    刚才窗口每次移动一个像素。将 `stride` 改为2，就跳过中间一些位置；
    在边缘补一圈0，即 `padding=1`，窗口便能覆盖到原图边缘附近的新位置。
    设步幅为 s、四边各补 p 个像素，核高为 $k_H$。补零后高度是 $H+2p$，
    一个完整窗口可沿竖直方向移动 $\lfloor(H+2p-k_H)/s\rfloor$ 次，再加初始位置，得到
    $$H_{out}=\left\lfloor\frac{H+2p-k_H}{s}\right\rfloor+1,$$
    宽度用同样方法计算。下格会打印各窗口在补零图中的左上角坐标，
    它们正是下一节切片所需的起点。
    ''', r'''
    padding, stride = 1, 2
    padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    height_out = (image.shape[0] + 2 * padding - kernel.shape[0]) // stride + 1
    width_out = (image.shape[1] + 2 * padding - kernel.shape[1]) // stride + 1
    print('补零后形状:', padded.shape, '输出形状:', (height_out, width_out))
    print('各输出窗口的左上角:', [(r * stride, c * stride)
                                    for r in range(height_out) for c in range(width_out)])
    ''')
    section(L, '6. 把已经看懂的步骤装成函数', r'''
    原来的两个循环可以继续使用，只需把窗口起点改成 `row*stride`、`col*stride`，
    并从补零图中取值。函数里的 `oh`、`ow` 是输出可放的行列数，
    `kh`、`kw` 是每次切片应取的大小。
    末尾用默认步幅1、补零0重算原例，结果应与刚才逐窗口得到的 output 相同。
    ''', r'''
    def conv2d_numpy(array, weight, stride=1, padding=0):
        array = np.asarray(array, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        kh, kw = weight.shape
        padded = np.pad(array, ((padding, padding), (padding, padding)))
        oh = (array.shape[0] + 2 * padding - kh) // stride + 1
        ow = (array.shape[1] + 2 * padding - kw) // stride + 1
        result = np.zeros((oh, ow), dtype=np.float64)
        for row in range(oh):
            for col in range(ow):
                start_r, start_c = row * stride, col * stride
                patch = padded[start_r:start_r + kh, start_c:start_c + kw]
                result[row, col] = np.sum(patch * weight)
        return result
    check('函数复现手算', conv2d_numpy(image, kernel), output)
    ''')
    L.ex('8.3', '先算输出尺寸',
         '输入高宽均为7，核3×3，padding=1，stride=2。按公式返回 `[Hout,Wout]`。',
         r'''ex_t83_result = None''', '整数除法 `//` 完成向下取整。',
         r'''
         ex_t83_size = (7 + 2 * 1 - 3) // 2 + 1
         ex_t83_result = [ex_t83_size, ex_t83_size]
         ''', r'''check('卷积输出高宽', ex_t83_result, [4, 4])''')
    section(L, '7. 从二维表到 NCHW', r'''
    PyTorch 的卷积要同时处理多张图、多种通道，因此输入约定为 `(N,C,H,W)`：
    图像张数、通道数、高、宽。本例只有一张灰度图，
    `[None,None,:,:]` 插入两个长度1的轴，得到 `(1,1,4,4)`，像素值保持原样。

    权重的轴是 `(Cout,Cin,kH,kW)`，这份核为 `(1,1,3,3)`。
    下格将步幅2、补零1传给两种实现，取 PyTorch 输出的 `[0,0]`，
    就能与手写函数的二维输出逐项比较。
    ''', r'''
    torch_image = torch.tensor(image)[None, None, :, :]
    torch_kernel = torch.tensor(kernel)[None, None, :, :]
    reference = F.conv2d(torch_image, torch_kernel, stride=2, padding=1)
    print('输入:', tuple(torch_image.shape), '核:', tuple(torch_kernel.shape),
          '输出:', tuple(reference.shape))
    check('NumPy与Conv2d', conv2d_numpy(image, kernel, stride=2, padding=1), reference[0, 0])
    ''')
    section(L, '8. 多个输入通道汇成一个输出通道', r'''
    一张彩色图有多个输入通道。要计算一个输出通道，每个输入通道各有一片核，
    它们的窗口响应相加后，再加这个输出通道的偏置：
    $$Y_{n,o,r,c}=b_o+\sum_{i=1}^{C_{in}}\sum_{u,v}
      K_{o,i,u,v}X_{n,i,rs+u-p,cs+v-p}.$$
    下格的第一个输入通道全是1，第二个全是2，各取2×2窗口、使用全1权重，
    输出就是 $4\times1+4\times2=12$。
    要得到更多输出通道，就学习更多组核。参数总量为
    $C_{out}(C_{in}k_Hk_W+1)$，其中每个输出通道有一个偏置，参数量与图像高宽无关。
    ''', r'''
    channels = torch.stack([torch.ones(3, 3), 2 * torch.ones(3, 3)])[None]
    weights = torch.ones(1, 2, 2, 2)
    combined = F.conv2d(channels, weights)
    print('两个输入通道:', channels.shape, '一个输出通道:', combined.shape)
    print('4个1 + 4个2 =', combined[0, 0, 0, 0].item())
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1)
    print('可学习参数:', sum(p.numel() for p in layer.parameters()))
    ''')
    L.ex('8.4', '卷积参数量',
         'Conv2d 输入3通道、输出8通道、核3×3、带偏置，返回参数量。图像从32×32变成64×64时参数量是否改变？',
         r'''ex_t84_result = None''', '每个输出通道有3×3×3个核权重和1个偏置。',
         r'''ex_t84_result = 8 * (3 * 3 * 3 + 1)''',
         r'''check('共享核参数总量', ex_t84_result, 224)''')
    section(L, '9. 一个输出的核梯度与输入梯度', r'''
    将刚才的局部乘加放回损失中。下面缩小为2×2输入与2×2核，只产生一个输出，
    $q=\sum_{u,v}K_{uv}X_{uv}$，目标为0，$L=\tfrac12q^2$。
    对乘积中的核元素求导留下对应像素，再乘上游导数 q，得到
    $$\frac{\partial L}{\partial K_{uv}}=qX_{uv},\qquad
      \frac{\partial L}{\partial X_{uv}}=qK_{uv}.$$
    对输入求导则留下核元素。下格 q=6，核梯度应为输入表的6倍，
    输入梯度应为核的6倍。这个单窗口例子仍然只是标量乘法的链式法则。
    ''', r'''
    tiny_x = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
    tiny_k = torch.tensor([[[[1., 0.], [-1., 2.]]]], requires_grad=True)
    tiny_q = F.conv2d(tiny_x, tiny_k)
    tiny_loss = 0.5 * tiny_q.square().sum()
    tiny_loss.backward()
    print('q:', tiny_q.item(), '\n核梯度:\n', tiny_k.grad[0, 0], '\n输入梯度:\n', tiny_x.grad[0, 0])
    check('核的局部梯度', tiny_k.grad, tiny_q.detach() * tiny_x.detach())
    check('输入的局部梯度', tiny_x.grad, tiny_q.detach() * tiny_k.detach())
    ''')
    section(L, '10. 权重共享让核梯度在位置之间相加', r'''
    将窗口移遍图像后，同一个 $K_{uv}$ 影响许多输出。
    每个输出都会沿自己的乘法路径给它一份贡献，所有贡献在这个共享权重处相加：
    $$\frac{\partial L}{\partial K_{uv}}
      =\sum_{r,c}\frac{\partial L}{\partial Y_{rc}}X_{r+u,c+v}.$$
    下格取损失为全部输出之和，每个输出的上游梯度都为1。
    核左上角梯度就是四个窗口左上角像素之和：$1+2+4+5=12$。
    同时，图中央像素也被四个窗口使用，它会收到四份输入梯度。
    前一种累加发生在共享参数处，后一种发生在重叠输入处。
    ''', r'''
    overlap_x = torch.arange(1., 10.).reshape(1, 1, 3, 3).requires_grad_()
    overlap_k = torch.ones(1, 1, 2, 2, requires_grad=True)
    overlap_y = F.conv2d(overlap_x, overlap_k)
    overlap_y.sum().backward()
    print('共享核的梯度:\n', overlap_k.grad[0, 0])
    print('像素收到的贡献次数:\n', overlap_x.grad[0, 0])
    check('核左上角累加', overlap_k.grad[0, 0, 0, 0], 1 + 2 + 4 + 5)
    ''')
    L.ex('8.5', '自己汇总共享权重梯度',
         '图像是1到9的3×3表，2×2核、步幅1，损失是全部输出之和。用窗口累加返回核梯度矩阵。',
         r'''ex_t85_result = None''', '每个输出的上游导数为1，把四份2×2窗口相加。',
         r'''
         ex_t85_image = np.arange(1., 10.).reshape(3, 3)
         ex_t85_result = np.zeros((2, 2))
         for ex_t85_r in range(2):
             for ex_t85_c in range(2):
                 ex_t85_result += ex_t85_image[ex_t85_r:ex_t85_r+2, ex_t85_c:ex_t85_c+2]
         ''', r'''check('共享核梯度', ex_t85_result, [[12., 16.], [24., 28.]])''')
    section(L, '11. 画出重叠输入的梯度', r'''
    右图的中心是4、角点是1。原因是本例核元素都为1、每个输出梯度也为1，
    所以像素每被一个窗口覆盖一次，就收到一份单位梯度。
    左图则是核各位置收到的像素数值之和，大小为12、16、24、28。
    两张图都来自四份窗口，但它们分别回答了“改核权重”和“改输入像素”怎样影响损失。
    ''', r'''
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3))
    for ax, values, title in zip(axes, [overlap_k.grad[0, 0], overlap_x.grad[0, 0]],
                                 ['Shared-kernel gradient', 'Input gradient: overlap counts']):
        table = values.numpy()
        ax.imshow(table, cmap='Blues')
        for row, col in np.ndindex(table.shape):
            ax.text(col, row, f'{table[row, col]:g}', ha='center', va='center', fontsize=13)
        ax.set(title=title, xticks=[], yticks=[])
    fig.tight_layout(); show(fig)
    ''')
    L.ex('8.6', '输入中央像素的梯度',
         '3×3输入、全1的2×2核，损失是全部输出之和。建立图并返回输入梯度。',
         r'''ex_t86_result = None''', '输入要是叶节点：先reshape，再 `.requires_grad_()`。',
         r'''
         ex_t86_x = torch.zeros(1, 1, 3, 3, requires_grad=True)
         F.conv2d(ex_t86_x, torch.ones(1, 1, 2, 2)).sum().backward()
         ex_t86_result = ex_t86_x.grad[0, 0].clone()
         ''', r'''check('重叠处输入梯度', ex_t86_result, [[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])''')
    L.reading('LeCun 等：Gradient-Based Learning Applied to Document Recognition（1998）',
              'https://bottou.org/papers/lecun-98h',
              '阅读介绍局部感受野和共享权重的部分。假设输入32×32、单通道、6个5×5核、无补零、带偏置：'
              '分别计算共享卷积和每个输出位置各用独立权重的参数量。说明共享规则如何限制模型。\n\n'
              '**单变量实验：**只把步幅从1改成2，先预测输出形状及中心像素被覆盖次数，再用代码核验。')
    return L


def vision():
    L = Lesson('09_cnn_training_and_residuals', '09｜真实数字分类、卷积归纳偏置与残差连接',
               ['在同一离线数字数据划分上训练参数量接近的 MLP 与 CNN',
                '追踪每层 NCHW 形状、池化和位移响应',
                '推导残差的输入梯度，并写出恒等与投影捷径'],
               '06—08：训练循环、交叉熵、窗口卷积与共享权重')
    L.md(r'''
    这章让网络读出手写数字。每张图只有8×8个灰度像素，我们会用参数量几乎相同的
    MLP 和 CNN 训练，查看它们读对了哪些、读错了哪些。

    还会把测试图整体右移一个像素。按本章固定的种子和划分，CNN 原图准确率约98%，
    右移后降到约58%。一个像素已经占了图像宽度的八分之一，
    这个变化将把池化、边界和训练数据覆盖范围的作用显出来。
    后半章再给已有特征加一条直接通路，算清残差连接怎样同时改变前向结果和反向梯度。
    ''')
    section(L, '1. 读取真实的手写数字', r'''
    `load_digits` 将1797张真实手写数字图像随 scikit-learn 一起提供，读取时无需下载。
    每张8×8、像素范围0到16，标签是0到9的类别编号。
    除以已知像素上限16后，输入落在0到1之间；标签保持整数，用于交叉熵。

    `[:,None]` 在样本维之后插入单通道轴，所以打印的输入形状是 `(1797,1,8,8)`，
    每个下标现在都有明确含义：第几张图、哪个通道、哪一行、哪一列。
    ''', r'''
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    torch.set_num_threads(1); torch.manual_seed(9); style()
    digits = load_digits()
    images = torch.tensor(digits.images, dtype=torch.float32)[:, None] / 16.
    targets = torch.tensor(digits.target, dtype=torch.long)
    print('NCHW:', tuple(images.shape), '标签:', targets.shape)
    ''')
    section(L, '2. 同一个划分用于两个模型', r'''
    让两个模型回答同一批测试题，比较才有共同的落点。
    这里保留360张测试图，再从其余图中取360张验证图，剩下1077张参与训练。
    `stratify` 按类别分层抽样，使每份数据都含有相近比例的0到9。

    下格打印训练集中各类的数量，可以确认每个数字都有一百多张样本。
    两种模型使用同一划分、各训练18轮；每轮的验证 CE 用来保存参数快照。
    ''', r'''
    indices = np.arange(len(images))
    train_valid, test_ids = train_test_split(indices, test_size=.2, random_state=9,
                                             stratify=digits.target)
    train_ids, valid_ids = train_test_split(train_valid, test_size=.25, random_state=9,
                                            stratify=digits.target[train_valid])
    print('训练/验证/测试:', len(train_ids), len(valid_ids), len(test_ids))
    print('训练各类别数:', np.bincount(digits.target[train_ids], minlength=10))
    ''')
    section(L, '3. 看清模型要识别的笔画', r'''
    下图展示12张训练图。同一个数字可以写得很宽、很窄，笔画的灰度也有差别。
    8×8分辨率将很多细节压到一两个像素中，模型需要从这些数值中找到类别线索。

    画图使用 `images[index,0]` 取出二维灰度表；模型计算时仍保留长度1的通道轴。
    记住这些图的清晰程度，后面看错例与位移实验时便有实际参照。
    ''', r'''
    fig, axes = plt.subplots(2, 6, figsize=(8, 3.2))
    for ax, index in zip(axes.flat, train_ids[:12]):
        ax.imshow(images[index, 0], cmap='gray_r', vmin=0, vmax=1)
        ax.set(title=f'Class {targets[index].item()}', xticks=[], yticks=[])
    fig.tight_layout(); show(fig)
    ''')
    L.ex('9.1', '给灰度图加入通道维',
         '输入张量形状 `(5,8,8)`，用 `[:,None,:,:]` 变成 NCHW，返回新形状。',
         r'''ex_t91_result = None''', '新插入的维度位于样本数之后，长度是1。',
         r'''
         ex_t91_images = torch.zeros(5, 8, 8)
         ex_t91_result = list(ex_t91_images[:, None, :, :].shape)
         ''', r'''check('NCHW形状', ex_t91_result, [5, 1, 8, 8])''')
    section(L, '4. MLP 基线：每个像素是一个输入特征', r'''
    把64个像素逐行排成一个向量，就能沿用上一章的 MLP。
    `Flatten` 把 `(B,1,8,8)` 变成 `(B,64)`，隐藏层用26个响应，最后输出10个类别分数。
    每个隐藏单元都给64个像素各分配一个权重，可以直接使用整张图的信息。
    参数量为
    $64\times26+26+26\times10+10=1960$。
    ''', r'''
    class DigitMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.hidden = nn.Linear(64, 26)
            self.output = nn.Linear(26, 10)
        def forward(self, x):
            x = self.flatten(x)
            x = F.relu(self.hidden(x))
            return self.output(x)
    ''')
    section(L, '5. CNN：在各处使用同一组局部规则', r'''
    CNN 将上一章的窗口计算用于每张数字图。第一层的每份3×3核只读取一小块笔画，
    在各位置重复使用；第二层读取第一层的局部响应，继续组合特征。
    两层3×3卷积都补零1，保持8×8空间尺寸；每层输出8个通道。
    2×2最大池化、步幅2后变成4×4。展平得到 $8\times4\times4=128$ 个特征，
    最后线性层输出10个 logits。总参数量1954，与 MLP 的1960接近。
    这些参数被安排成局部连接、共享核与池化，规定了模型怎样组合图像信息。
    这种由结构预先给定的偏好称为归纳偏置。训练会学习核的具体数值，
    而“同一核用于各处”这条结构关系在训练前已经确定。
    ''', r'''
    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.output = nn.Linear(8 * 4 * 4, 10)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return self.output(x.flatten(start_dim=1))
    ''')
    section(L, '6. 逐层验证形状', r'''
    用四张图穿过每一层，打印它们的形状。卷积把通道数改为8，补零让空间尺寸保持8×8；
    池化后变成4×4，展平就是128项。最后的 Linear 因而接收128项，输出10项。

    `flatten(start_dim=1)` 从通道轴开始合并，保留最前面的批维。
    输出末尾的 `(4,10)` 表示四份样本各有十个分数，正好接上类别编号组成的目标。
    ''', r'''
    torch.manual_seed(9)
    models = {'MLP': DigitMLP(), 'CNN': DigitCNN()}
    shape_net = models['CNN']
    shape_x = images[train_ids[:4]]
    for name, layer in [('conv1', shape_net.conv1), ('relu1', nn.ReLU()),
                         ('conv2', shape_net.conv2), ('relu2', nn.ReLU()),
                         ('pool', shape_net.pool), ('flatten', nn.Flatten()),
                         ('output', shape_net.output)]:
        shape_x = layer(shape_x)
        print(name, tuple(shape_x.shape))
    print('参数量:', {name: sum(p.numel() for p in net.parameters()) for name, net in models.items()})
    ''')
    L.ex('9.2', '卷积与池化的组合形状',
         '输入 `(4,3,12,12)`，先 Conv2d(3,5,3,padding=1)，再 MaxPool2d(2)，返回输出形状。',
         r'''ex_t92_result = None''', '卷积保持空间尺寸，池化把高宽各减半。',
         r'''
         ex_t92_net = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1), nn.MaxPool2d(2))
         ex_t92_result = list(ex_t92_net(torch.zeros(4, 3, 12, 12)).shape)
         ''', r'''check('组合输出形状', ex_t92_result, [4, 5, 6, 6])''')
    section(L, '7. 训练数据如何组成 batch', r'''
    一批64张图的输入是 `(64,1,8,8)`，目标是64个类别编号，模型输出 `(64,10)`。
    `TensorDataset` 将每张图与同一下标的标签配对，DataLoader 负责打乱并分批。

    两次训练各自调用 `make_loader()`，都从种子9开始取样，因此各轮样本顺序也相同。
    下格实际取出一批并计算输出，可以沿着这三个形状追踪样本维。
    ''', r'''
    def make_loader():
        dataset = TensorDataset(images[train_ids], targets[train_ids])
        return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(9))
    example_x, example_y = next(iter(make_loader()))
    print('batch输入:', tuple(example_x.shape), '目标:', tuple(example_y.shape))
    print('模型输出:', tuple(models['CNN'](example_x).shape))
    ''')
    section(L, '8. 用同一份训练函数更新两个模型', r'''
    循环不需要知道传进来的是 MLP 还是 CNN：两者都接收这一批图像，返回十类 logits。
    每一批的交叉熵反传到各自参数，Adam 完成更新。
    `total` 收集这一轮的损失和，再除以1077得到训练集平均值。

    一轮结束后，模型在360张验证图上给出 CE 和准确率。
    验证 CE 降低时保存当前参数，结束后恢复这份参数；records 则保留每轮的三项指标。
    ''', r'''
    def train_digits(net, epochs=18):
        loader = make_loader()
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        records, best_value, best_state = [], float('inf'), None
        for epoch in range(epochs):
            net.train(); total = 0.
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(net(inputs), labels)
                loss.backward(); optimizer.step()
                total += loss.item() * len(inputs)
            net.eval()
            with torch.no_grad():
                z = net(images[valid_ids])
                valid_loss = F.cross_entropy(z, targets[valid_ids]).item()
                valid_acc = (z.argmax(1) == targets[valid_ids]).float().mean().item()
            records.append((total / len(train_ids), valid_loss, valid_acc))
            if valid_loss < best_value:
                best_value = valid_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        return np.asarray(records)
    ''')
    L.ex('9.3', '最大池化把梯度送到哪里',
         '输入2×2表 `[[1,4],[2,3]]`，最大池化得到一个数，再对输出之和反传，返回输入梯度。',
         r'''ex_t93_result = None''', '只有最大元素4影响这个窗口的最大值，它收到单位梯度。',
         r'''
         ex_t93_x = torch.tensor([[[[1., 4.], [2., 3.]]]], requires_grad=True)
         F.max_pool2d(ex_t93_x, 2).sum().backward()
         ex_t93_result = ex_t93_x.grad[0, 0].clone()
         ''', r'''check('最大池化输入梯度', ex_t93_result, [[0., 1.], [0., 0.]])''')
    section(L, '9. 两个模型在留出数字上的表现', r'''
    在这份划分上，MLP 测试准确率约96.1%，CNN 约98.1%。
    这两个小模型分别用1960、1954个参数，以同样批大小训练18轮，
    卷积中的同一参数仍会在多个位置参与运算。

    打印的“验证最高准确率”取自训练过程，测试准确率则使用验证 CE 最低时保存的参数。
    准确率与 CE 关注的数值不同：前者只数类别对错，后者还取决于正确类概率的大小。
    ''', r'''
    histories = {name: train_digits(net) for name, net in models.items()}
    test_predictions = {}
    for name, net in models.items():
        net.eval()
        with torch.no_grad():
            predictions = net(images[test_ids]).argmax(1)
        test_predictions[name] = predictions
        accuracy = (predictions == targets[test_ids]).float().mean().item()
        print(name, '参数:', sum(p.numel() for p in net.parameters()),
              '验证最高准确率:', round(histories[name][:, 2].max(), 4),
              '测试准确率:', round(accuracy, 4))
    ''')
    section(L, '10. 曲线和错例提供不同的信息', r'''
    曲线把18轮训练展开在同一横轴上。左图记录训练 CE，右图记录验证准确率，
    可以沿同一轮的横坐标，对照训练误差的下降与留出图正确比例的变化。

    随后的错例图按测试集顺序取 CNN 的前几个错误，标题 T/P 为真实类别与预测类别。
    对一张错图，可以具体指出哪段笔画模糊、闭合或偏移，并与图中的预测数字比较。
    一个总准确率没有保留这些图像细节。
    ''', r'''
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for name, records in histories.items():
        axes[0].plot(np.arange(1, len(records) + 1), records[:, 0], label=name)
        axes[1].plot(np.arange(1, len(records) + 1), records[:, 2], label=name)
    axes[0].set(xlabel='Epoch', ylabel='Training CE'); axes[0].legend()
    axes[1].set(xlabel='Epoch', ylabel='Validation accuracy', ylim=(0, 1)); axes[1].legend()
    fig.tight_layout(); show(fig)
    ''')
    L.code(r'''
    wrong = torch.where(test_predictions['CNN'] != targets[test_ids])[0][:8]
    fig, axes = plt.subplots(1, 8, figsize=(11, 2))
    for ax in axes:
        ax.axis('off')
    for ax, local_index in zip(axes, wrong):
        index = test_ids[local_index.item()]
        ax.imshow(images[index, 0], cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'T {targets[index].item()} / P {test_predictions["CNN"][local_index].item()}', fontsize=11)
    fig.tight_layout(); show(fig)
    ''')
    section(L, '11. 右移一个像素后，哪些东西改变了', r'''
    2×2最大池化只保留窗口内的最大响应。下格把峰值2在同一个窗口里换位置，
    两次输出都是2；跨过池化窗口边界时，这个结果就可能改变。

    接着将360张测试图整体右移一像素，左边补0、最右列移出图像。
    MLP 与 CNN 的准确率都会明显下降。除了像素位置改变，
    池化窗口分组、图像边缘与展平后的特征位置也随之改变。
    本次训练使用的原始图像并未覆盖这整组平移后的输入，
    因而“局部共享规则”在这里带来的稳定程度，要由实际预测结果回答。
    ''', r'''
    peak_a = torch.tensor([[[[0., 2.], [0., 0.]]]])
    peak_b = torch.tensor([[[[0., 0.], [2., 0.]]]])
    print('移动前后池化:', F.max_pool2d(peak_a, 2).item(), F.max_pool2d(peak_b, 2).item())
    shifted_test = torch.zeros_like(images[test_ids])
    shifted_test[:, :, :, 1:] = images[test_ids][:, :, :, :-1]
    for name, net in models.items():
        with torch.no_grad():
            shifted_acc = (net(shifted_test).argmax(1) == targets[test_ids]).float().mean().item()
        print(name, '右移1像素测试准确率:', round(shifted_acc, 4))
    ''')
    L.ex('9.4', '平移时边界怎样处理',
         '对一行像素 `[1,2,3,4]` 实现右移一格、左边补0，返回新行。不要让最后一个像素绕回左边。',
         r'''ex_t94_result = None''', '先建立全0张量，再令新行的1:等于旧行的:-1。',
         r'''
         ex_t94_row = torch.tensor([1., 2., 3., 4.])
         ex_t94_result = torch.zeros_like(ex_t94_row)
         ex_t94_result[1:] = ex_t94_row[:-1]
         ''', r'''check('补零平移', ex_t94_result, [0., 1., 2., 3.])''')
    section(L, '12. 让新增的一层保留已有输入', r'''
    假设前面已经算出一份有用的特征 x，还想通过更多层改进它。
    用 $y=x+F(x)$ 时，新增分支只需给出要加到原特征上的修正；
    当 F=0，整个块便输出原来的 x。这就是残差连接的直接用途。

    输入经过直接路径和 F 分支两次到达 y。对输入求导：
    $$\frac{dy}{dx}=1+F'(x),\qquad
      \frac{dL}{dx}=\frac{dL}{dy}\bigl(1+F'(x)\bigr).$$
    下格 $F(x)=\tanh(0.2x)$、x=0.5，分支导数约0.198，直接路径导数为1，
    所以总导数约1.198。这个1在 F 分支导数较小时仍存在；
    两路贡献带着正负号相加，若某处 $F'(x)=-1$，总和也会为0。
    ''', r'''
    residual_x = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    residual_w = torch.tensor(.2, dtype=torch.float64)
    branch = torch.tanh(residual_w * residual_x)
    residual_y = residual_x + branch
    residual_y.backward()
    expected = 1 + residual_w * (1 - branch.detach().square())
    print('直接路径:', 1., '分支路径:', (expected - 1).item(), '总梯度:', residual_x.grad.item())
    check('标量残差导数', residual_x.grad, expected)
    ''')
    section(L, '13. 向量情形：I+J 的每个维度', r'''
    图像特征含有许多坐标，直接路径将每个输入坐标复制到输出的同一坐标，
    因此它的导数是单位矩阵 I。采用列向量 $x,y\in\mathbb R^d$，$F:\mathbb R^d\to\mathbb R^d$。
    雅可比矩阵 $J_F$ 的第 $(i,j)$ 项为 $\partial F_i/\partial x_j$，
    所以 $J_y=I+J_F$。若上游梯度为列向量 $g=\nabla_yL$，
    $$\nabla_xL=(I+J_F)^\top g=g+J_F^\top g.$$
    转置来自反向收集：一个输入坐标需要收齐它对所有输出坐标的影响。
    下格令 $F(x)=Wx$，上游梯度 $g=[2,-1]$；分支回传 $W^\top g=[0,-0.7]$，
    再加直接路径的 `[2,-1]`，得到输入梯度 `[2,-1.7]`。
    ''', r'''
    vector_x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    matrix_w = torch.tensor([[.2, -.3], [.4, .1]], dtype=torch.float64)
    upstream = torch.tensor([2., -1.], dtype=torch.float64)
    vector_y = vector_x + matrix_w @ vector_x
    vector_loss = upstream @ vector_y
    vector_loss.backward()
    print('直接贡献:', upstream, '分支贡献:', matrix_w.T @ upstream)
    check('向量残差回传', vector_x.grad, upstream + matrix_w.T @ upstream)
    ''')
    L.ex('9.5', '把两路梯度相加',
         r'$F(x)=Wx$，$W=[[0,2],[1,0]]$，上游梯度 $g=[3,4]$。手算残差输出 $x+Wx$ 对输入的梯度。',
         r'''ex_t95_result = None''', r'列向量约定下是 $g+W^\top g$。',
         r'''
         ex_t95_W = torch.tensor([[0., 2.], [1., 0.]])
         ex_t95_g = torch.tensor([3., 4.])
         ex_t95_result = ex_t95_g + ex_t95_W.T @ ex_t95_g
         ''', r'''check('残差两路回传', ex_t95_result, [7., 10.])''')
    section(L, '14. 写出同形状的卷积残差块', r'''
    将标量分支换成两层卷积，就能对整份图像特征应用同一结构。
    输入 `(B,C,H,W)` 经过两层3×3卷积，步幅1、补零1使高宽保持相同，
    输入输出通道数也相同，因此 x 与 branch 可以逐元素相加。

    下面的 `return x + branch` 正是 $x+F(x)$。中间的 ReLU 属于 F 的内部计算，
    其局部导数会包含在 $J_F$ 中。把分支参数全部设为0的练习，将直接核验保留输入及其梯度的效果。
    ''', r'''
    class SmallResidual(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        def forward(self, x):
            branch = self.conv2(F.relu(self.conv1(x)))
            return x + branch
    block = SmallResidual(8)
    print('残差块输入输出:', (4, 8, 8, 8), tuple(block(torch.zeros(4, 8, 8, 8)).shape))
    ''')
    section(L, '15. 改变通道或尺寸时补上投影', r'''
    如果主分支把8个通道改成16个，并将8×8缩成4×4，原输入就少了通道、多了空间位置，
    无法与分支逐格相加。捷径用1×1卷积混合通道，步幅2同时缩小空间尺寸，
    两路便都成为 `(B,16,4,4)`。

    这个块写作 $y=P(x)+F(x)$，P 是刚才的投影。
    输入梯度随之成为 $J_P^\top g+J_F^\top g$。直接复制产生的 I，
    在这里换成了投影计算的雅可比；两条路径仍在同一个输入处相加。
    ''', r'''
    projection = nn.Conv2d(8, 16, kernel_size=1, stride=2)
    down_branch = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
    demo_features = torch.randn(4, 8, 8, 8)
    projected = projection(demo_features)
    branch_features = down_branch(demo_features)
    print('捷径与分支:', tuple(projected.shape), tuple(branch_features.shape))
    print('相加后:', tuple((projected + branch_features).shape))
    ''')
    L.md(r'''
    [打开残差交互实验](../interactive/index.html#residual)：选择“F=0，保留输入”，
    逐步观察输出与输入梯度；再选择“两路导数抵消”，把两份带符号的贡献亲手相加。
    最后切到相加后接 ReLU，观察加和为负时，后激活怎样同时改变两条反向路径。
    ''')
    L.ex('9.6', '零分支给出的可核验证据',
         '新建 SmallResidual(2)，把分支的所有参数设成0。输入随机 `(1,2,3,3)`，计算输出之和的输入梯度。',
         r'''ex_t96_result = None''', '在 `no_grad` 中把参数 `zero_()`；分支为0，直接路径导数仍为1。',
         r'''
         ex_t96_block = SmallResidual(2)
         with torch.no_grad():
             for ex_t96_p in ex_t96_block.parameters():
                 ex_t96_p.zero_()
         ex_t96_x = torch.randn(1, 2, 3, 3, requires_grad=True)
         ex_t96_block(ex_t96_x).sum().backward()
         ex_t96_result = ex_t96_x.grad.clone()
         ''', r'''check('零分支的恒等梯度', ex_t96_result, torch.ones(1, 2, 3, 3))''')
    L.md(r'''
    ## 16. 从这个小实验走向 NTU 的图像作业

    本地 [HW06.ipynb](../../NTU_HYLee/GenAI_ML/hw/HW06.ipynb) 分类 MyGO/Ave Mujica 的10个人物，
    原图是150×150×3，作业预处理调整为128×128×3，图像比本章大得多，也增加了颜色通道。
    它仍然经历“读取图像与标签→NCHW batch→类别 logits→交叉熵”这条数据流。
    其中输入张量的通道数改为3，输出仍有10类，卷积层与 Linear 层的维度要跟着数据变化。

    可在作业中沿一张图追到预测结果，特别看清 resize、标签映射和增强发生在哪一步。
    本章右移实验已经给出一个参照：增强怎样改变输入分布、增强后类别是否仍成立，
    都会影响模型收到的训练信息。预训练模型又提供了已有特征这一新起点，
    理解本章各层参数怎样更新后，就能继续追踪它的哪些层会被微调。
    ''')
    L.reading('He 等：Deep Residual Learning for Image Recognition（2015）',
              'https://arxiv.org/abs/1512.03385',
              '对照论文的 basic block，标出卷积、激活、相加和捷径投影。'
              '如果相加之后再接 ReLU，把本课的输入梯度公式补上这个激活的雅可比，写清它乘在哪一侧。\n\n'
              '**单变量实验：**保持数据、划分、轮数不变，只把 CNN 的 MaxPool2d(2) 换成 AvgPool2d(2)。'
              '比较验证曲线、原测试准确率、右移测试准确率；从这三项证据判断池化方式的影响。')
    return L


def build():
    return [tensors(), modules(), mlps(), convolution(), vision()]
