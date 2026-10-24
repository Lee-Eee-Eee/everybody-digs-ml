"""Time, gates, graphs, and liquid dynamics: models are built inside each lesson."""
from coursekit import Lesson


def rnn_lesson():
    l = Lesson('10_rnn_time_and_bptt', '10 时间展开：自己写 RNN 与 BPTT',
               ['按 B、T、F 读取序列', '手算共享权重的跨时梯度', '从循环写出并训练一个序列模型'],
               '张量、矩阵乘法、链式法则、PyTorch 参数与优化器')
    l.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(10)
    style()
    ''')
    l.md(r'''
    一次短脉冲过去以后，滤波器里的状态还会逐渐回落。只看当前输入，无法恢复前面留下的这部分影响；计算时需要把一个状态从上一步带到下一步。本课最后训练的就是这样一个滤波过程：每步保留旧状态的一部分，再加入当前输入。

    ## 1 把时间保留在数组里

    先把几条独立记录排成一个数组。B 是序列数，T 是每条序列的时刻数，F 是每时刻的特征数，因而 `X.shape=(B,T,F)`。
    `X[b,t]` 取第 b 条序列在时刻 t 的 F 个数；`X[:,t,:]` 取所有序列在同一时刻的输入。后面每次循环取一个这样的时间切片，同时更新 B 条序列的状态。
    ''')
    l.code('''
    X_shape = np.arange(24).reshape(2, 4, 3)
    print('B,T,F:', X_shape.shape)
    print('one sequence:', X_shape[0])
    print('all sequences at t=1:', X_shape[:, 1, :])
    ''')
    l.ex('r01', '取一个时间切片', '数组形状为 (2,3,2)，取所有序列在 t=1 的输入。',
         '''
    ex_r01_X = np.arange(12).reshape(2, 3, 2)
    ex_r01_slice = None  # TODO
    ex_r01_shape = None  # TODO
    ''', '时间轴是第 1 轴，写 `X[:,1,:]`。',
         '''
    ex_r01_X = np.arange(12).reshape(2, 3, 2)
    ex_r01_slice = ex_r01_X[:, 1, :]
    ex_r01_shape = ex_r01_slice.shape
    ''', '''
    check('时间切片', ex_r01_slice, [[2, 3], [8, 9]])
    check('切片形状', ex_r01_shape, (2, 2))
    ''')
    l.md(r'''
    ## 2 同一组参数，连续使用三次

    先把输入 $x_t$、状态 $h_t$ 都缩成一个数，t 从 1 编到 3。初态 $h_0=0$。每一步拿到两个量：上一步留下的状态和当前输入，分别乘 a、b 后相加：
    $$h_t=a h_{t-1}+b x_t.$$
    取 $a=0.5,b=2$，输入依次为 1、2、−1。a、b 在三个时刻保持相同；变化的是输入和传下来的状态。第一步得到 2，第二步会把这个 2 继续带入计算。
    ''')
    l.code('''
    x_small = np.array([1., 2., -1.])
    a, b = .5, 2.
    first_state = a * 0. + b * x_small[0]
    print('h_1 =', first_state)
    states = [0.]
    for x_t in x_small:
        states.append(a * states[-1] + b * x_t)
    print('h_0,h_1,h_2,h_3:', states)
    ''')
    l.ex('r02', '补齐一次递推', '从 h=2 接收 x=2，再接收 x=−1，参数仍为 a=.5、b=2。',
         '''
    ex_r02_h = 2.
    ex_r02_second = None  # TODO
    ex_r02_third = None  # TODO
    ''', '每步都用刚得到的新状态作为下一步的旧状态。',
         '''
    ex_r02_h = 2.
    ex_r02_second = .5 * ex_r02_h + 2. * 2.
    ex_r02_third = .5 * ex_r02_second + 2. * (-1.)
    ''', '''
    check('h_2', ex_r02_second, 5.)
    check('h_3', ex_r02_third, .5)
    ''')
    l.md('把循环沿时间摊开，就能看到状态从哪一步传到哪一步。横向箭头带来旧状态，竖向箭头注入当前输入；三个节点使用同一组 a、b。图中每个状态值都来自刚才的递推。')
    l.code('''
    fig, ax = plt.subplots(figsize=(9, 2.4))
    for t, h in enumerate(states):
        ax.text(t, 0, f'$h_{t}={h:g}$', ha='center', bbox=dict(fc='white', ec='#37798A'))
        if t:
            ax.annotate('', (t-.15, 0), (t-1+.15, 0), arrowprops=dict(arrowstyle='->'))
            ax.text(t-.5, .14, '$a$', ha='center')
            ax.annotate(f'$x_{t}={x_small[t-1]:g}$, $b=2$', (t, .1), (t, .65), ha='center', arrowprops=dict(arrowstyle='->'))
    ax.set(xlim=(-.4, 3.5), ylim=(-.4, 1.1)); ax.axis('off')
    show(fig)
    ''')
    l.md(r'''
    ## 3 从末端开始回传

    三步算出的末端状态为 $h_3=0.5$。现在给它一个目标 $d=1.5$，用 $L=\frac12(h_3-d)^2$ 衡量差距。先考察：如果某一步的状态变动一点，末端损失会怎样变动。这个敏感度记作 $\delta_t=\partial L/\partial h_t$。

    从末端开始，$\delta_3=h_3-d=-1$。要回到前一步，就沿着状态箭头穿过一次乘 a 的运算，其局部导数为 $\partial h_t/\partial h_{t-1}=a$：
    $$\delta_2=a\delta_3=-0.5,\qquad\delta_1=a\delta_2=-0.25.$$
    ''')
    l.code('''
    d = 1.5
    delta = np.zeros(3)
    delta[2] = states[3] - d
    for t in [1, 0]:
        delta[t] = a * delta[t+1]
    print('L:', .5 * (states[3] - d) ** 2)
    print('dL/dh_1, dL/dh_2, dL/dh_3:', delta)
    ''')
    l.ex('r03', '沿另一条链回传', 'a=.2，末端 δ3=3，算出 [δ1,δ2,δ3]。',
         '''
    ex_r03_delta = None  # TODO: 从右向左递推
    ex_r03_a = .2
    ex_r03_last = 3.
    ''', '两次相乘分别得到 .6 和 .12，输出按时间正序。',
         '''
    ex_r03_a = .2
    ex_r03_last = 3.
    ex_r03_delta = np.zeros(3)
    ex_r03_delta[-1] = ex_r03_last
    for ex_r03_t in [1, 0]:
        ex_r03_delta[ex_r03_t] = ex_r03_a * ex_r03_delta[ex_r03_t+1]
    ''', "check('三个时刻的敏感度', ex_r03_delta, [.12, .6, 3.])")
    l.md(r'''
    三步用了同一个 a。改变 a 时，它在每个时刻的使用位置都会影响损失，所以要把这些位置的贡献相加：
    $$\frac{\partial L}{\partial a}=\sum_{t=1}^3\delta_t h_{t-1},\quad
      \frac{\partial L}{\partial b}=\sum_{t=1}^3\delta_t x_t,\quad
      \frac{\partial L}{\partial x_t}=b\delta_t.$$
    各步收到的旧状态为 `[0,2,5]`，对应的敏感度是 `[-.25,-.5,-1]`。相乘后得到 a 的三份贡献 `[0,-1,-5]`，合计为 −6。第一步的旧状态为 0，因此该处对 a 的贡献为 0；另外两处通过同一个参数合并。将 a 增大一个很小的 ε，损失的一阶变化就是 −6ε。

    b 的局部乘数则是当前输入。对输入的梯度 $b\delta_t$ 描述改变某个观测会怎样影响损失；优化器根据参数梯度更新 a、b。
    ''')
    l.code('''
    contributions_a = delta * np.array(states[:-1])
    contributions_b = delta * x_small
    print('each use of a:', contributions_a, '; sum:', contributions_a.sum())
    print('each use of b:', contributions_b, '; sum:', contributions_b.sum())
    print('input sensitivities:', b * delta)
    ''')
    l.ex('r04', '共享权重为什么要相加', '按原来的三步轨迹计算 a 的总梯度、b 的总梯度和三个输入梯度。',
         '''
    ex_r04_delta = np.array([-.25, -.5, -1.])
    ex_r04_da = None  # TODO
    ex_r04_db = None  # TODO
    ex_r04_dx = None  # TODO
    ''', 'a 乘旧状态 [0,2,5]；b 乘输入 [1,2,−1]；输入局部导数为 b=2。',
         '''
    ex_r04_delta = np.array([-.25, -.5, -1.])
    ex_r04_da = np.sum(ex_r04_delta * [0., 2., 5.])
    ex_r04_db = np.sum(ex_r04_delta * [1., 2., -1.])
    ex_r04_dx = 2. * ex_r04_delta
    ''', '''
    check('dL/da', ex_r04_da, -6.)
    check('dL/db', ex_r04_db, -.25)
    check('dL/dx', ex_r04_dx, [-.5, -1., -2.])
    ''')
    l.md('刚才的相加也能从自动微分里看到。a、b 各创建一次，循环反复使用它们，`backward()` 就会把各个使用位置的贡献累加到同一个 `.grad`。这里采用 float64，把数值核对误差压小。')
    l.code('''
    a_t = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    x_torch = torch.tensor([1., 2., -1.], dtype=torch.float64, requires_grad=True)
    h = torch.zeros((), dtype=torch.float64)
    for value in x_torch:
        h = a_t * h + b_t * value
    loss_small = .5 * (h - 1.5) ** 2
    loss_small.backward()
    check('autograd a', a_t.grad, -6.)
    check('autograd input', x_torch.grad, [-.5, -1., -2.])
    ''')
    l.code('''
    def scalar_loss(a_value):
        h_value = 0.
        for value in [1., 2., -1.]:
            h_value = a_value * h_value + 2. * value
        return .5 * (h_value - 1.5) ** 2
    epsilon = 1e-6
    finite_a = (scalar_loss(.5+epsilon) - scalar_loss(.5-epsilon)) / (2*epsilon)
    check('中心差分', finite_a, -6.)
    ''')
    l.ex('r05', '检查一次有限差分', '调用 scalar_loss，在 a=.5 两侧取 ε=1e−5，算中心差分。',
         '''
    ex_r05_eps = 1e-5
    ex_r05_gradient = None  # TODO
    ex_r05_expected = -6.
    ''', '分子是 L(a+ε)−L(a−ε)，分母是 2ε。',
         '''
    ex_r05_eps = 1e-5
    ex_r05_gradient = (scalar_loss(.5+ex_r05_eps)-scalar_loss(.5-ex_r05_eps))/(2*ex_r05_eps)
    ex_r05_expected = -6.
    ''', "check('有限差分梯度', ex_r05_gradient, -6., atol=1e-5)")
    l.md(r'''
    ## 4 扩展到向量与每步损失

    标量状态只有一个存储位置。把它扩展为 H 个分量后，不同分量可以保留不同的历史组合。按行存样本时，$W_x$ 为 `(F,H)`，$W_h$ 为 `(H,H)`：
    $$z_t=x_tW_x+h_{t-1}W_h+b,\quad h_t=\tanh z_t.$$
    刚才只有最后一步收到目标；后面的滤波预测每步都有损失 $\ell_t$。令 $L=\sum_t\ell_t$，记当前损失带来的敏感度为 $e_t=\partial\ell_t/\partial h_t$，经过 tanh 之前的总敏感度为 $\Delta_t=\partial L/\partial z_t$，则
    $$\Delta_t=(e_t+\Delta_{t+1}W_h^\top)\odot(1-h_t^2).$$
    括号内的第一项来自当前目标，第二项来自下一步再往后的影响。两条支路先在 $h_t$ 处相加，再乘 tanh 的局部导数 $1-h_t^2$。只有末端目标的例子中，前面时刻的 $e_t$ 都为 0，回传只剩下沿时间边的连乘。若训练使用平均损失，平均系数包含在 $e_t$ 中。
    ''')
    l.code('''
    steps = np.arange(31)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for shared_a in [.5, 1., 1.5]:
        ax.semilogy(steps, np.abs(shared_a ** steps), label=f'a={shared_a}')
    ax.set(xlabel='Number of linear recurrent links', ylabel='Absolute sensitivity')
    ax.legend(frameon=False)
    show(fig)
    ''')
    l.md('横轴每增加 1，敏感度就多乘一次 a。a=.5 的曲线逐步衰减，a=1.5 的曲线逐步增长；这些差异来自同一种运算反复出现。向量 RNN 还会乘各步的 tanh 导数与状态权重矩阵。接下来把这条可求导的时间链接到开头的滤波任务。')
    l.md(r'''
    ## 5 合成一批相互独立的序列

    每条序列独立生成输入 $u_t$，真实状态按 $y_{t+1}=0.8y_t+0.2u_t$ 更新，$y_0=0$。一次输入对下一状态的贡献为 .2，再往后一时刻还会留下 $.2\times.8$，随后继续衰减。

    模型每步只接收 $u_t$，目标是该步更新后的 $y_{t+1}$，因而需要在自己的 h 中保存过去的影响。320 条序列各有 24 步，前 240 条用于训练，后 80 条用于测试；每条序列内部保持时间顺序。
    ''')
    l.code('''
    generator = torch.Generator().manual_seed(101)
    inputs = torch.randn(320, 24, 1, generator=generator)
    target_steps, true_state = [], torch.zeros(320, 1)
    for t in range(inputs.shape[1]):
        true_state = .8 * true_state + .2 * inputs[:, t]
        target_steps.append(true_state)
    targets = torch.stack(target_steps, dim=1)
    train_X, test_X = inputs[:240], inputs[240:]
    train_Y, test_Y = targets[:240], targets[240:]
    ''')
    l.ex('r06', '给下一时刻预测对齐输入与目标', '两条序列各四个时刻。取前三个时刻作输入，后三个时刻作目标，序列轴保持不变。',
         '''
    ex_r06_series = np.array([[0.,1.,2.,3.], [10.,11.,12.,13.]])[:, :, None]
    ex_r06_X = None  # TODO
    ex_r06_Y = None  # TODO
    ''', '`series[:,:-1,:]` 与 `series[:,1:,:]` 相差一个时刻。',
         '''
    ex_r06_series = np.array([[0.,1.,2.,3.], [10.,11.,12.,13.]])[:, :, None]
    ex_r06_X = ex_r06_series[:, :-1, :]
    ex_r06_Y = ex_r06_series[:, 1:, :]
    ''', '''
    check('输入时刻', ex_r06_X, np.array([[0.,1.,2.],[10.,11.,12.]])[:,:,None])
    check('目标时刻', ex_r06_Y, np.array([[1.,2.,3.],[11.,12.,13.]])[:,:,None])
    ''')
    l.md('`nn.Linear(F,H)` 在内部存 `(H,F)` 的权重，调用时执行 `x @ weight.T + bias`。下面将输入线性、旧状态线性和读出层逐项写在类中。每条新序列从全零状态开始。')
    l.code('''
    class TinyRNN(nn.Module):
        def __init__(self, hidden=8):
            super().__init__()
            self.input_map = nn.Linear(1, hidden)
            self.state_map = nn.Linear(hidden, hidden, bias=False)
            self.readout = nn.Linear(hidden, 1)
        def forward(self, X):
            h = X.new_zeros(X.shape[0], self.state_map.in_features)
            outputs = []
            for t in range(X.shape[1]):
                h = torch.tanh(self.input_map(X[:, t]) + self.state_map(h))
                outputs.append(self.readout(h))
            return torch.stack(outputs, dim=1)
    ''')
    l.md('模型内部的 `for t` 负责走完一条时间链；下面的 `for epoch` 负责反复学习同一批训练序列。一次前向保留了全部时刻的计算关系，`loss.backward()` 会沿这些关系回到共享参数，随后优化器统一更新它们。')
    l.code('''
    torch.manual_seed(102)
    model = TinyRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    losses = []
    for epoch in range(160):
        optimizer.zero_grad()
        loss = (model(train_X) - train_Y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ''')
    l.code('''
    with torch.no_grad():
        prediction = model(test_X)
        test_mse = (prediction-test_Y).square().mean().item()
    print('independent-sequence MSE:', test_mse)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    axes[0].semilogy(losses); axes[0].set(xlabel='Epoch', ylabel='Training MSE')
    axes[1].plot(test_Y[0, :, 0], label='True filter state')
    axes[1].plot(prediction[0, :, 0], '--', label='RNN prediction')
    axes[1].set(xlabel='Time step', ylabel='State'); axes[1].legend(frameon=False)
    show(fig)
    ''')
    l.md('右图取一条独立测试序列，将真实滤波状态和预测放在同一时间轴上。真实曲线中，峰值以后仍然延续的部分来自旧状态的 .8 倍保留；新输入通过 .2 的系数改变当前状态。比较两条曲线的幅度、转折和回落速度，就能看到模型在哪些时刻偏离了这个过程；打印的 MSE 汇总了全部 80 条测试序列。')
    l.md(r'''
    ## 6 自己做一次时间划分实验

    当前数据是一批独立序列。若数据来自一条长实验记录，先按时间截出训练段、验证段与未来测试段，再在各段内部做滑动窗口。相邻窗口共享大量采样点，分段边界应先于窗口生成。

    只把生成方程中的 0.8 改为 0.98，保持序列数、长度、模型、优化器和训练轮数不变。记录早期输入梯度、测试 MSE 和预测滞后；再单独增加序列长度，比较记忆跨度变化。
    ''')
    l.reading('Pascanu、Mikolov、Bengio：On the difficulty of training RNNs',
              'https://arxiv.org/abs/1211.5063',
              '找出梯度的时间乘积表示。梯度裁剪改变了哪一个量？用本课的 a=.5 线性链说明：把最大梯度限制住，对已经衰减的梯度有什么影响？')
    return l


def lstm_lesson():
    l = Lesson('11_lstm_gates_and_memory', '11 LSTM：把门、记忆和导数逐项拆开',
               ['手算一个带门记忆单元', '核对自写单元与 nn.LSTM 的权重顺序', '训练跨越干扰输入的记忆任务'],
               'RNN、BPTT、sigmoid、tanh、PyTorch 模型')
    l.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(11)
    style()
    ''')
    l.md(r'''
    序列第一步给出一个 ±0.8 的数，接下来 23 步都是小幅随机干扰，最后才要求读回最初的数。接收新输入时，需要保留有用的旧内容，也需要控制新内容写入多少。本课用这个延迟记忆任务理解 LSTM 的门。

    ## 1 用一个数控制保留比例

    sigmoid 定义为 $\sigma(z)=1/(1+e^{-z})$，把实数 z 映射到 0 至 1。将它的输出乘在旧记忆上，就能连续调节保留比例。
    先取旧记忆 $c_{t-1}=1$，遗忘门 $f_t=0.8$；保留下来的量是 $0.8\times1=0.8$。后面将让门值也根据输入与旧状态计算。
    ''')
    l.code('''
    logits = np.linspace(-6, 6, 100)
    gate_values = 1 / (1 + np.exp(-logits))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(logits, gate_values)
    ax.set(xlabel='Gate logit z', ylabel='Sigmoid gate value')
    show(fig)
    ''')
    l.ex('l01', '从 logit 得到门', 'z=0 时门值是多少？旧记忆是 2，经过这个门剩多少？',
         '''
    ex_l01_z = 0.
    ex_l01_gate = None  # TODO
    ex_l01_kept = None  # TODO
    ''', '直接代入 sigmoid，再乘旧记忆。',
         '''
    ex_l01_z = 0.
    ex_l01_gate = 1 / (1 + np.exp(-ex_l01_z))
    ex_l01_kept = ex_l01_gate * 2.
    ''', '''
    check('门值', ex_l01_gate, .5)
    check('保留量', ex_l01_kept, 1.)
    ''')
    l.md(r'''
    ## 2 两个状态，四个受学习控制的量

    x 是当前输入，c 保存内部记忆，h 提供当前读出，也用于下一步的门值计算。三个门 i、f、o 分别控制写入、保留、读出，候选 g 提供新内容。
    先把保留项与写入项相加得到 c，再从 c 计算 h。向量中每个分量都做这样的更新；各门有自己的权重与偏置，下面用 z 表示它们各自的仿射结果：
    $$i_t=\sigma(z_i),\quad f_t=\sigma(z_f),\quad g_t=\tanh(z_g),\quad o_t=\sigma(z_o),$$
    $$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).$$
    每个 $z_q=x_tW_{xq}+h_{t-1}W_{hq}+b_q$。符号 $\odot$ 表示对应分量相乘。
    ''')
    l.code('''
    f, i, g, o = .8, .2, .5, .7
    c_previous = 1.
    retained = f * c_previous
    written = i * g
    c_next = retained + written
    h_next = o * np.tanh(c_next)
    print('retained / written / cell / hidden:', retained, written, c_next, h_next)
    ''')
    l.md('旧内容留下 .8，新内容写入 .2×.5=.1，合成的新记忆 c 为 .9。读出先取 tanh(.9)，再乘输出门 .7。c 与 h 都传到下一时刻：c 沿记忆通路继续保留和更新，h 参与下一步四个门的计算。')
    l.ex('l02', '一步记忆更新', 'f=.9，i=.1，g=−.5，o=.8，旧 c=2，求新 c 和新 h。',
         '''
    ex_l02_c = None  # TODO: 保留项加写入项
    ex_l02_h = None  # TODO: 读出门乘 tanh
    ex_l02_previous = 2.
    ''', 'c=.9×2+.1×(−.5)。随后计算 .8×tanh(c)。',
         '''
    ex_l02_previous = 2.
    ex_l02_c = .9 * ex_l02_previous + .1 * (-.5)
    ex_l02_h = .8 * np.tanh(ex_l02_c)
    ''', '''
    check('新记忆', ex_l02_c, 1.75)
    check('新隐藏状态', ex_l02_h, .8*np.tanh(1.75))
    ''')
    l.md(r'''
    ## 3 沿记忆通路回传，再把门的支路接上

    新记忆由四个量共同决定。沿每一条输入边单独看：旧记忆先乘 f，遗忘门先乘旧记忆，输入门与候选值互相相乘。因此固定当前门值时，直接记忆通路的局部导数为
    $$\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{i,f,g}=f_t,\quad
    \frac{\partial c_t}{\partial f_t}=c_{t-1},\quad
    \frac{\partial c_t}{\partial i_t}=g_t,\quad
    \frac{\partial c_t}{\partial g_t}=i_t.$$
    读出还有 $\partial h_t/\partial c_t=o_t[1-\tanh^2(c_t)]$。末端损失沿读出回到 c 后，会分别乘上以上局部导数，继续传给旧记忆和各个门。

    把多个时刻接起来，直接记忆通路每经过一步就乘一次 f，得到 $\prod_t f_t$。同时，旧 h 会改变下一步各门，形成 $h_{t-1}\to z_q\to$ 各门的支路。完整 BPTT 会把这些支路的贡献也传回来，在同一个旧状态处相加。
    ''')
    l.code('''
    c0 = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    f0 = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    i0 = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    g0 = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    c1 = f0*c0 + i0*g0
    direct_grads = torch.autograd.grad(c1, (c0, f0, i0, g0))
    print('dc/d(c_previous,f,i,g):', [v.item() for v in direct_grads])
    check('四条局部导数', torch.stack(direct_grads), [.8, 1., .5, .2])
    ''')
    l.ex('l03', '将读出导数继续传回旧记忆', '门值固定为 f=.8、i=.2、g=.5、o=.7，旧 c=1。用链式法则求 dh_new/dc_previous。',
         '''
    ex_l03_c = .8 * 1. + .2 * .5
    ex_l03_gradient = None  # TODO
    ex_l03_f = .8
    ''', '先通过 o*tanh(c)，再通过直接通路的 f。',
         '''
    ex_l03_c = .8 * 1. + .2 * .5
    ex_l03_f = .8
    ex_l03_gradient = .7 * (1-np.tanh(ex_l03_c)**2) * ex_l03_f
    ''', "check('读出到旧记忆的导数', ex_l03_gradient, .7*(1-np.tanh(.9)**2)*.8)")
    l.code('''
    horizon = np.arange(101)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for forget in [.5, .9, .99]:
        ax.semilogy(horizon, forget**horizon, label=f'f={forget}')
    ax.set(xlabel='Number of memory links', ylabel='Direct-path sensitivity')
    ax.legend(frameon=False)
    show(fig)
    ''')
    l.ex('l04', '把 logit 的导数也算进去', 'i=sigmoid(z)，c=.8×1+i×.5；取 i=.2。计算 dc/dz。',
         '''
    ex_l04_i = .2
    ex_l04_gradient = None  # TODO: 候选值 × sigmoid 导数
    ex_l04_candidate = .5
    ''', 'sigmoid 的导数是 i(1−i)，并乘上 dc/di=g。',
         '''
    ex_l04_i = .2
    ex_l04_candidate = .5
    ex_l04_gradient = ex_l04_candidate * ex_l04_i * (1-ex_l04_i)
    ''', "check('门 logit 的梯度', ex_l04_gradient, .08)")
    l.md(r'''
    ## 4 把四次仿射合成一个张量

    四个门使用同一份 x 和旧 h，各有一组仿射参数。把四组参数并排存放，就能用两次矩阵乘法一次算出四组 z。

    F 为输入特征数，H 为隐藏宽度。输入是 `(B,F)`，旧 h、c 都是 `(B,H)`。一个 `Linear(F,4H)` 与一个 `Linear(H,4H)` 相加，得到 `(B,4H)`；沿最后一维分成 i、f、g、o 四块，每块 `(B,H)`。这个顺序与 PyTorch LSTM 一致。
    ''')
    l.code('''
    packed = torch.arange(12.).reshape(1, 12)
    pieces = packed.chunk(4, dim=-1)
    for name, piece in zip(['i', 'f', 'g', 'o'], pieces):
        print(name, piece)
    ''')
    l.ex('l05', '找到遗忘门那一块', 'H=2，打包向量为 [0,1,2,3,4,5,6,7]。取 f 的两个 logit。',
         '''
    ex_l05_packed = torch.arange(8.).reshape(1, 8)
    ex_l05_forget = None  # TODO
    ex_l05_H = 2
    ''', 'i 占前 H 列，f 占第 H 到 2H 列。',
         '''
    ex_l05_packed = torch.arange(8.).reshape(1, 8)
    ex_l05_H = 2
    ex_l05_forget = ex_l05_packed[:, ex_l05_H:2*ex_l05_H]
    ''', "check('遗忘门 logit', ex_l05_forget, [[2.,3.]])")
    l.md('下面把遗忘门的输入偏置初始化为 2，其余偏置为 0。输入和旧 h 都为零时，遗忘门由 sigmoid(2) 决定，开始时倾向保留较多旧记忆。实际门值还会受到输入、旧状态和权重的共同影响；偏置也随训练更新。')
    l.code('''
    class ManualLSTMCell(nn.Module):
        def __init__(self, features, hidden):
            super().__init__()
            self.hidden = hidden
            self.i2h = nn.Linear(features, 4*hidden)
            self.h2h = nn.Linear(hidden, 4*hidden)
            with torch.no_grad():
                self.i2h.bias.zero_(); self.h2h.bias.zero_()
                self.i2h.bias[hidden:2*hidden].fill_(2.)
        def forward(self, x, h, c):
            zi, zf, zg, zo = (self.i2h(x)+self.h2h(h)).chunk(4, dim=-1)
            i, f = torch.sigmoid(zi), torch.sigmoid(zf)
            g, o = torch.tanh(zg), torch.sigmoid(zo)
            c = f*c + i*g
            h = o*torch.tanh(c)
            return h, c, torch.stack((i, f, g, o), dim=1)
    ''')
    l.md('现在将同一组参数交给两种实现，检查刚才的公式是否逐步对得上。用 float64 把手写单元的两组权重、两组偏置原样复制给 nn.LSTM；两组偏置在前向中相加。给它们完全相同的四步输入，再比较所有时刻的 h 与最后的 c。首次出现差异的时刻，可以帮助定位门顺序或状态更新中的问题。')
    l.code('''
    manual = ManualLSTMCell(2, 3).double()
    builtin = nn.LSTM(2, 3, batch_first=True).double()
    with torch.no_grad():
        builtin.weight_ih_l0.copy_(manual.i2h.weight)
        builtin.weight_hh_l0.copy_(manual.h2h.weight)
        builtin.bias_ih_l0.copy_(manual.i2h.bias)
        builtin.bias_hh_l0.copy_(manual.h2h.bias)
    alignment_X = torch.randn(2, 4, 2, dtype=torch.float64)
    ''')
    l.code('''
    h, c = torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)
    manual_outputs = []
    for t in range(4):
        h, c, gates = manual(alignment_X[:, t], h, c)
        manual_outputs.append(h)
    reference_output, (reference_h, reference_c) = builtin(alignment_X)
    check('所有时刻的 h', torch.stack(manual_outputs, dim=1), reference_output, atol=1e-10)
    check('最终 c', c, reference_c[0], atol=1e-10)
    ''')
    l.ex('l06', '两组偏置怎样进入同一个门', '输入仿射给出 [.1,.2,.3,.4]，状态仿射给出 [.5,.6,.7,.8]。H=1，取合并后 f 的 logit 与门值。',
         '''
    ex_l06_input = np.array([.1,.2,.3,.4])
    ex_l06_state = np.array([.5,.6,.7,.8])
    ex_l06_zf = None  # TODO
    ex_l06_f = None  # TODO
    ''', '两数组先相加，f 位于索引 1，再计算 sigmoid。',
         '''
    ex_l06_input = np.array([.1,.2,.3,.4])
    ex_l06_state = np.array([.5,.6,.7,.8])
    ex_l06_zf = (ex_l06_input+ex_l06_state)[1]
    ex_l06_f = 1/(1+np.exp(-ex_l06_zf))
    ''', '''
    check('f 的 logit', ex_l06_zf, .8)
    check('f 门值', ex_l06_f, 1/(1+np.exp(-.8)))
    ''')
    l.md(r'''
    ## 5 让记忆跨过 23 步干扰

    回到开头的任务。每条序列长 24，第一时刻输入随机的 ±0.8，第二通道同时给出“写入标记”1；后续时刻第一通道只有独立小噪声，标记为 0。标记告诉网络哪一步的内容需要记住，目标是在最后一步读回那个数。

    训练和测试使用独立生成的整条序列。平方误差只从最后读出进入计算图；第一步的输入和门要得到学习信号，梯度就需要经过中间 23 次状态更新。
    ''')
    l.code('''
    generator = torch.Generator().manual_seed(111)
    memory_X = .15*torch.randn(384, 24, 2, generator=generator)
    memory_target = (torch.randint(0, 2, (384, 1), generator=generator)*2-1).float()*.8
    memory_X[:, :, 1] = 0.
    memory_X[:, 0, 0] = memory_target[:, 0]
    memory_X[:, 0, 1] = 1.
    memory_train_X, memory_test_X = memory_X[:288], memory_X[288:]
    memory_train_Y, memory_test_Y = memory_target[:288], memory_target[288:]
    ''')
    l.code('''
    class MemoryLSTM(nn.Module):
        def __init__(self, hidden=12):
            super().__init__()
            self.cell = ManualLSTMCell(2, hidden)
            self.readout = nn.Linear(hidden, 1)
        def forward(self, X):
            h = X.new_zeros(X.shape[0], self.cell.hidden)
            c = torch.zeros_like(h)
            gate_trace, cell_trace = [], []
            for t in range(X.shape[1]):
                h, c, gates = self.cell(X[:, t], h, c)
                gate_trace.append(gates); cell_trace.append(c)
            return self.readout(h), torch.stack(gate_trace, dim=1), torch.stack(cell_trace, dim=1)
    ''')
    l.code('''
    torch.manual_seed(112)
    memory_model = MemoryLSTM()
    optimizer = torch.optim.Adam(memory_model.parameters(), lr=.02)
    memory_losses = []
    for epoch in range(240):
        optimizer.zero_grad()
        output, _, _ = memory_model(memory_train_X)
        loss = (output-memory_train_Y).square().mean()
        loss.backward(); optimizer.step()
        memory_losses.append(loss.item())
    ''')
    l.code('''
    with torch.no_grad():
        memory_output, gate_trace, cell_trace = memory_model(memory_test_X)
    print('test MSE:', (memory_output-memory_test_Y).square().mean().item())
    print('zero-prediction MSE:', memory_test_Y.square().mean().item())
    print('target / prediction:', torch.cat([memory_test_Y[:6], memory_output[:6]], dim=1))
    trace_unit = int(cell_trace[0].var(dim=0).argmax())
    ''')
    l.md('先看前六条测试序列的目标与预测，再选测试序列 0 中随时间变化最大的记忆分量。下图始终画这个分量的 i、f、o、c 和候选 g。逐时刻对照 f×旧 c 与 i×g，就能将记忆的变化接回刚才的一步公式。候选 g 的范围是 −1 至 1；c 由保留项和写入项累加，可以超出这个范围，读出 h 再经过 tanh。')
    l.code('''
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout='constrained')
    axes[0].semilogy(memory_losses); axes[0].set(xlabel='Epoch', ylabel='Training MSE')
    for gate_index, name in [(0,'input'), (1,'forget'), (3,'output')]:
        axes[1].plot(gate_trace[0, :, gate_index, trace_unit], label=name)
    axes[1].set(xlabel='Time step', ylabel=f'Gate, unit {trace_unit}', ylim=(-.03,1.03))
    axes[1].legend(frameon=False)
    axes[2].plot(cell_trace[0, :, trace_unit], label='Cell state')
    axes[2].plot(gate_trace[0, :, 2, trace_unit], '--', label='Candidate g')
    axes[2].set(xlabel='Time step', ylabel='Value'); axes[2].legend(frameon=False)
    show(fig)
    ''')
    l.md('如果一段时间里候选 g 持续同号、输入门 i 持续开启，写入项就可能累积；遗忘门 f 接近 1 时，旧内容也保留得较多。读门值时，把相邻两步 c 的数值一起看，判断这一段变化主要来自保留还是写入。')
    l.md('把延迟从 24 改为 48，其余设置保留，时间链就增加了 24 次状态更新。记录测试误差与第一时刻输入梯度，比较记忆能否跨过更长的干扰段。另做一次实验，将遗忘门初始偏置从 2 改为 0；对照直接通路乘积与训练后的实际门值，观察初始化怎样影响学习过程。')
    l.md('权重布局可继续查 [PyTorch LSTM 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)。把自写单元的 g、o 块对调，运行逐时刻核对，定位首次偏离的状态。')
    l.reading('Hochreiter 与 Schmidhuber：Long Short-Term Memory（1997）',
              'https://ml.jku.at/publications/older/2604.pdf',
              '在原文图中找到线性记忆通路、输入门与输出门，画出与本课公式的对应。继续阅读 [Gers 等 Learning to Forget（2000）](https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf)：持续输入流为什么需要可学习的重置机制？把本课改成连续出现两次写入标记的任务，检查遗忘门怎样处理旧内容。')
    return l


def gnn_lesson():
    l = Lesson('16_gnn_message_passing', '16 图神经网络：从邻居求和写到 GCN',
               ['手算邻接表与消息聚合', '写出归一化 GCN 并验证置换等变', '用训练节点标签学习，检查传播过深的变化'],
               '矩阵乘法、分类交叉熵、PyTorch 训练循环')
    l.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(16)
    style()
    ''')
    l.md(r'''
    一个节点自己的观测可能很嘈杂，而与它相连的节点提供了另一份线索。本课后面的分类数据就这样生成：每个节点有四个带噪声的数，同组节点之间更容易连边。模型需要学习怎样结合节点自身与邻居的信息。

    ## 1 一行邻接矩阵，列出一个点接收谁的消息

    图包含 N 个节点。无向图的邻接矩阵 A 为 `(N,N)`，$A_{ij}=1$ 表示节点 i 与 j 相连。第 i 行列出 i 的邻居。
    节点特征矩阵 X 为 `(N,F)`，每行 F 个实数。先把图缩到四个节点：0—1—2，节点 3 单独存在。节点 1 的两个邻居各有一行特征，矩阵乘法 `A @ X` 会按 A 中的 1 取出这些行，再相加。
    ''')
    l.code('''
    A = np.array([[0.,1.,0.,0.], [1.,0.,1.,0.], [0.,1.,0.,0.], [0.,0.,0.,0.]])
    X = np.array([[1.,0.], [0.,2.], [3.,1.], [2.,2.]])
    print('neighbors of node 1:', np.flatnonzero(A[1]))
    print('their features:', X[A[1].astype(bool)])
    print('A @ X:', A @ X)
    ''')
    l.ex('g01', '把邻居求和写成一行', '节点 1 接收节点 0、2 的特征。取出两行并相加，再用邻接矩阵乘法核对。',
         '''
    ex_g01_X = np.array([[1.,0.], [0.,2.], [3.,1.], [2.,2.]])
    ex_g01_sum = None  # TODO
    ex_g01_degree = None  # TODO: 节点 1 的邻居数
    ''', '整数行索引 [0,2] 取两个邻居；沿 axis=0 对这两行求和。',
         '''
    ex_g01_X = np.array([[1.,0.], [0.,2.], [3.,1.], [2.,2.]])
    ex_g01_sum = ex_g01_X[[0, 2]].sum(axis=0)
    ex_g01_degree = 2
    ''', '''
    check('邻居特征和', ex_g01_sum, [4.,1.])
    check('邻居数', ex_g01_degree, 2)
    ''')
    l.md('节点 1 收到 [1,0] 与 [3,1]，相加为 [4,1]。下图节点内写原始特征，每条青色箭头对应一份发往节点 1 的向量。图中的加号与 `A @ X` 第 1 行做的是同一次计算。')
    l.code(r'''
    small_pos = np.array([[0.,0.], [1.,.5], [2.,0.], [3.2,.3]])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for j in [0, 2]:
        ax.annotate('', small_pos[1], small_pos[j], arrowprops=dict(arrowstyle='->', color='#37798A', lw=2, shrinkA=28, shrinkB=28))
    for j, point in enumerate(small_pos):
        ax.text(*point, f'{j}: {X[j].tolist()}', ha='center', bbox=dict(fc='white', ec='#37798A'))
    ax.text(1., .85, r'$\Sigma=[4,1]$', ha='center')
    ax.set(xlim=(-.5,3.9), ylim=(-.4,1.)); ax.axis('off')
    show(fig)
    ''')
    l.md(r'''
    ## 2 自环与度数归一化

    刚才节点 1 的聚合结果只包含邻居，原有的 [0,2] 没有参加相加。增加单位阵 I，就给每个节点加上一条发给自己的消息：$\widetilde A=A+I$。

    邻居越多，直接求和包含的项也越多。GCN 在每条消息上乘一个由两端度数决定的系数。新度数 $\widetilde d_i=\sum_j\widetilde A_{ij}$，对称归一化权重为
    $$\widehat A_{ij}=\frac{\widetilde A_{ij}}{\sqrt{\widetilde d_i\widetilde d_j}},\qquad
    \widehat A=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}.$$
    $\widetilde D$ 是把度数放在对角线上的矩阵。左乘 $\widetilde D^{-1/2}$ 按接收端缩放行，右乘它按发送端缩放列；所以边 i、j 的权重同时包含两端度数。
    ''')
    l.code('''
    A_self = A + np.eye(len(A))
    degree = A_self.sum(axis=1)
    inverse_sqrt = degree ** -.5
    A_hat = inverse_sqrt[:, None] * A_self * inverse_sqrt[None, :]
    print('degrees:', degree)
    print('normalized adjacency:', np.round(A_hat, 4))
    print('aggregated features:', A_hat @ X)
    ''')
    l.ex('g02', '孤立节点加入自环以后', '节点 3 原来没有邻居。求加自环后的度数，以及聚合后它自己的特征。',
         '''
    ex_g02_degree = None  # TODO
    ex_g02_feature = None  # TODO
    ex_g02_node = 3
    ''', '自环给它一条自己到自己的边，权重为 1/√(1×1)。',
         '''
    ex_g02_node = 3
    ex_g02_degree = (A+np.eye(4)).sum(axis=1)[ex_g02_node]
    ex_g02_feature = (A_hat @ X)[ex_g02_node]
    ''', '''
    check('加自环后的度数', ex_g02_degree, 1.)
    check('保留的节点特征', ex_g02_feature, [2.,2.])
    ''')
    l.md(r'''
    节点 1 的加自环度数为 3，节点 0、2 的为 2。因此它收到的三个加权向量是
    $$\frac{X_0}{\sqrt6},\quad\frac{X_1}{3},\quad\frac{X_2}{\sqrt6}.$$
    两条邻居消息的系数为 $1/\sqrt{3\times2}$，自己的消息为 $1/\sqrt{3\times3}$。把这三项逐坐标相加，再对照 `A_hat @ X` 的第 1 行。若改用行归一化平均，三条消息的系数才都是 1/3。
    ''')
    l.ex('g03', '手写一行归一化聚合', '按三个向量的权重计算节点 1 的新特征。',
         '''
    ex_g03_messages = None  # TODO: 相加三个加权向量
    ex_g03_scale = np.sqrt(6.)
    ex_g03_self_degree = 3.
    ''', 'X[0]/√6 + X[1]/3 + X[2]/√6。',
         '''
    ex_g03_scale = np.sqrt(6.)
    ex_g03_self_degree = 3.
    ex_g03_messages = X[0]/ex_g03_scale + X[1]/ex_g03_self_degree + X[2]/ex_g03_scale
    ''', "check('节点 1 的归一化消息', ex_g03_messages, [4/np.sqrt(6), 2/3+1/np.sqrt(6)])")
    l.md(r'''
    ## 3 所有节点共享同一个线性映射

    聚合规定了信息从哪里来，还需要学习怎样处理这些特征。W 的形状为 `(F,H)`，把每个节点的 F 维特征映射为 H 维。同一层的所有节点使用同一组 W；某种特征组合有用时，这条处理规则可以用于图中任何节点。

    把第 l 层全体节点的表示记作 $H^{(l)}$，初值 $H^{(0)}=X$：
    $$H^{(l+1)}=\operatorname{ReLU}(\widehat A H^{(l)}W^{(l)}).$$
    右侧先用 W 改变特征通道，再用 $\widehat A$ 混合节点；这两个线性步骤也可交换结合顺序。ReLU 最后逐项把负数置零。下一格保留三个中间张量，能逐行核对一次完整传播。
    ''')
    l.code('''
    W = np.array([[1., -.5], [.5, 1.]])
    transformed = X @ W
    aggregated = A_hat @ transformed
    H = np.maximum(aggregated, 0.)
    print('shared transform:', transformed)
    print('after aggregation and ReLU:', H)
    ''')
    l.md(r'''
    ## 4 换一套节点编号，结果也只换编号

    给节点重新编号，连接关系和特征所描述的对象仍然相同。一个节点原先得到的结果，也应跟着它移动到新的行。

    置换矩阵 P 每行每列恰有一个 1，表示这种重排。重排输入 $X'=PX$、图 $A'=PAP^\top$ 后，GCN 输出应为 $H'=PH$，这叫置换等变。A 的行表示接收者，列表示发送者，因此两个轴都要同步重排。下面保持 W 不变，用同一组特征和边检查这个关系。
    ''')
    l.code('''
    permutation = np.array([2, 0, 3, 1])
    P = np.eye(4)[permutation]
    permuted_features = P @ X
    permuted_graph = P @ A_hat @ P.T
    permuted_output = np.maximum(permuted_graph @ permuted_features @ W, 0.)
    check('节点重排等变', permuted_output, P @ H)
    ''')
    l.ex('g04', '只重排特征会发生什么', '正确重排需要同步处理邻接矩阵的两个轴。构造 A_hat[np.ix_(p,p)] 并计算新输出。',
         '''
    ex_g04_p = np.array([3, 2, 1, 0])
    ex_g04_A = None  # TODO
    ex_g04_output = None  # TODO
    ''', '新特征是 X[p]；A 的新第 i、j 项来自旧第 p[i]、p[j] 项。',
         '''
    ex_g04_p = np.array([3, 2, 1, 0])
    ex_g04_A = A_hat[np.ix_(ex_g04_p, ex_g04_p)]
    ex_g04_output = np.maximum(ex_g04_A @ X[ex_g04_p] @ W, 0.)
    ''', "check('同步重排的输出', ex_g04_output, H[[3,2,1,0]])")
    l.md(r'''
    ## 5 一张图里的节点分类任务

    现在把小图中的运算用于开头的分类任务。生成 120 个节点，两组各 60 个。同组连边概率 .16，跨组 .015；四维特征为噪声，其中第 0 维带较弱的组别偏移。单看一个节点的特征，组别信号容易被噪声盖住；沿边收集邻居特征，可以利用同组更常相连的结构。

    组号是待预测标签。每组取 10 个训练节点、10 个验证节点，其余用于测试，共有 20 个训练标签。前向使用整张已观察到的图与节点特征；训练损失只读取这 20 个标签。这是同一图上的传导式节点分类。
    ''')
    l.code('''
    rng = np.random.default_rng(161)
    labels_np = np.repeat([0, 1], 60)
    probabilities = np.where(labels_np[:, None] == labels_np[None, :], .16, .015)
    upper = np.triu(rng.random((120,120)) < probabilities, k=1)
    graph_np = (upper | upper.T).astype(float)
    features_np = rng.normal(size=(120, 4))
    features_np[:, 0] += .35 * (2*labels_np-1)
    ''')
    l.code('''
    train_mask = torch.zeros(120, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    for group in [0, 1]:
        ids = rng.permutation(np.flatnonzero(labels_np == group))
        train_mask[ids[:10]] = True
        val_mask[ids[10:20]] = True
    test_mask = ~(train_mask | val_mask)
    print('train / validation / test:', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    ''')
    l.code('''
    graph_self = graph_np + np.eye(120)
    graph_scale = graph_self.sum(axis=1) ** -.5
    graph_tensor = torch.tensor(graph_scale[:,None]*graph_self*graph_scale[None,:], dtype=torch.float32)
    features = torch.tensor(features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    print('A_hat / X / labels:', graph_tensor.shape, features.shape, labels.shape)
    ''')
    l.md('第一层把四维输入变成 12 维隐藏表示，并混合一跳邻居；第二层再次聚合，所以输出也能受到两跳范围内特征的影响。最后每个节点有两个类别分数，交叉熵内部计算 log-softmax。两层都使用刚才手算过的 `A_hat @ 特征`。')
    l.code('''
    class ManualGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 12, bias=False)
            self.second = nn.Linear(12, 2, bias=False)
        def forward(self, X, A_hat):
            hidden = torch.relu(A_hat @ self.first(X))
            return A_hat @ self.second(hidden)
    ''')
    l.ex('g05', '标签只从训练掩码进入损失', '4 个节点的 logits 全为 0，标签为 [0,1,1,0]，仅前两个节点参加交叉熵。求 logits 梯度。',
         '''
    ex_g05_logits = torch.zeros(4, 2, requires_grad=True)
    ex_g05_labels = torch.tensor([0,1,1,0])
    ex_g05_mask = torch.tensor([True,True,False,False])
    ex_g05_grad = None  # TODO
    ''', '计算 cross_entropy(logits[mask],labels[mask])，再用 autograd.grad。两训练节点的损失取均值。',
         '''
    ex_g05_logits = torch.zeros(4, 2, requires_grad=True)
    ex_g05_labels = torch.tensor([0,1,1,0])
    ex_g05_mask = torch.tensor([True,True,False,False])
    ex_g05_loss = F.cross_entropy(ex_g05_logits[ex_g05_mask], ex_g05_labels[ex_g05_mask])
    ex_g05_grad = torch.autograd.grad(ex_g05_loss, ex_g05_logits)[0]
    ''', "check('掩码后的分数梯度', ex_g05_grad, [[-.25,.25],[.25,-.25],[0.,0.],[0.,0.]])")
    l.md('这个练习中，后两个节点的分数梯度为零，因为它们没有直接参加损失。在完整 GCN 中，这些节点的特征仍能沿边影响训练节点的分数，进而影响共享参数的梯度。标签由 `train_mask` 选入损失，消息由图中的边传递，是两条不同的信息通路。下面训练 180 轮，最后分别读取验证与测试节点的准确率。')
    l.code('''
    torch.manual_seed(162)
    model = ManualGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03, weight_decay=5e-4)
    losses = []
    for epoch in range(180):
        optimizer.zero_grad()
        logits = model(features, graph_tensor)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    ''')
    l.code('''
    with torch.no_grad():
        predicted = model(features, graph_tensor).argmax(dim=1)
    val_accuracy = (predicted[val_mask] == labels[val_mask]).float().mean().item()
    test_accuracy = (predicted[test_mask] == labels[test_mask]).float().mean().item()
    print('validation / test accuracy:', val_accuracy, test_accuracy)
    angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    ''')
    l.code('''
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, coloring, title in zip(axes, [labels_np, predicted.numpy()], ['Generated labels', 'GCN predictions']):
        for left, right in np.argwhere(np.triu(graph_np, 1)):
            ax.plot(positions[[left,right],0], positions[[left,right],1], color='#B8C0C3', lw=.35, zorder=1)
        ax.scatter(*positions.T, c=np.array(['#37798A','#B66A47'])[coloring], s=25, zorder=2)
        ax.scatter(*positions[train_mask.numpy()].T, facecolors='none', edgecolors='black', s=70, zorder=3)
        ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
    show(fig)
    ''')
    l.md('左图颜色来自生成标签，右图颜色来自模型预测；黑色圆圈标出提供训练标签的节点。对照同一位置的两种颜色，可以看到预测错误落在哪些节点，再沿边查看它们接收的信息。')
    l.md(r'''
    ## 6 只改变传播次数，观察过平滑

    邻居聚合让一个节点获得更多信息，反复聚合也会不断混合原有差异。另建一个 24 节点环，每节点连接左右邻居，加自环后度数均为 3。固定初始特征与 W=I，反复做 $H_{k+1}=\widehat A H_k$，只考察传播次数的作用。

    在这个规则环上，每一步都将自己与左右邻居取平均。传播次数增加后，各节点特征趋向同一个全图均值。左图用节点间方差量化差异，右图沿节点编号画同一特征通道：起伏逐渐变平，就对应可区分信息的减少。
    ''')
    l.code('''
    ring = np.eye(24)
    for node in range(24):
        ring[node, (node-1)%24] = ring[node, (node+1)%24] = 1.
    ring /= 3.
    initial_features = np.random.default_rng(163).normal(size=(24, 3))
    propagation = [initial_features]
    for depth in range(64):
        propagation.append(ring @ propagation[-1])
    variances = [np.var(value, axis=0).mean() for value in propagation]
    ''')
    l.code('''
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    axes[0].semilogy(variances); axes[0].set(xlabel='Propagation count', ylabel='Between-node feature variance')
    for depth in [0, 4, 64]:
        axes[1].plot(propagation[depth][:,0], label=f'{depth} steps')
    axes[1].set(xlabel='Node index', ylabel='Feature channel 0')
    axes[1].legend(frameon=False)
    show(fig)
    ''')
    l.ex('g06', '检查局部平均保留什么', '对规则环做 20 次传播，比较每个特征通道在节点上的平均值是否改变。',
         '''
    ex_g06_H = initial_features.copy()
    ex_g06_mean = None  # TODO: 传播后对节点轴求均值
    ex_g06_steps = 20
    ''', '循环 H=ring@H；mean(axis=0) 沿节点轴平均。规则环矩阵的行和与列和都是 1。',
         '''
    ex_g06_H = initial_features.copy()
    ex_g06_steps = 20
    for ex_g06_depth in range(ex_g06_steps):
        ex_g06_H = ring @ ex_g06_H
    ex_g06_mean = ex_g06_H.mean(axis=0)
    ''', "check('保留的全图通道均值', ex_g06_mean, initial_features.mean(axis=0))")
    l.md('环上的各节点逐渐接近，但每个特征通道的全图均值保留下来。上面的检查把这两件事分开：传播可以保留总体平均，同时抹去节点之间的差异。实际分类需要的是后者能够区分组别。')
    l.md('回到分类图，只把跨组连边概率从 .015 改到 .08，保留组内概率、节点特征、划分和优化设置。跨组消息变多后，比较 GCN 与将 A_hat 换成 I 的同宽模型，记录训练误差、测试误差和节点表示距离。重复至少五个图随机种子，检验更频繁的跨组混合是否削弱了这份数据中的邻域线索。')
    l.reading('Kipf 与 Welling：Semi-Supervised Classification with GCNs',
              'https://arxiv.org/abs/1609.02907',
              '对应原文传播式逐项标出本课的自环、度数、X 和 W。为什么没有标签的节点仍可参加前向？附录深度实验中，层数增加同时改变了哪些变量？怎样用本课的规则环控制实验把传播效应单独取出来？')
    return l


def liquid_lesson():
    l = Lesson('17_liquid_neural_networks', '17 液体神经网络：从松弛方程到可训练 LTC',
               ['由解析解理解时间常数', '推导并实现液体单元的半隐式步', '学习不规则采样序列并检查有效时间常数'],
               'RNN、梯度、指数函数、常微分方程的导数含义')
    l.code('''
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(17)
    style()
    ''')
    l.md(r'''
    前面的序列模型每前进一步就执行一次更新。若采样不规则，同一个输入可能持续 .05，也可能持续 .4 个时间单位；状态有多长时间响应这个输入，会直接改变下一次观测。后面的实验将把每段持续时间也交给模型。

    ## 1 从一个物理松弛过程开始

    先看最简单的响应过程。h(t) 是随物理时间 t 变化的状态，u 是固定的平衡值，$\tau>0$ 是时间常数，单位与 t 相同：
    $$\frac{dh}{dt}=\frac{u-h}{\tau}.$$
    令 $v=h-u$，则 $dv/dt=-v/\tau$；积分得到 $v(t)=v(0)e^{-t/\tau}$，所以
    $$h(t)=u+[h(0)-u]e^{-t/\tau}.$$
    h 低于 u 时斜率为正，高于 u 时斜率为负；越接近平衡，变化越慢。经过一个时间常数，偏离平衡的量剩下原来的 $e^{-1}$。下图的三条曲线平衡值相同，差别只在接近平衡的速度。
    ''')
    l.code('''
    times = np.linspace(0, 6, 200)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for tau in [.5, 1., 2.]:
        state = 1-np.exp(-times/tau)
        ax.plot(times, state, label=f'tau={tau}')
    ax.set(xlabel='Physical time', ylabel='Relaxation state'); ax.legend(frameon=False)
    show(fig)
    ''')
    l.ex('q01', '半衰时间与时间常数', 'h(0)=0、u=2、τ=3，计算 t=τ ln2 时的 h。',
         '''
    ex_q01_tau = 3.
    ex_q01_time = None  # TODO
    ex_q01_state = None  # TODO
    ''', '把 t/τ=ln2 代入解析式，指数项等于 1/2。',
         '''
    ex_q01_tau = 3.
    ex_q01_time = ex_q01_tau*np.log(2.)
    ex_q01_state = 2. + (0.-2.)*np.exp(-ex_q01_time/ex_q01_tau)
    ''', "check('半衰时刻的状态', ex_q01_state, 1.)")
    l.md(r'''
    ## 2 数值步长也参与计算

    已有导数后，可以用一小段直线近似真实曲线。两次采样之间的时间差记作 $\Delta t$，显式 Euler 用起点斜率走一步：
    $$h_{n+1}=h_n+\Delta t\,(u-h_n)/\tau.$$
    减去 u 得 $h_{n+1}-u=(1-\Delta t/\tau)(h_n-u)$：对平衡值的偏离每步乘 $1-\Delta t/\tau$。这个系数的绝对值小于 1 时，误差逐步衰减，对应 $0<\Delta t/\tau<2$；绝对值超过 1 时，误差增长。
    ''')
    l.code('''
    def euler_relax(dt, tau=.6, total=6.):
        grid = np.arange(round(total/dt)+1)*dt
        values = [0.]
        for _ in grid[1:]:
            values.append(values[-1]+dt*(1-values[-1])/tau)
        return grid, np.array(values)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for dt in [.2, 1., 2.]:
        grid, values = euler_relax(dt); ax.plot(grid, values, 'o-', label=f'dt={dt}')
    ax.plot(times, 1-np.exp(-times/.6), 'k--', label='Exact')
    ax.set(xlabel='Physical time', ylabel='State'); ax.legend(frameon=False)
    show(fig)
    ''')
    l.md('图中 τ=.6。步长 .2 的偏离乘数为 2/3，状态逐步靠近平衡；步长 1 的乘数为 −2/3，状态越过平衡后交替接近；步长 2 的乘数为 −7/3，振荡幅度反而越来越大。三个数值轨迹使用同一条微分方程，差异来自离散步长。')
    l.ex('q02', '手算 Euler 一步', 'h=0、u=2、τ=2、Δt=.5，计算下一状态与误差乘数。',
         '''
    ex_q02_next = None  # TODO
    ex_q02_factor = None  # TODO
    ex_q02_dt = .5
    ''', '增量是 Δt×(u−h)/τ；误差乘数是 1−Δt/τ。',
         '''
    ex_q02_dt = .5
    ex_q02_next = 0.+ex_q02_dt*(2.-0.)/2.
    ex_q02_factor = 1-ex_q02_dt/2.
    ''', '''
    check('Euler 下一状态', ex_q02_next, .5)
    check('误差乘数', ex_q02_factor, .75)
    ''')
    l.md(r'''
    ## 3 让状态变化速度受到输入控制

    固定 τ 的松弛过程在所有输入下按同一个尺度响应。液体神经网络让这个尺度也受输入和当前状态控制。本课实现 Liquid Time-Constant（LTC）的一种简化单元：u 为外部输入，h 为隐藏状态，A 为可学习的驱动水平，f 是依赖 u、h 的非负门函数：
    $$\dot h=-\left(\frac1\tau+f(u,h)\right)h+f(u,h)A.$$
    取 $f\ge0$，瞬时衰减系数为 $k=1/\tau+f$，相应有效时间常数为
    $$\tau_{\rm eff}=\frac1{1/\tau+f}=\frac{\tau}{1+\tau f}.$$
    将 f 暂时固定，方程可写成 $\dot h=-(h-h_*)/\tau_{\rm eff}$，其中平衡值为 $h_*=(fA)/(1/\tau+f)$。因此 f 改变时，接近平衡的速度与平衡位置会一起改变。沿轨迹计算 $\tau_{\rm eff}$，就能看到模型在不同输入和状态下使用了怎样的响应尺度。
    下图先固定 h=0、τ=1.3，取门函数 $f(u,0)=\operatorname{softplus}(1.3u-.2)$。
    $\operatorname{softplus}(z)=\log(1+e^z)$ 把实数平滑地映到正数；`np.logaddexp(0,z)` 稳定计算同一个表达式。
    [LTC 原论文](https://arxiv.org/abs/2006.04439) 从这种依赖状态的动力学构造可训练网络。
    ''')
    l.code('''
    external = np.linspace(-3, 3, 100)
    positive_gate = np.logaddexp(0., 1.3*external-.2)
    tau_effective = 1/(1/1.3 + positive_gate)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout='constrained')
    axes[0].plot(external, positive_gate); axes[0].set(xlabel='Input u, with h=0', ylabel='Positive gate f')
    axes[1].plot(external, tau_effective); axes[1].set(xlabel='Input u, with h=0', ylabel='Effective time constant')
    show(fig)
    ''')
    l.md('这里固定了 h，因此横轴只改变外部输入。门 f 增大时，分母 1/τ+f 增大，有效时间常数缩短。恢复状态依赖以后，即使两次输入相同，旧状态不同也会产生不同的门值。')
    l.ex('q03', '分开基础 τ 与有效 τ', '基础 τ=2，三个门值为 [0,.5,1.5]，计算对应有效时间常数。',
         '''
    ex_q03_gate = np.array([0., .5, 1.5])
    ex_q03_effective = None  # TODO
    ex_q03_tau = 2.
    ''', '计算 1/(1/τ+f)，数组里的每个 f 各算一次。',
         '''
    ex_q03_gate = np.array([0., .5, 1.5])
    ex_q03_tau = 2.
    ex_q03_effective = 1/(1/ex_q03_tau+ex_q03_gate)
    ''', "check('三种有效时间常数', ex_q03_effective, [2.,1.,.5])")
    l.md(r'''
    ## 4 把线性状态项放在新时刻

    要从导数得到下一状态，需要选择一个数值更新式。在旧状态计算 $f_n=f(u_n,h_n)$，并在这一个子步内固定该门值；把线性衰减作用到新状态：
    $$h_{n+1}=h_n+\Delta t[-(1/\tau+f_n)h_{n+1}+f_nA].$$
    把所有含新状态的项移到左边，再除以系数，得到半隐式步：
    $$h_{n+1}=\frac{h_n+\Delta t f_n A}{1+\Delta t(1/\tau+f_n)}.$$
    当 $\Delta t>0$ 时，分母大于 1；固定门以后，旧状态的直接系数为这个分母的倒数。门在子步开始时仍由旧 h 算出，所以回传梯度时还需要经过 $h_n\to f_n\to h_{n+1}$。下面先看前向数值，再把这条支路的导数接上。
    ''')
    l.code('''
    h_old, tau, gate, drive, dt = 1., 2., .5, 3., .2
    numerator = h_old + dt*gate*drive
    denominator = 1+dt*(1/tau+gate)
    semi_implicit = numerator/denominator
    explicit = h_old+dt*(-(1/tau+gate)*h_old+gate*drive)
    print('numerator / denominator:', numerator, denominator)
    print('semi-implicit / explicit:', semi_implicit, explicit)
    ''')
    l.ex('q04', '从移项写出一步更新', '沿用上一格的 h、τ、f、A、Δt，从分子、分母计算半隐式结果。',
         '''
    ex_q04_h, ex_q04_tau, ex_q04_f = 1., 2., .5
    ex_q04_A, ex_q04_dt = 3., .2
    ex_q04_next = None  # TODO
    ''', '分子是 1+.2×.5×3；分母是 1+.2×(.5+.5)。',
         '''
    ex_q04_h, ex_q04_tau, ex_q04_f = 1., 2., .5
    ex_q04_A, ex_q04_dt = 3., .2
    ex_q04_next = (ex_q04_h+ex_q04_dt*ex_q04_f*ex_q04_A)/(1+ex_q04_dt*(1/ex_q04_tau+ex_q04_f))
    ''', "check('半隐式一步', ex_q04_next, 13/12)")
    l.md(r'''
    这一格的常数给出 $h_*=1.5$、$\tau_{\rm eff}=1$。冻结 f 后跨越时间 T，可以写出精确参考解：$h(T)=h_*+[h(0)-h_*]e^{-T/\tau_{\rm eff}}$。

    将同一段 T=1 分成更多半隐式子步，每个子步只走较短的一段，比较终点与精确解的距离。这个对照单独检验积分误差；后面的状态依赖单元会在每个子步重新计算门。
    ''')
    l.code('''
    exact_end = 1.5 + (1.-1.5)*np.exp(-1.)
    integration_errors = []
    for substeps in [1, 2, 4, 8, 16, 32]:
        value, small_dt = 1., 1./substeps
        for _ in range(substeps):
            value = (value+small_dt*.5*3)/(1+small_dt*(1/2+.5))
        integration_errors.append(abs(value-exact_end))
    print('errors, 1 to 32 substeps:', integration_errors)
    ''')
    l.md(r'''
    ## 5 给参数施加可微的正值约束

    优化器直接加减参数，而时间常数需要保持为正。让原始参数 r 在实数范围更新，再用 $\tau=\operatorname{softplus}(r)+0.05$ 得到实际时间常数，其中 $\operatorname{softplus}(r)=\log(1+e^r)$。损失对 r 的梯度会经过这个可微变换。

    门取 $f(u,h)=\operatorname{softplus}(w_u u+w_h h+b)$。基础 τ、输入权重、状态权重、偏置与驱动 A 都参加学习。同一组训练好的参数沿序列反复使用；每一步的 u、h 改变时，门 f 和有效 τ 也随之改变。
    ''')
    l.ex('q05', '原始参数改变，时间常数仍为正', '原始参数为 [−3,0,2]，用 PyTorch softplus 加 .05，返回三个时间常数。',
         '''
    ex_q05_raw = torch.tensor([-3.,0.,2.])
    ex_q05_tau = None  # TODO
    ex_q05_offset = .05
    ''', 'F.softplus(raw) 实现数值稳定的 log(1+exp(raw))。',
         '''
    ex_q05_raw = torch.tensor([-3.,0.,2.])
    ex_q05_offset = .05
    ex_q05_tau = F.softplus(ex_q05_raw)+ex_q05_offset
    ''', "check('正时间常数', ex_q05_tau, np.logaddexp(0.,[-3.,0.,2.])+.05)")
    l.md(r'''
    ### 一个子步的完整状态导数

    旧 h 有两条路影响新状态：一条直接进入分子，另一条先改变门 f，再同时改变分子与分母。

    记分母为 $D=1+\Delta t(1/\tau+f)$，分子为 $N=h+\Delta t fA$，新状态为 $h_+=N/D$。先将 h 固定、把 f 当中间变量，用商法则得到
    $$\frac{\partial h_+}{\partial f}=\frac{\Delta t A D-N\Delta t}{D^2}
    =\frac{\Delta t(A-N/D)}D
    =\frac{\Delta t(A-h_+)}D.$$
    再令 $z=w_u u+w_hh+b$。softplus 的导数是 $\sigma(z)$，因此门对旧状态的导数为 $f_h=w_h\sigma(z)$。将两条路径的贡献相加：
    $$\frac{d h_+}{d h}=\frac1D+\frac{\Delta t(A-h_+)}D f_h.$$
    第一项来自分子中的直接 h，第二项来自门的变化。第二项的符号取决于 $A-h_+$ 与状态权重 $w_h$，会增强或抵消第一项。下一格用自动微分对照这两项相加的结果。
    ''')
    l.code('''
    h_grad = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    u_grad, dt_grad, tau_grad, A_grad = .3, .2, 1.3, 1.4
    z_grad = 1.3*u_grad-.4*h_grad-.2
    f_grad = F.softplus(z_grad)
    denominator_grad = 1+dt_grad*(1/tau_grad+f_grad)
    h_plus = (h_grad+dt_grad*f_grad*A_grad)/denominator_grad
    analytic_grad = 1/denominator_grad + dt_grad*(A_grad-h_plus)/denominator_grad*(-.4*torch.sigmoid(z_grad))
    auto_grad = torch.autograd.grad(h_plus, h_grad)[0]
    check('包含门支路的状态导数', auto_grad, analytic_grad.detach(), atol=1e-10)
    ''')
    l.ex('q06', '冻结门与完整梯度分别是多少', '若 D=1.4、Δt=.2、A=1.5、新状态=.6、f_h=−.1，分别求直接通路和总导数。',
         '''
    ex_q06_direct = None  # TODO
    ex_q06_total = None  # TODO
    ex_q06_D = 1.4
    ''', '直接项 1/D；再加 Δt(A−h_new)f_h/D。',
         '''
    ex_q06_D = 1.4
    ex_q06_direct = 1/ex_q06_D
    ex_q06_total = ex_q06_direct+.2*(1.5-.6)/ex_q06_D*(-.1)
    ''', '''
    check('直接通路', ex_q06_direct, 1/1.4)
    check('完整导数', ex_q06_total, .982/1.4)
    ''')
    l.md('再让原始时间常数参数也参加梯度核对。`gradcheck` 对每个输入做微小正负扰动，用中心差分比较自动微分；这里同时检查状态、外部输入和原始 τ 参数。')
    l.code('''
    def parameterized_liquid_step(h, u, raw_tau):
        tau = F.softplus(raw_tau)+.05
        f = F.softplus(1.3*u-.4*h-.2)
        return (h+.2*f*1.4)/(1+.2*(1/tau+f))
    gradient_inputs = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                            for v in [.4, .3, .2])
    liquid_gradient_ok = torch.autograd.gradcheck(parameterized_liquid_step, gradient_inputs,
                                                 eps=1e-6, atol=1e-5)
    check('状态、输入和 tau 原始参数差分核对', liquid_gradient_ok, True)
    ''')
    l.md(r'''
    ## 6 构造不规则时间间隔的数据

    现在把一段更新接成整条序列。每条独立序列有 24 段，第 t 段输入 $u_t$ 在该段内保持常值，持续时间 $\Delta t_t$ 独立均匀取自 [.05,.4]。相同的输入幅度遇到不同的持续时间，就会留下不同的状态变化。

    用已知动力学生成参考状态：$\tau=1.3$、$A=1.4$、$f=\operatorname{softplus}(1.3u-0.4h-0.2)$。参考积分每段使用 12 个 RK4 小步，并记录段末状态。模型收到输入幅度和时间间隔，学习重现这些状态。
    ''')
    l.code('''
    rng = np.random.default_rng(171)
    liquid_inputs = rng.uniform(-1., 1., size=(224,24,1))
    liquid_dt = rng.uniform(.05, .4, size=(224,24,1))
    def reference_rhs(h, u):
        f = np.logaddexp(0., 1.3*u-.4*h-.2)
        return -(1/1.3+f)*h+f*1.4
    ''')
    l.md('生成参考轨迹时，RK4 先估计小步起点、两个中点和终点的斜率，再按 1、2、2、1 加权。每次估计都会用新的中间状态重新计算门。同一段内 u 保持不变，h 在 12 个小步中连续推进；段末值随后成为监督目标。')
    l.code('''
    reference_state, reference_trajectory = np.zeros((224,1)), []
    for t in range(24):
        u, step = liquid_inputs[:,t], liquid_dt[:,t]/12
        for _ in range(12):
            k1 = reference_rhs(reference_state, u)
            k2 = reference_rhs(reference_state+step*k1/2, u)
            k3 = reference_rhs(reference_state+step*k2/2, u)
            k4 = reference_rhs(reference_state+step*k3, u)
            reference_state = reference_state+step*(k1+2*k2+2*k3+k4)/6
        reference_trajectory.append(reference_state.copy())
    ''')
    l.code('''
    liquid_X = torch.tensor(liquid_inputs, dtype=torch.float32)
    liquid_steps = torch.tensor(liquid_dt, dtype=torch.float32)
    liquid_Y = torch.tensor(np.stack(reference_trajectory, axis=1), dtype=torch.float32)
    train_X, test_X = liquid_X[:168], liquid_X[168:]
    train_dt, test_dt = liquid_steps[:168], liquid_steps[168:]
    train_Y, test_Y = liquid_Y[:168], liquid_Y[168:]
    print('independent training/test sequences:', len(train_X), len(test_X))
    ''')
    l.md(r'''
    ## 7 把公式逐行写成可训练的单元

    x 为 `(B,1)`，h 为 `(B,H)`，dt 为 `(B,1)`；拼接 `[x,h]` 后用共享线性层算出 H 个非负门。基础 τ 与驱动 A 各有 H 个分量，按刚才的式子逐分量更新状态。

    每个采样间隔划为四个子步，每次用更新后的 h 重新计算 f。下面默认 H=1，使模型状态能与生成方程中的标量直接对照。类中的 `forward` 对应一段物理时间，外层序列循环再把 24 段接起来。
    ''')
    l.code('''
    class MinimalLTCCell(nn.Module):
        def __init__(self, hidden=1, substeps=4):
            super().__init__()
            self.hidden, self.substeps = hidden, substeps
            self.gate = nn.Linear(1+hidden, hidden)
            self.raw_tau = nn.Parameter(torch.zeros(hidden))
            self.drive = nn.Parameter(torch.ones(hidden))
        @property
        def tau(self):
            return F.softplus(self.raw_tau)+.05
        def forward(self, x, h, dt):
            small_dt = dt/self.substeps
            for _ in range(self.substeps):
                f = F.softplus(self.gate(torch.cat([x,h], dim=-1)))
                h = (h+small_dt*f*self.drive)/(1+small_dt*(1/self.tau+f))
            return h
    ''')
    l.code('''
    class LiquidSequence(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = MinimalLTCCell()
        def forward(self, X, dt):
            h = X.new_zeros(X.shape[0], self.cell.hidden)
            states = []
            for t in range(X.shape[1]):
                h = self.cell(X[:,t], h, dt[:,t])
                states.append(h)
            return torch.stack(states, dim=1)
    ''')
    l.md('每段的四个子步都保留在计算图中。训练损失比较 24 个段末状态，反向传播会穿过这些数值更新，回到门的权重、原始时间常数与驱动 A。梯度检验中拆开的直接通路和门通路，在这里都会参加参数学习。')
    l.code('''
    torch.manual_seed(172)
    liquid_model = LiquidSequence()
    liquid_optimizer = torch.optim.Adam(liquid_model.parameters(), lr=.025)
    liquid_losses = []
    for epoch in range(170):
        liquid_optimizer.zero_grad()
        liquid_loss = (liquid_model(train_X, train_dt)-train_Y).square().mean()
        liquid_loss.backward()
        liquid_optimizer.step()
        liquid_losses.append(liquid_loss.item())
    ''')
    l.code('''
    with torch.no_grad():
        liquid_prediction = liquid_model(test_X, test_dt)
        liquid_mse = (liquid_prediction-test_Y).square().mean().item()
    print('test MSE:', liquid_mse)
    print('learned base tau / drive:', liquid_model.cell.tau.detach(), liquid_model.cell.drive.detach())
    print('gate input/state weights:', liquid_model.cell.gate.weight.detach())
    physical_times = test_dt[0,:,0].cumsum(dim=0).numpy()
    ''')
    l.md('比较预测时，用真实 Δt 累加得到物理时间轴：时间点之间的横向距离保留了采样疏密。逐段读取门时，使用该段输入和起点的预测状态，算出该段起始的有效时间常数；下图将这个值标在对应段的终点，与输入段及状态点对齐。')
    l.code('''
    with torch.no_grad():
        previous_states = torch.cat([torch.zeros(1,1), liquid_prediction[0,:-1]], dim=0)
        observed_gate = F.softplus(liquid_model.cell.gate(torch.cat([test_X[0], previous_states], dim=1)))
        observed_tau = 1/(1/liquid_model.cell.tau+observed_gate)
    ''')
    l.code('''
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), layout='constrained')
    axes[0,0].semilogy(liquid_losses); axes[0,0].set(xlabel='Epoch', ylabel='Training MSE')
    axes[0,1].plot(physical_times, test_Y[0,:,0], 'o-', label='RK4 reference')
    axes[0,1].plot(physical_times, liquid_prediction[0,:,0], 'x--', label='Learned LTC')
    axes[0,1].set(xlabel='Physical time', ylabel='State'); axes[0,1].legend(frameon=False)
    input_plot_times = np.r_[0., physical_times]
    input_plot_values = torch.cat([test_X[0,:1,0], test_X[0,:,0]])
    axes[1,0].step(input_plot_times, input_plot_values, where='pre')
    axes[1,0].set(xlabel='Interval endpoint', ylabel='Input during interval')
    axes[1,1].plot(physical_times, observed_tau[:,0], 'o-')
    axes[1,1].axhline(liquid_model.cell.tau.item(), ls='--', color='#B66A47', label='Base tau')
    axes[1,1].set(xlabel='Interval endpoint', ylabel='Effective time constant')
    axes[1,1].legend(frameon=False)
    show(fig)
    ''')
    l.md(r'''
    右上图比较同一条测试序列的段末状态，左下图给出各段持续的输入。右下图的虚线是训练后固定的基础 τ，实线点是各段起始状态下的有效 τ。输入和旧状态一起改变门，所以基础参数固定后，响应尺度仍随序列变化。

    ## 8 接到论文中的三类连续时间模型

    | 模型 | 本课对应的计算对象 |
    |---|---|
    | [Neural ODE](https://arxiv.org/abs/1806.07366) | 用网络参数化 $\dot h=F_\theta(h,u,t)$，再积分 |
    | [LTC](https://arxiv.org/abs/2006.04439) | 用泄漏与非负状态依赖门共同决定瞬时衰减系数，数值求解 |
    | [CfC](https://arxiv.org/abs/2106.13898) | 从 LTC 动力学近似推导显含时间间隔的闭式更新结构 |

    本课每个状态使用一个非负门；作者 [ncps](https://github.com/mlech26l/ncps) 的 `ltc_cell.py` 按多条突触分别求电导与反转电位贡献，分子与分母都含跨突触求和。阅读实现时，先找到有效电容与时间步组成的 `cm_t`，再看泄漏 `gleak`、驱动项 `w_numerator` 与衰减项 `w_denominator`，将多条突触的求和接回本课的分子、分母。
    ''')
    l.md(r'''
    ## 9 将预测误差拆成可以检验的问题

    先冻结训练好的参数，将每段子步数从 4 改为 1、2、8、16。比较同一测试序列的 MSE 与计算时间，观察增加积分精度能改变多少预测误差。这个实验直接接着前面的固定门积分对照。

    然后保留真实采样时间，只把模型收到的 Δt 替换成训练集平均值。将误差按实际间隔长短分组，就能检查模型是否用到了时间间隔信息。

    门的状态依赖可以单独拿出来检验：保持参数量接近，把门中的状态系数固定为零，再重新训练多次。比较输入依赖与输入加状态依赖两种门的误差和有效 τ 轨迹。

    最后对照生成方程中的 τ=1.3、A=1.4，记录多次训练学到的参数。若预测接近而参数变化较大，可以继续改变观测噪声与时间范围，检查现有轨迹提供了多少参数辨识信息。接到实验记录时，可用相同划分比较物理松弛基线、RNN 与 LTC，再改变采样密度观察差异。
    ''')
    l.reading('Hasani 等：Liquid Time-Constant Networks，式 (1) 与融合求解步',
              'https://arxiv.org/html/2006.04439v4',
              '从 ODE 重新移项得到半隐式步，标出哪些状态项在旧时刻计算。论文时间常数界的证明对门函数用了什么有界性条件？本课 softplus 门满足哪些条件？在作者 ncps 代码中找到对应的多突触求和，解释简化单门与其参数化的关系。')
    return l


def build():
    return [rnn_lesson(), lstm_lesson(), gnn_lesson(), liquid_lesson()]
