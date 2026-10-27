"""Four incremental lessons: Python, NumPy, and a fully handwritten MLP."""
from coursekit import Lesson


def python_values():
    l = Lesson("01_python_values_control", "01 从变量到一张数字表", goals=[
        "读懂变量、列表、字典、条件与循环，并逐行解释代码",
        "独立处理一张小测量表，计算预测、误差和筛选结果",
        "把样本数据、模型参数和计算结果分开保存"], prereq="会新建并运行一个代码格")
    l.md(r'''
    ## 1 让电脑替我们算一次位置

    假设一个物体从 1 米处出发，每秒前进 2 米。到了第 3 秒，位置应当是 $1+2\times3=7$ 米。这一步心算很容易；如果表里有许多个时刻，就可以把相同的计算交给程序。

    第一段代码把时间、速度、初始位置分别保存，再按这条规则算位置。`time_s = 3` 中，等号把右边的数交给左边的名字。名字也说明这个数在计算里做什么：time_s 存时间，position 存结果。
    ''')
    l.code(r'''
    time_s = 3
    speed = 2.0
    initial_position = 1.0
    position = speed * time_s + initial_position
    print(position)
    print(type(time_s), type(speed))
    ''')
    l.md(r'''
    第一行输出是 `7.0`，与心算一致。后两项告诉我们保存了哪种数：`int` 是整数，`float` 是浮点数，能够表示带小数的量。`type` 查看类型，`print` 显示结果。

    `+ - * /` 分别做加、减、乘、除，`**` 做乘方，括号改变计算次序。Python 从上往下执行，所以算 position 时，前面三个名字都已经有值。把时间改为 4 后重新运行，这一组赋值和计算也会重新发生。
    ''')
    l.ex("py01", "把计算规则写出来", "时间改为 4 秒，其余相同。填写预测位置和位置的平方。", r'''
    ex_py01_time = 4
    ex_py01_position = None  # TODO
    ex_py01_square = None    # TODO
    ''', "先用 2 * 时间 + 1，再使用 ** 2。", r'''
    ex_py01_position = 2 * ex_py01_time + 1
    ex_py01_square = ex_py01_position ** 2
    ''', r'''
    check("位置", ex_py01_position, 9)
    check("位置的平方", ex_py01_square, 81)
    ''')
    l.md(r'''
    ## 2 同一个名字可以保存新的值

    表里多个数需要相加时，中间结果也要保存。`total = total + value` 的右边读取旧的 total，加上 value 后，将新结果存回同一个名字。`total += value` 是这一步的简写。
    ''')
    l.code(r'''
    total = 0
    total = total + 3
    total += 5
    print(total)
    equal_to_eight = total == 8
    print(equal_to_eight)
    ''')
    l.md(r'''
    total 从 0 变成 3，又变成 8。最后的 `total == 8` 得到 `True`，表示比较成立。`=` 保存值，`==` 比较值；比较的另一个结果是 `False`。`< <= > >=` 也产生这种布尔值，后面可以据此保留某些测量行。

    ## 3 字符串给数字加上可读的名字

    只有一串数时，还容易记住它们的含义。输出多次测量后，编号和单位也需要一起显示。引号包住的 `"run-A"` 是字符串；`f"...{变量}..."` 把变量的值插进文字。`:.2f` 指定显示两位小数，变量里仍保存原来的数值。
    ''')
    l.code(r'''
    run_name = "run-A"
    measured = 7.12
    message = f"{run_name}: position = {measured:.2f} m"
    print(message)
    print("原始数值：", measured)
    ''')
    l.md(r'''
    ## 4 列表把一串数放在一起

    现在把四次位置读数放在一起。时刻是 0、1、2、3 秒，读数是 1.1、3.0、5.2、6.9 米；它们接近刚才的直线，却没有完全落在上面。

    方括号建立列表，位置编号从 0 开始。`values[0]` 取第一个数，`values[-1]` 取最后一个数；`len(values)` 数元素个数。切片 `values[1:3]` 包含编号 1、2，遇到右端编号 3 就停止。
    ''')
    l.code(r'''
    times = [0.0, 1.0, 2.0, 3.0]
    positions = [1.1, 3.0, 5.2, 6.9]
    print(times[0], positions[-1])
    print(positions[1:3])
    print("测量次数：", len(times))
    ''')
    l.ex("py02", "按编号取出测量值", "取出第二次测量、最后两次测量，以及测量次数。", r'''
    ex_py02_values = [1.1, 3.0, 5.2, 6.9]
    ex_py02_second = None  # TODO
    ex_py02_tail = None    # TODO
    ex_py02_count = None   # TODO
    ''', "第二次测量的编号是 1；最后两个数用 [-2:]；个数用 len。", r'''
    ex_py02_second = ex_py02_values[1]
    ex_py02_tail = ex_py02_values[-2:]
    ex_py02_count = len(ex_py02_values)
    ''', r'''
    check("第二次测量", ex_py02_second, 3.0)
    check("最后两次", ex_py02_tail, [5.2, 6.9])
    check("次数", ex_py02_count, 4)
    ''')
    l.md(r'''
    ## 5 循环把同一个动作交给每个元素

    四次测量的平均位置可以用加法和除法求出。逐个手写加法仍然是在重复同一个动作：拿到一个读数，把它加入总和。

    `for value in positions:` 将这个动作写成循环。value 依次接到四个读数，冒号后缩进的行跟着执行四次。缩进退回后，计算总和除以次数，此时四个数都已加入。
    ''')
    l.code(r'''
    position_sum = 0.0
    for value in positions:
        position_sum += value
        print("加入", value, "以后：", position_sum)
    position_mean = position_sum / len(positions)
    print("平均位置：", position_mean)
    ''')
    l.md(r'''
    累加输出从 1.1、4.1、9.3 走到约 16.2，平均位置为 4.05。末尾多出的 `000000000000003` 来自浮点数的近似存储：像 1.1 这样的十进制小数，转成二进制时往往需要舍入。这几位尾数也会出现在后面的残差中。
    ''')
    l.ex("py03", "用循环算平方和", "列表中每个数先平方，再累加。不要直接写出最后答案。", r'''
    ex_py03_values = [1., 2., 3.]
    ex_py03_total = 0.0
    for ex_py03_value in ex_py03_values:
        # TODO：累加平方
        pass
    ''', "循环体使用 total += value ** 2。", r'''
    ex_py03_total = 0.0
    for ex_py03_value in ex_py03_values:
        ex_py03_total += ex_py03_value ** 2
    ''', r'''
    check("平方和", ex_py03_total, 14.)
    ''')
    l.md(r'''
    ## 6 按编号处理两列数据

    平均位置概括了这组读数落在哪里。要比较读数与运动规则，还需要给每个时刻算一次预测。相同编号的时间与位置属于同一次测量，因此用同一个 i 查询两列。

    把运动规则记成 $p(t)=wt+b$：$t$ 是时间，$p$ 是预测位置，$w$ 是速度参数，$b$ 是初始位置参数。这一轮取 $w=2$、$b=1$。

    `range(n)` 提供 0 到 n−1 的编号，`append` 把算出的一个结果加到列表末尾。循环结束时，就有了与观测逐行对应的一整列预测。
    ''')
    l.code(r'''
    weight = 2.0
    bias = 1.0
    predictions = []
    for i in range(len(times)):
        prediction = weight * times[i] + bias
        predictions.append(prediction)
        print(i, times[i], positions[i], prediction)
    print("整列预测：", predictions)
    ''')
    l.ex("py04", "生成一整列预测", "对 0、2、4 秒分别预测位置，速度为 3，初始位置为 −1。", r'''
    ex_py04_times = [0., 2., 4.]
    ex_py04_predictions = []
    for ex_py04_t in ex_py04_times:
        # TODO：算一次预测，再 append
        pass
    ''', "先算 3 * t - 1，再用 predictions.append(...)。", r'''
    ex_py04_predictions = []
    for ex_py04_t in ex_py04_times:
        ex_py04_predictions.append(3 * ex_py04_t - 1)
    ''', r'''
    check("三次预测", ex_py04_predictions, [-1., 5., 11.])
    ''')
    l.md(r'''
    ## 7 条件决定哪些结果保留

    预测列是 `[1,3,5,7]`。与读数相减，第 2 行偏高 0.2 米，第 3 行偏低 0.1 米；正负号保存了偏差方向。

    记残差 $r_i=y_i-p_i$，$y_i$ 是观测，$p_i$ 是预测，i 编号测量。现在挑出绝对偏差超过 0.15 米的行。`abs(r_i)` 去掉方向，`if 条件:` 在条件成立时执行缩进内容，`else:` 处理其余行。
    ''')
    l.code(r'''
    residuals = []
    large_error_ids = []
    for i in range(len(times)):
        residual = positions[i] - predictions[i]
        residuals.append(residual)
        if abs(residual) > 0.15:
            large_error_ids.append(i)
        else:
            print("阈值内的测量编号：", i)
    print("残差：", residuals, "超过阈值：", large_error_ids)
    ''')
    l.ex("py05", "筛出较大的误差", "保留绝对值大于 0.1 的误差。正负误差都应检查。", r'''
    ex_py05_errors = [0.2, 0.05, -0.3, 0.0]
    ex_py05_selected = []
    for ex_py05_error in ex_py05_errors:
        # TODO：判断，再 append
        pass
    ''', "先 if abs(error) > 0.1，再在下一层缩进里 append 原始误差。", r'''
    ex_py05_selected = []
    for ex_py05_error in ex_py05_errors:
        if abs(ex_py05_error) > 0.1:
            ex_py05_selected.append(ex_py05_error)
    ''', r'''
    check("筛出的误差", ex_py05_selected, [0.2, -0.3])
    ''')
    l.md(r'''
    ## 8 字典让一行数据的各项有名字

    筛选结果只留下编号 2，因为它的残差约为 0.2。到这里，每行已经有时间、读数和编号；如果再加测量批次、温度等信息，只靠“第几列”记含义会越来越费力。

    字典给各项加上名字：花括号里，冒号左边是键，右边是值。`row["time"]` 直接取得时间。把多行字典放进列表，就形成了有列名的一张表。
    ''')
    l.code(r'''
    rows = [{"name": "A", "time": 0., "position": 1.1},
            {"name": "B", "time": 1., "position": 3.0},
            {"name": "C", "time": 2., "position": 5.2},
            {"name": "D", "time": 3., "position": 6.9}]
    print(rows[2])
    for row in rows:
        print(row["name"], row["time"], row["position"])
    ''')
    l.ex("py06", "从字典读取输入并计算", "读取一行的 time，用参数 weight 和 bias 预测位置，再计算观测减预测。", r'''
    ex_py06_row = {"time": 2., "position": 5.2}
    ex_py06_parameters = {"weight": 2., "bias": 1.}
    ex_py06_prediction = None  # TODO
    ex_py06_residual = None    # TODO
    ''', "两个字典分别取参数与输入，预测后再与 row['position'] 相减。", r'''
    ex_py06_prediction = ex_py06_parameters["weight"] * ex_py06_row["time"] + ex_py06_parameters["bias"]
    ex_py06_residual = ex_py06_row["position"] - ex_py06_prediction
    ''', r'''
    check("预测", ex_py06_prediction, 5.)
    check("残差", ex_py06_residual, 0.2)
    ''')
    l.md(r'''
    ## 9 用一个数概括整张表的误差

    阈值筛选给出需要查看的行。若要比较两组速度参数，则希望每次计算得到一个整体误差。直接把残差相加会让正负偏差抵消，平方后，每行都贡献一个非负数。

    平方误差和记为 $E=\sum_{i=0}^{n-1}r_i^2$，n 是测量次数；平均平方误差为 $M=E/n$。这组残差约为 `[0.1,0,0.2,-0.1]`，所以 $E=0.01+0+0.04+0.01=0.06$，$M=0.015$。循环将这笔手算对整张表执行一次。
    ''')
    l.code(r'''
    squared_error_sum = 0.0
    for row in rows:
        predicted = weight * row["time"] + bias
        residual = row["position"] - predicted
        squared_error_sum += residual ** 2
    mse = squared_error_sum / len(rows)
    print("平方误差和：", squared_error_sum, "平均平方误差：", mse)
    ''')
    l.ex("py07", "独立计算平均平方误差", "已有三个残差，先累加平方，再除以测量次数。", r'''
    ex_py07_residuals = [0.1, -0.2, 0.3]
    ex_py07_sum = 0.0
    for ex_py07_r in ex_py07_residuals:
        # TODO
        pass
    ex_py07_mse = None  # TODO
    ''', "循环内 sum += r ** 2；循环外 mse = sum / len(residuals)。", r'''
    ex_py07_sum = 0.0
    for ex_py07_r in ex_py07_residuals:
        ex_py07_sum += ex_py07_r ** 2
    ex_py07_mse = ex_py07_sum / len(ex_py07_residuals)
    ''', r'''
    check("平方误差和", ex_py07_sum, 0.14)
    check("平均平方误差", ex_py07_mse, 0.14 / 3)
    ''')
    l.md(r'''
    ## 10 把数字与图对起来

    一个整体误差隐藏了偏差发生的位置。把四个读数和预测画在同一坐标轴上，第 2 行的偏高就会直接出现在图里。

    绘图库 Matplotlib 中，`plt.subplots` 建立图与坐标轴，`ax.scatter` 画观测点，`ax.plot` 按顺序连接预测。横轴是时间，纵轴是位置，同一时刻的竖直差正是刚算过的残差。
    ''')
    l.code(r'''
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(times, positions, label="Measurements")
    ax.plot(times, predictions, label="Prediction: 2t + 1")
    ax.set(xlabel="Time (s)", ylabel="Position (m)")
    ax.legend()
    show(fig)
    ''')
    l.md(r'''
    ## 把另一份读数交给程序

    在新代码格里处理时间 `[0,1,2,3]`、位置 `[0.9,3.2,4.8,7.1]`。这一回把逐行预测、残差、整体误差与筛选接成一段程序。速度为 2、初始位置为 1 时，平均平方误差应为 0.025，绝对残差超过 0.15 的行编号为 1、2。

    代码跑通后，将速度依次换成 1.8、2.0、2.2，保存三组误差。三次使用同一份读数，结果差别就可以追溯到速度参数。时刻与位置是给定的样本，速度与初始位置组成预测规则，误差由两者共同算出。

    这已经是一次小型模型比较。下一课会把重复的预测和误差计算变成函数，让另一张表也能直接使用。

    ## 带着问题读原文

    [Python 官方教程：基础语法与控制流](https://docs.python.org/3/tutorial/) 中的列表、for 和 if，已经在这张表的处理过程中出现过。读例子时，可以把其中的数换成自己的读数，观察一次缩进变化会让哪些行重复执行。
    ''')
    return l


def python_functions():
    l = Lesson("02_python_functions_files_classes", "02 函数、文件与最小模型类", goals=[
        "把重复计算写成接收参数并返回结果的函数",
        "用 Path 与 CSV 保存、读取一张小表",
        "理解实例、self、__init__ 与 super，为 nn.Module 做准备"], prereq="第 01 课：列表、字典、条件与循环")
    l.md(r'''
    ## 1 同一条预测规则还要用很多次

    上一课换了几次速度参数，每次都要算一整列预测，再和读数比较。换一张表以后，这些动作还会重复。可以把其中最小的一步“给出时间和参数，返回预测位置”写成一个函数。

    这条规则仍是 $p(x)=wx+b$，x 是时间，w、b 是速度和初始位置。`def predict(...)` 给计算起名字，括号里列出输入参数；`return` 把结果交回调用处。调用 `predict(3)` 时，3 交给形参 x，weight、bias 则取定义里给出的默认值。它会返回刚才算过的 7。
    ''')
    l.code(r'''
    def predict(x, weight=2.0, bias=1.0):
        result = weight * x + bias
        return result

    print(predict(3.))
    print(predict(3., weight=1.5, bias=0.))
    saved_prediction = predict(2.)
    print(saved_prediction)
    ''')
    l.ex("fn01", "写一个能返回误差的函数", "定义 residual(observed, predicted)，返回观测减预测。填写函数体，并计算给定例子。", r'''
    def ex_fn01_residual(observed, predicted):
        return None  # TODO
    ex_fn01_result = ex_fn01_residual(5.2, 5.0)
    ''', "return observed - predicted。函数把数交回来，调用结果才可以接着计算。", r'''
    def ex_fn01_residual(observed, predicted):
        return observed - predicted
    ex_fn01_result = ex_fn01_residual(5.2, 5.0)
    ''', r'''
    check("函数返回的残差", ex_fn01_result, 0.2)
    ''')
    l.md(r'''
    ## 2 返回值与局部变量

    三次调用得到 7、4.5、5，说明同一段函数体已经处理了不同的时间和参数。返回值还能继续平方、相加或保存。

    函数里的 result 在这次调用中保存中间结果，外面也可以有一个同名变量。下一格故意把外部 result 设为 100：调用函数后，它仍为 100，返回的 9 被另一个名字接住。这就是局部作用域在这里的具体表现。
    ''')
    l.code(r'''
    result = 100.
    returned_value = predict(4.)
    print("函数返回：", returned_value)
    print("外部 result：", result)
    combined = returned_value ** 2
    print("返回值可继续参加计算：", combined)
    ''')
    l.md(r'''
    算误差时，平方和与平均平方误差经常都要保留。一个函数可以把这两个数一起返回：`return total, mean` 将它们组成元组，调用处的两个名字按相同顺序接收。

    输入 `[0.1,-0.2,0.3]` 时，平方和是 0.14，除以当前三项就得到约 0.04667。函数遍历传入的残差列表，因此换成两次或十次测量，计算规则仍然适用。
    ''')
    l.code(r'''
    def error_summary(residuals):
        total = 0.0
        for residual in residuals:
            total += residual ** 2
        return total, total / len(residuals)

    total_value, mean_value = error_summary([0.1, -0.2, 0.3])
    print(total_value, mean_value)
    ''')
    l.ex("fn02", "函数处理任意长度的列表", "写出均值函数；分别处理两个数和四个数，检查除数确实来自输入长度。", r'''
    def ex_fn02_mean(values):
        # TODO：循环累加，再返回均值
        return None
    ex_fn02_a = ex_fn02_mean([2., 4.])
    ex_fn02_b = ex_fn02_mean([1., 3., 5., 7.])
    ''', "total 从 0 开始，每次加一个 value，最后 return total / len(values)。", r'''
    def ex_fn02_mean(values):
        total = 0.0
        for value in values:
            total += value
        return total / len(values)
    ex_fn02_a = ex_fn02_mean([2., 4.])
    ex_fn02_b = ex_fn02_mean([1., 3., 5., 7.])
    ''', r'''
    check("两个数的均值", ex_fn02_a, 3.)
    check("四个数的均值", ex_fn02_b, 4.)
    ''')
    l.md(r'''
    ## 3 列表推导是一个短循环

    生成预测列的循环只有一个动作：调用 predict，并收集结果。这样短的循环常写成列表推导式 `[表达式 for 元素 in 列表]`。它仍按顺序取每个元素，算完表达式后组成新列表。把两种写法放在一起，可以看到 `[1,3,5,7]` 的四项从哪里来。
    ''')
    l.code(r'''
    times = [0., 1., 2., 3.]
    predictions_loop = []
    for t in times:
        predictions_loop.append(predict(t))
    predictions_short = [predict(t) for t in times]
    print(predictions_loop, predictions_short)
    ''')
    l.ex("fn03", "把循环压成列表推导", "给定观测与预测，用相同编号配对并生成残差列表。", r'''
    ex_fn03_y = [1.1, 3.0, 5.2]
    ex_fn03_p = [1.0, 3.0, 5.0]
    ex_fn03_residuals = None  # TODO：列表推导
    ''', "表达式是 y[i] - p[i]，编号来自 range(len(y))。", r'''
    ex_fn03_residuals = [ex_fn03_y[ex_fn03_i] - ex_fn03_p[ex_fn03_i]
                         for ex_fn03_i in range(len(ex_fn03_y))]
    ''', r'''
    check("残差列表", ex_fn03_residuals, [0.1, 0.0, 0.2])
    ''')
    l.md(r'''
    ## 4 用字典返回有名字的结果

    预测函数与误差函数已经能分别调用，现在把它们接成一次评估。`zip(times, observations)` 每次给出同一位置的时间和读数，因此循环里能直接写“观测减预测”。

    这次返回三项：残差列表、平方和、平均平方误差。把它们装进字典后，`summary["mse"]` 就能按含义取值。演示只传入前三次观测，所以返回的平方和是 0.05，平均平方误差是 $0.05/3$；第四次读数尚未包含在这次调用里。
    ''')
    l.code(r'''
    def evaluate(times, observations, weight=2., bias=1.):
        residuals = []
        for t, observed in zip(times, observations):
            residuals.append(observed - predict(t, weight, bias))
        total, mean = error_summary(residuals)
        return {"residuals": residuals, "sse": total, "mse": mean}

    summary = evaluate([0., 1., 2.], [1.1, 3., 5.2])
    print(summary, summary["mse"])
    ''')
    l.md(r'''
    ## 5 Path 表示文件所在的位置

    到目前为止，读数都写在代码里。保存成文件后，下一次评估就可以从文件读入，也方便换成另一份测量。

    标准库 `pathlib` 的 Path 表示位置，`csv` 负责表格文本。`ROOT / "data" / "generated"` 将几段目录名拼成路径，`.mkdir(parents=True, exist_ok=True)` 建立缺少的目录。`.name` 取文件名，`.parent` 取父目录。

    这里使用课程中的 `data/generated/02_measurements_demo.csv` 保存练习表；重新运行写入格时，会用同样四行数据重建它。
    ''')
    l.code(r'''
    from pathlib import Path
    import csv
    generated_dir = ROOT / "data" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = generated_dir / "02_measurements_demo.csv"
    print(csv_path.name, csv_path.parent)
    ''')
    l.md(r'''
    打开文件并写入后，屏幕上应出现列名 `time,position` 和四行逗号分隔的数。每个字典成为文件中的一行。

    `with 文件.open(...) as f:` 在缩进中使用文件，结束后关闭它。`"w"` 表示写入，`encoding="utf-8"` 指定文本编码，`newline=""` 将行结束符交给 CSV 模块处理。`writeheader` 写列名，`writerows` 写数据行。
    ''')
    l.code(r'''
    table_rows = [{"time": 0., "position": 1.1}, {"time": 1., "position": 3.0},
                  {"time": 2., "position": 5.2}, {"time": 3., "position": 6.9}]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "position"])
        writer.writeheader()
        writer.writerows(table_rows)
    print(csv_path.read_text(encoding="utf-8"))
    ''')
    l.ex("fn04", "构造文件路径", "在已给出的目录对象下拼出 trace.csv，再取出文件名是否符合预期的布尔值。只构造路径，不需要创建文件。", r'''
    ex_fn04_directory = ROOT / "data" / "generated"
    ex_fn04_path = None  # TODO
    ex_fn04_name_ok = None  # TODO
    ''', "路径拼接使用 /；Path.name 读取文件名。用 == 得到用于检查的布尔值。", r'''
    ex_fn04_path = ex_fn04_directory / "trace.csv"
    ex_fn04_name_ok = ex_fn04_path.name == "trace.csv"
    ''', r'''
    check("文件名", ex_fn04_name_ok, True)
    ''')
    l.md(r'''
    ## 6 读回表格，注意字符串与数值

    文件里看到的 `2.0` 是一段文本。`DictReader` 用列名建立字典，刚读出的各项仍是字符串；用于乘法、求误差时，需要 `float(...)` 转回数值。

    下一格一行行完成转换。读回后打印出的四个字典，应与写入前对应，最后的类型检查显示时间已经是 float。文件的保存与读取在这里接成了一个来回。
    ''')
    l.code(r'''
    loaded_rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loaded_rows.append({"time": float(row["time"]),
                                "position": float(row["position"])})
    print(loaded_rows)
    print(type(loaded_rows[0]["time"]))
    ''')
    l.ex("fn05", "将一行文本转成数值", "模拟读出一行 CSV，将 time 和 position 转成浮点数，再做预测与误差计算。", r'''
    ex_fn05_row = {"time": "2.0", "position": "5.2"}
    ex_fn05_t = None  # TODO
    ex_fn05_y = None  # TODO
    ex_fn05_error = None  # TODO：用 2t+1 预测
    ''', "float(row['time'])；误差为 y - (2*t + 1)。", r'''
    ex_fn05_t = float(ex_fn05_row["time"])
    ex_fn05_y = float(ex_fn05_row["position"])
    ex_fn05_error = ex_fn05_y - (2 * ex_fn05_t + 1)
    ''', r'''
    check("时间数值", ex_fn05_t, 2.)
    check("位置数值", ex_fn05_y, 5.2)
    check("误差", ex_fn05_error, 0.2)
    ''')
    l.md(r'''
    ## 7 类把参数与计算规则放在一起

    评估两组参数时，函数每次都要接收 weight、bias。如果希望把某一组参数保留下来，随时对不同输入做预测，可以让一个对象保存它们，并提供自己的计算方法。

    `class LinearModel:` 定义这种对象的结构，`LinearModel(2, 1)` 创建其中一个实例。创建时执行 `__init__`，把 2、1 存成这个实例的 weight、bias。`self` 指当前实例，所以 `self.weight` 读到的是它自己的参数。

    `model.forward(3)` 调用这个实例的方法，Python 会自动把 model 交给方法的第一个 self 参数，3 则交给 x。算出的仍然是 $2\times3+1=7$。
    ''')
    l.code(r'''
    class LinearModel:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias

        def forward(self, x):
            return self.weight * x + self.bias

    model = LinearModel(2., 1.)
    print(model.forward(3.), model.weight)
    ''')
    l.md(r'''
    类只定义一次，却可以创建多组参数彼此独立的模型。下一格将 A 的速度改成 1.5，A 在 2 秒处给出 4；B 仍保存速度 3、初始位置 −1，给出 5。各实例保存的数值开始真正影响它们的行为。

    后续 PyTorch 网络也把前向计算写在 `forward` 中。这里已经有了同样的关系：实例保存参数，方法使用参数处理传入的样本。
    ''')
    l.code(r'''
    model_a = LinearModel(2., 1.)
    model_b = LinearModel(3., -1.)
    model_a.weight = 1.5
    print("A 的预测：", model_a.forward(2.))
    print("B 的预测：", model_b.forward(2.))
    ''')
    l.ex("fn06", "自己写 forward 方法", "类已经接收两个参数。补好 forward，使用当前实例保存的值。", r'''
    class ExFn06Linear:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias
        def forward(self, x):
            return None  # TODO
    ex_fn06_model = ExFn06Linear(3., -2.)
    ex_fn06_output = ex_fn06_model.forward(4.)
    ''', "方法内部通过 self.weight、self.bias 读取参数。", r'''
    class ExFn06Linear:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias
        def forward(self, x):
            return self.weight * x + self.bias
    ex_fn06_model = ExFn06Linear(3., -2.)
    ex_fn06_output = ex_fn06_model.forward(4.)
    ''', r'''
    check("实例的前向结果", ex_fn06_output, 10.)
    ''')
    l.md(r'''
    ## 8 继承与 super 先做一件小事

    模型多起来后，有些状态可以共用一个写法。这里让父类 ModelBase 负责保存名字，子类 NamedLinear 保留刚才的直线计算。

    `class NamedLinear(ModelBase)` 声明继承关系，`super().__init__("linear")` 在当前实例上执行父类的初始化；随后两行保存子类的 weight、bias。因此同一个 named_model 既能读出名字，也能预测位置。稍后继承 `nn.Module` 时，父类的初始化还会建立管理参数、子模块的结构。
    ''')
    l.code(r'''
    class ModelBase:
        def __init__(self, name):
            self.name = name

    class NamedLinear(ModelBase):
        def __init__(self, weight, bias):
            super().__init__("linear")
            self.weight, self.bias = weight, bias
        def forward(self, x):
            return self.weight * x + self.bias
    ''')
    l.code(r'''
    named_model = NamedLinear(2., 1.)
    print("父类保存的名字：", named_model.name)
    print("子类计算的结果：", named_model.forward(3.))
    ''')
    l.md(r'''
    ## 9 用函数整理一次受控比较

    现在可以把文件里的四行读数直接交回评估函数。每个候选速度只改一次函数参数，预测、残差和均方误差便一起算出。

    在同样的 bias=1 下，三种速度 1.8、2.0、2.2 对应的均方误差约为 0.165、0.015、0.145。中间这个候选在这四次测量上最接近读数。它来自整张表的一次评估，而不是某一行恰好对上了。
    ''')
    l.code(r'''
    loaded_t = [row["time"] for row in loaded_rows]
    loaded_y = [row["position"] for row in loaded_rows]
    parameter_trials = []
    for trial_weight in [1.8, 2., 2.2]:
        trial = evaluate(loaded_t, loaded_y, weight=trial_weight, bias=1.)
        parameter_trials.append({"weight": trial_weight, "mse": trial["mse"]})
    print(parameter_trials)
    ''')
    l.md(r'''
    ## 做一个能接收新文件的评估器

    尝试将这一课的计算接成 `assess_file(path, weight, bias)`。传入一个带 time、position 两列的 CSV，它返回测量次数、预测、残差和均方误差。用自己写的三行新数据调用它，便能看出计算是否真正由传入的文件和参数决定。

    还可以把内部的 predict 换成 LinearModel 实例的 forward，逐项比较两种写法的预测。保存参数的方式变了，位置公式仍然相同。

    留一份原始结果，将三行数据中的一行增加 0.5，再调用评估器。写下改动前后的残差和均方误差，观察一次局部改动怎样进入整体评价。函数和文件在这里让实验能重做，也让每个结果有可追查的输入。

    ## 带着问题读原文

    [Python 官方教程](https://docs.python.org/3/tutorial/) 的函数、输入输出和类章节，可以与这个评估器对照着读。遇到 self 时，沿着一次 `model.forward(时间)` 调用，找出对象保存的参数与方法本次收到的输入。后面看到 `nn.Module`，这两类量仍然要分别辨认。
    ''')
    return l


def numpy_algebra():
    l = Lesson("03_numpy_shapes_linear_algebra", "03 从循环到 NumPy 与线性代数", goals=[
        "在计算前说清每个数组的轴与形状",
        "将逐坐标循环改写为广播、求和与矩阵乘法",
        "独立实现一批样本的线性变换并与循环核对"], prereq="第 02 课：函数、列表推导与对象")
    l.md(r'''
    ## 1 四个点放进一张表

    这一课处理四个点：$(0,0)$、$(0,2)$、$(4,0)$、$(4,2)$，它们组成一个矩形。要把矩形移到原点附近，或者为每个点计算几种加权输出，上一课的循环已经能完成。NumPy 让我们用数组表达同样的计算，省去逐个元素写循环的部分。

    每行保存一个点，每列保存一个坐标。将这张表记为 $X\in\mathbb R^{n\times d}$，n=4 是样本数，d=2 是坐标数。以后把一行换成图像像素或某次测量的多个特征，这种排列仍然能用。

    `np.array(..., dtype=float)` 将嵌套列表转成浮点数组，外层列表给行，内层给列。`shape` 是各轴长度，`ndim` 是轴数。本课用 np 作为 NumPy 的别名，用 Matplotlib 画出计算后的点。
    ''')
    l.code(r'''
    import numpy as np
    import matplotlib.pyplot as plt
    style()
    X = np.array([[0., 0.], [0., 2.], [4., 0.], [4., 2.]], dtype=float)
    print(X)
    print("shape:", X.shape, "ndim:", X.ndim, "dtype:", X.dtype)
    ''')
    l.md(r'''
    ## 2 行、列与切片

    打印的表有四行两列，对应 `(4,2)`。取编号 2 的点时，希望拿到它的两个坐标；取纵坐标时，则希望拿到四个点各自的一个数。

    `X[i,r]` 用 i 选行、r 选列，下标仍从 0 开始。`X[i]` 得到一维的 `(d,)`，`X[i:i+1]` 则保留成二维的一行 `(1,d)`。`X[:,r]` 取一列，返回一维数组 `(n,)`。这些结果的数值可能相同，轴的组织方式却会影响它们接下来怎样参与运算。
    ''')
    l.code(r'''
    print("一行：", X[2], X[2].shape)
    print("二维的一行：", X[2:3], X[2:3].shape)
    print("纵坐标列：", X[:, 1], X[:, 1].shape)
    print("指定元素：", X[3, 1])
    ''')
    l.ex("np01", "预先判断形状", "取出最后一行、前两行和全部横坐标。再填写前两行的形状。", r'''
    ex_np01_X = np.array([[0., 0.], [0., 2.], [4., 0.], [4., 2.]])
    ex_np01_last = None   # TODO
    ex_np01_first2 = None  # TODO
    ex_np01_x = None      # TODO
    ex_np01_shape = None  # TODO
    ''', "最后一行 [-1]；前两行 [:2]；横坐标 [:, 0]。形状通过取出结果的 .shape 获取。", r'''
    ex_np01_last = ex_np01_X[-1]
    ex_np01_first2 = ex_np01_X[:2]
    ex_np01_x = ex_np01_X[:, 0]
    ex_np01_shape = ex_np01_first2.shape
    ''', r'''
    check("最后一行", ex_np01_last, [4., 2.])
    check("前两行", ex_np01_first2, [[0., 0.], [0., 2.]])
    check("横坐标", ex_np01_x, [0., 0., 4., 4.])
    check("前两行形状", ex_np01_shape, (2, 2))
    ''')
    l.md(r'''
    ## 3 先用循环计算每列均值

    矩形的中心在 $(2,1)$。用程序求它时，每个坐标应分别平均：横坐标 $[0,0,4,4]$ 的均值为 2，纵坐标 $[0,2,0,2]$ 的均值为 1。

    循环固定一列 r，把这一列的四项加起来，再除以样本数 n。`np.zeros(d)` 建立 d 个 0，给两个坐标的结果留位置。这里每个下标都有明确工作：i 选择哪一个点，r 选择这个点的哪一个坐标。
    ''')
    l.code(r'''
    mean_loop = np.zeros(X.shape[1])
    for r in range(X.shape[1]):
        total = 0.
        for i in range(X.shape[0]):
            total += X[i, r]
        mean_loop[r] = total / X.shape[0]
    print(mean_loop)
    ''')
    l.md(r'''
    刚才的循环最后保留了两个坐标，四个样本被平均掉了。NumPy 用 `mean(axis=0)` 表达这件事：沿样本轴做均值，留下每列一个结果。相应地，`sum(axis=1)` 把每个点的坐标相加，留下四个结果。

    axis 指定被合并的轴。在这张表里，轴 0 是样本，轴 1 是坐标。`keepdims=True` 把被合并的轴保留为长度 1，于是均值可以排成 `(1,2)` 的一行。下一格既比较数值，也打印这种排列差别。
    ''')
    l.code(r'''
    mean_vector = X.mean(axis=0)
    mean_row = X.mean(axis=0, keepdims=True)
    row_sums = X.sum(axis=1)
    print(mean_vector, mean_vector.shape)
    print(mean_row, mean_row.shape)
    print(row_sums, row_sums.shape)
    check("循环与数组均值一致", mean_vector, mean_loop)
    ''')
    l.ex("np02", "沿正确的轴求和", "对一个 3×2 数组，计算每列的和、每行的和，以及保留二维形状的每行和。", r'''
    ex_np02_X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    ex_np02_columns = None  # TODO
    ex_np02_rows = None     # TODO
    ex_np02_column_shape = None  # TODO：(3, 1)
    ''', "每列的和用 axis=0；每行的和用 axis=1；保留成一列时加 keepdims=True。", r'''
    ex_np02_columns = ex_np02_X.sum(axis=0)
    ex_np02_rows = ex_np02_X.sum(axis=1)
    ex_np02_column_shape = ex_np02_X.sum(axis=1, keepdims=True)
    ''', r'''
    check("每列的和", ex_np02_columns, [9., 12.])
    check("每行的和", ex_np02_rows, [3., 7., 11.])
    check("保留列形状", ex_np02_column_shape, [[3.], [7.], [11.]])
    ''')
    l.md(r'''
    ## 4 广播前，先写逐元素相减

    找到中心以后，把每个点都减去 $(2,1)$，矩形就移到了原点周围。对第一个点，这一步给出 $(-2,-1)$；对最后一个点，给出 $(2,1)$。

    循环仍按样本 i、坐标 r 计算 `X[i,r] - mean[r]`。同一个均值坐标会在每一行重复使用，这正是接下来广播要表达的重复关系。
    ''')
    l.code(r'''
    centered_loop = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for r in range(X.shape[1]):
            centered_loop[i, r] = X[i, r] - mean_vector[r]
    print(centered_loop)
    ''')
    l.md(r'''
    `X - mean_vector` 将同一组两坐标与每行配对。形状 `(4,2)` 和 `(2,)` 从右边对齐，两个坐标分别相减，后者的同一组数被用于四行。这种按形状配对运算的方式叫广播。

    从右边检查各轴，对应长度相同，或其中一个长度为 1，就能配合。结果仍是四个二维点，中心化后的列均值变为 `[0,0]`。广播省掉了显式循环，计算出来的结果数组仍要占用内存。
    ''')
    l.code(r'''
    centered_vector = X - mean_vector
    print(centered_vector)
    print("中心化后的列均值：", centered_vector.mean(axis=0))
    check("相减的两种实现", centered_vector, centered_loop)
    ''')
    l.ex("np03", "给每一行加同一个向量", "每个样本的第一个坐标加 10，第二个坐标减 1。直接用广播完成。", r'''
    ex_np03_X = np.array([[0., 0.], [0., 2.], [4., 0.]])
    ex_np03_shift = np.array([10., -1.])
    ex_np03_shifted = None  # TODO
    ''', "两个输入形状为 (3,2)、(2,)，直接相加。", r'''
    ex_np03_shifted = ex_np03_X + ex_np03_shift
    ''', r'''
    check("平移后的三个点", ex_np03_shifted, [[10., -1.], [10., 1.], [14., -1.]])
    ''')
    l.md(r'''
    ## 5 同时为所有点与所有中心配对

    每行减同一个向量已经会了。现在给出两个参考点 $(0,0)$、$(4,2)$，希望知道每个样本到两者各有多远。一个样本需要两个结果，所以距离表应有四行两列。

    用 C 保存 K 个参考点，形状为 `(K,d)`。第 i 个样本到第 j 个参考点的平方距离记作 $D^2_{ij}=\sum_r(X_{ir}-C_{jr})^2$：坐标相减、平方、相加。循环遍历每一对 i、j，`np.sum` 负责最后的坐标求和。
    ''')
    l.code(r'''
    C = np.array([[0., 0.], [4., 2.]])
    D2_loop = np.zeros((len(X), len(C)))
    for i in range(len(X)):
        for j in range(len(C)):
            D2_loop[i, j] = np.sum((X[i] - C[j]) ** 2)
    print(D2_loop)
    ''')
    l.md(r'''
    这次重复发生在“样本与参考点的所有配对”上。给它们分别留一个可配对的轴：`X[:,None,:]` 成为 `(n,1,d)`，`C[None,:,:]` 成为 `(1,K,d)`。索引里的 None 插入长度为 1 的新轴，相减后便得到 `(n,K,d)`。

    三个轴依次是样本、参考点、坐标。例如 `diff[1]` 有两行，表示点 $(0,2)$ 到两个参考点的坐标差：`[0,2]` 与 `[-4,0]`。沿坐标轴 2 求平方和，得到 4、16；所有样本同样处理，就还原了循环算出的 `(n,K)` 距离表。
    ''')
    l.code(r'''
    X_expanded, C_expanded = X[:, None, :], C[None, :, :]
    diff = X_expanded - C_expanded
    D2 = (diff ** 2).sum(axis=2)
    print("加轴后：", X_expanded.shape, C_expanded.shape)
    print("差值：", diff.shape, "距离表：", D2.shape)
    print("点 1 对全部中心的坐标差：\n", diff[1])
    check("广播与循环距离一致", D2, D2_loop)
    ''')
    l.md(r'''
    图里每行对应一个样本，每列对应一个参考点。第 0 行的 0、20 表示第一个点就在参考点 0 上，到参考点 1 的平方距离为 20。沿着这些具体位置读图，三维的坐标差最终被压成了二维的配对结果。
    ''')
    l.code(r'''
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.imshow(D2, cmap="Blues", aspect="auto")
    for i in range(len(X)):
        for j in range(len(C)):
            ax.text(j, i, f"{D2[i,j]:.0f}", ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=range(len(X)), xlabel="Center j", ylabel="Sample i")
    show(fig)
    ''')
    l.ex("np04", "亲手完成三维到二维", "给定两个点、两个中心，使用广播计算平方距离表。不要把样本轴或中心轴加起来。", r'''
    ex_np04_X = np.array([[0., 0.], [2., 1.]])
    ex_np04_C = np.array([[0., 1.], [3., 1.]])
    ex_np04_diff = None  # TODO：(2,2,2)
    ex_np04_D2 = None    # TODO：(2,2)
    ''', "先 X[:,None,:] - C[None,:,:]，平方后 sum(axis=2)。", r'''
    ex_np04_diff = ex_np04_X[:, None, :] - ex_np04_C[None, :, :]
    ex_np04_D2 = (ex_np04_diff ** 2).sum(axis=2)
    ''', r'''
    check("配对的坐标差", ex_np04_diff, [[[0., -1.], [-3., -1.]], [[2., 0.], [-1., 0.]]])
    check("配对的平方距离", ex_np04_D2, [[1., 10.], [4., 1.]])
    ''')
    l.md(r'''
    ## 6 取最小值的编号，再按编号取值

    距离表有了，每个点可以选择更近的参考点。我们要的是列编号，才能知道选中了谁。`np.argmin(一行)` 返回最小值所在的列，`np.argmin(D2,axis=1)` 将这一步同时用于四行。

    结果 `[0,0,1,1]` 把矩形左边两点与右边两点分开。还可以按这些编号取出各自选中的距离：`np.arange(n)` 提供行编号，`D2[行编号数组,列编号数组]` 按相同位置配对索引，得到 `[0,4,4,0]`。
    ''')
    l.code(r'''
    nearest_loop = []
    for i in range(len(X)):
        nearest_loop.append(np.argmin(D2[i]))
    nearest = np.argmin(D2, axis=1)
    chosen_distances = D2[np.arange(len(X)), nearest]
    print(nearest, chosen_distances)
    check("逐行与整表分配一致", nearest, nearest_loop)
    ''')
    l.md(r'''
    `nearest==0` 给出 `[True,True,False,False]`。把这个条件放进 X 的方括号，就取得左边两点；这种布尔索引保留条件成立的行。整数数组索引则按给定编号取行，`X[[3,0]]` 会把最后一个点放在前面，第一个点放在后面。
    ''')
    l.code(r'''
    print("布尔条件：", nearest == 0)
    print("属于中心 0 的点：\n", X[nearest == 0])
    print("按编号重排两行：\n", X[[3, 0]])
    ''')
    l.ex("np05", "分配、筛选、取对应数值", "从给定距离表逐行选最近列，取每行选中的值，再保留分到中心 1 的样本行编号。", r'''
    ex_np05_D2 = np.array([[1., 8.], [9., 2.], [4., 3.]])
    ex_np05_nearest = None  # TODO
    ex_np05_chosen = None   # TODO
    ex_np05_ids = None      # TODO
    ''', "取编号用 argmin(axis=1)；取对应值用两列配对索引；行编号数组按 nearest==1 筛选。", r'''
    ex_np05_nearest = np.argmin(ex_np05_D2, axis=1)
    ex_np05_chosen = ex_np05_D2[np.arange(3), ex_np05_nearest]
    ex_np05_ids = np.arange(3)[ex_np05_nearest == 1]
    ''', r'''
    check("最近中心", ex_np05_nearest, [0, 1, 1])
    check("对应距离", ex_np05_chosen, [1., 2., 3.])
    check("中心 1 的样本行", ex_np05_ids, [1, 2])
    ''')
    l.md(r'''
    ## 7 内积：逐坐标相乘再相加

    距离将坐标差合成了一个数，另一种常见的合成方式是加权求和。例如输入 $(4,2)$，权重 $(2,-0.5)$，输出是 $4\times2+2\times(-0.5)=7$。第一项向上贡献 8，第二项向下贡献 1。

    两个 d 维向量 x、w 的内积定义为 $x^\top w=\sum_{r=0}^{d-1}x_rw_r$，上标 $\top$ 表示转置。两个一维数组的 `x @ w` 就得到这个标量；循环仍然是逐坐标相乘、累加。
    ''')
    l.code(r'''
    x = np.array([4., 2.])
    w = np.array([2., -0.5])
    dot_loop = 0.
    for r in range(len(x)):
        dot_loop += x[r] * w[r]
    print("循环：", dot_loop, "逐元素后求和：", np.sum(x*w), "内积：", x @ w)
    ''')
    l.ex("np06", "算一个神经元的加权和", "输入三个坐标、三个权重和一个偏置，计算 z = x @ w + b。先手算乘积再求和。", r'''
    ex_np06_x = np.array([1., 2., -1.])
    ex_np06_w = np.array([0.5, -0.2, 0.3])
    ex_np06_b = 0.1
    ex_np06_z = None  # TODO
    ''', "乘积分别是 0.5、−0.4、−0.3，再加偏置 0.1。", r'''
    ex_np06_z = ex_np06_x @ ex_np06_w + ex_np06_b
    ''', r'''
    check("激活前加权和", ex_np06_z, -0.1)
    ''')
    l.md(r'''
    ## 8 矩阵乘法：同时计算多个输出

    一个输入可以交给几组权重，产生几个不同的输出。将每组权重放在 W 的一列里，形状为 `(d,m)`，m 就是输出数。$W_{rj}$ 表示输入坐标 r 对输出 j 的权重，$Y_{ij}=\sum_r X_{ir}W_{rj}$。

    下一格选择的权重让输出 0 保留横坐标，输出 1 等于“半个横坐标加纵坐标”。四个点各算两项，每一项内部都做一个内积，因此循环中出现样本 i、输出 j、输入坐标 r 三个下标。
    ''')
    l.code(r'''
    W = np.array([[1., 0.5], [0., 1.]])
    Y_loop = np.zeros((len(X), W.shape[1]))
    for i in range(len(X)):
        for j in range(W.shape[1]):
            for r in range(X.shape[1]):
                Y_loop[i, j] += X[i, r] * W[r, j]
    print(Y_loop)
    ''')
    l.md(r'''
    矩阵乘法 `X @ W` 把这三个循环压在同一个运算里。形状是 `(n,d) @ (d,m) → (n,m)`，中间的 d 代表被相乘求和的输入坐标。`*` 做逐元素相乘，`@` 则按这个求和规则组合行与列。

    如果每个输出再加一个固定量，把它们放进 `(m,)` 的偏置 b，`X @ W + b` 就会给每行加同一组偏置。本例加 `[-1,0.2]` 后，最后一个点 $(4,2)$ 变成 $(3,4.2)$，可以直接核对图中的位置。
    ''')
    l.code(r'''
    Y = X @ W
    b = np.array([-1., 0.2])
    Y_with_bias = Y + b
    print("X、W、Y：", X.shape, W.shape, Y.shape)
    print("加上偏置：\n", Y_with_bias)
    check("矩阵乘法与三层循环一致", Y, Y_loop)
    ''')
    l.ex("np07", "独立做一批线性变换", "两行样本、三个输入坐标、两个输出。计算整批输出并填写输出形状。", r'''
    ex_np07_X = np.array([[1., 0., 2.], [0., 1., 1.]])
    ex_np07_W = np.array([[1., 0.], [0., 2.], [-1., 1.]])
    ex_np07_b = np.array([0.5, -0.5])
    ex_np07_Y = None      # TODO
    ex_np07_shape = None  # TODO
    ''', "(2,3) @ (3,2) 得到 (2,2)，再把 (2,) 偏置加给每一行。", r'''
    ex_np07_Y = ex_np07_X @ ex_np07_W + ex_np07_b
    ex_np07_shape = ex_np07_Y.shape
    ''', r'''
    check("线性变换输出", ex_np07_Y, [[-0.5, 1.5], [-0.5, 2.5]])
    check("输出形状", ex_np07_shape, (2, 2))
    ''')
    l.md(r'''
    ## 9 转置与形状决定可计算的连接

    刚才内积中的输入坐标沿着求和方向排好，矩阵乘法才表达了所需计算。需要交换输入与输出所占的轴时，可以对二维数组用 `.T` 转置：`(d,m)` 变成 `(m,d)`。下一课反向传播会用到这样的方向变化。

    `.reshape(...)` 按元素顺序重新排列形状，元素总数保持相同。它与转置各做自己的事。下面把一个 2×3 数组分别转成 3×2，观察同一个数最终落在哪一格。
    ''')
    l.code(r'''
    shape_demo = np.array([[1., 2., 3.], [4., 5., 6.]])
    print("原数组：\n", shape_demo)
    print("转置：\n", shape_demo.T)
    print("重新排成 3×2：\n", shape_demo.reshape(3, 2))
    ''')
    l.md(r'''
    转置的第一行是 `[1,4]`，它来自原来的第一列；reshape 的第一行是 `[1,2]`，它来自按行读出的前两个数。形状相同，元素关系已经不同。仅看输出尺寸，还不足以判断一段张量代码是否表达了正确计算。
    ''')
    l.md(r'''
    ## 10 保存副本，再修改当前状态

    下一课要比较参数更新前后的结果，需要留住一份旧值。这里有个容易直接看见的差别：NumPy 的基本切片常与原数组共享存储，改切片里的数，原来的表也会跟着改变。

    下一格把同一块区域分别作为切片和 `.copy()` 保存。将切片的第一项改成 99 后，原数组也出现 99，副本却仍保存 1。训练历史中的“旧参数”就需要这样的独立副本。
    ''')
    l.code(r'''
    copy_source = np.array([[1., 2.], [3., 4.], [5., 6.]])
    shared_slice = copy_source[:2]
    saved_copy = copy_source[:2].copy()
    shared_slice[0, 0] = 99.
    print("原数组：\n", copy_source)
    print("保存副本：\n", saved_copy)
    ''')
    l.md(r'''
    ## 11 把矩阵计算画成几何变换

    四个点已经经过 `X @ W + b`：横坐标整体减 1，新的纵坐标混入半个旧横坐标，再加 0.2。原来的矩形因此倾斜并平移。

    左右两图保留相同的样本编号，可以把每一行数组与它的几何变化联系起来。绘图里的 `enumerate(points)` 同时提供行编号和这一行的点。
    ''')
    l.code(r'''
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, points, title in zip(axes, [X, Y_with_bias], ["X", "X @ W + b"]):
        ax.scatter(points[:, 0], points[:, 1])
        for i, point in enumerate(points):
            ax.annotate(str(i), point, xytext=(5, 5), textcoords="offset points")
        ax.set(xlabel="Coordinate 0", ylabel="Coordinate 1", title=title)
        ax.set_aspect("equal", adjustable="datalim")
    show(fig)
    ''')
    l.md(r'''
    ## 换成三个输入坐标，计算规则还成立

    给出三行样本 `[[1,0,2],[0,1,1],[2,2,0]]`、权重 `[[1,0],[0,2],[-1,1]]`、偏置 `[0.5,-0.5]`。试着独立写出循环版和矩阵版，两份输出应为 `[[-0.5,1.5],[-0.5,2.5],[2.5,3.5]]`。从手算第一行开始，可以确定各列究竟怎样进入两个输出。

    用输出第 0 列大于 0 的条件筛选样本，再求这些原始样本的列均值。条件有三个布尔值，对应三行；W 将三个输入变成两个输出。这些形状是计算关系的记录。

    表变长以后，两份实现也可以用来做计时实验。`time.perf_counter()` 两次取值相减得到秒数。让样本行重复成较长的表，比较同一组输出的计算时间，并记录表有多大。广播距离里还出现过 `(n,K,d)` 的中间数组，可以按元素数估算它会占多少内存。

    ## 带着问题读原文

    [NumPy 广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html) 与[《动手学深度学习》线性代数](https://d2l.ai/chapter_preliminaries/linear-algebra.html) 都可以对照这四个点来读。书中的轴、内积、矩阵乘法，分别对应表里的哪一行计算？下一课会给加权输出接上非线性，并追踪误差怎样回到 W 的每一项。
    ''')
    return l


def derivatives_mlp():
    l = Lesson("04_derivatives_numpy_mlp", "04 从一个导数到手写神经网络训练", goals=[
        "用微小扰动解释导数，沿每条依赖路径计算梯度",
        "完整推导并核对 2–2–1 网络的所有参数梯度与输入梯度",
        "用手写 NumPy 前向、反向和梯度下降真正训练 XOR"], prereq="第 03 课：数组形状、广播、矩阵乘法；本课从差商引入导数")
    l.md(r'''
    ## 1 参数应该往哪边改

    第 02 课试过三个速度：1.8、2.0、2.2，对应误差约 0.165、0.015、0.145。候选只有三个，可以全部试算。神经网络会同时有许多参数，因而需要更细的线索：在当前取值附近，某个参数增大一点，会把误差推向哪边？

    用一条能手算的曲线练习这件事。设 $f(w)=w^3-2w$，给 w 增加小量 $\varepsilon$，函数值变化为 $f(w+\varepsilon)-f(w)$。用变化量除以走过的距离，$[f(w+\varepsilon)-f(w)]/\varepsilon$，得到这一小段的平均斜率。

    展开立方，分子变为 $(3w^2-2)\varepsilon+3w\varepsilon^2+\varepsilon^3$。除以 $\varepsilon$ 后留下 $3w^2-2+3w\varepsilon+\varepsilon^2$；当步长越来越小时，后两项趋于 0，当前点的导数便是 $f'(w)=3w^2-2$。下面用 NumPy 计算，用 Matplotlib 将这条曲线画出来。
    ''')
    l.code(r'''
    import numpy as np
    import matplotlib.pyplot as plt
    style()
    def scalar_function(w):
        return w ** 3 - 2 * w
    w0 = 0.7
    analytic_slope = 3 * w0 ** 2 - 2
    print("当前位置、函数值、导数：", w0, scalar_function(w0), analytic_slope)
    ''')
    l.md(r'''
    在 w=0.7 时，函数值为 −1.057，导数为 −0.53。前者说曲线有多高，后者说附近向右移动时下降得多快。

    还可以用左右两侧的函数值估斜率：$[f(w+\varepsilon)-f(w-\varepsilon)]/(2\varepsilon)$，这叫中心差分。对这条立方曲线展开后，得到 $3w^2-2+\varepsilon^2$，只剩一个平方步长误差。输出中步长从 0.1 缩到 0.0001，中心差分便从 −0.52 靠近 −0.52999999；每次都在估同一点的斜率。
    ''')
    l.code(r'''
    for epsilon in [0.1, 0.01, 0.001, 0.0001]:
        forward_difference = (scalar_function(w0+epsilon) - scalar_function(w0)) / epsilon
        central_difference = (scalar_function(w0+epsilon) - scalar_function(w0-epsilon)) / (2*epsilon)
        print(f"eps={epsilon:.4f}, forward={forward_difference:.8f}, central={central_difference:.8f}")
    ''')
    l.ex("dv01", "用扰动求一个斜率", "函数改为 f(w)=w²，在 w=2 处使用中心差分。自己填出左右两次函数值和差商。", r'''
    ex_dv01_w, ex_dv01_epsilon = 2., 1e-4
    ex_dv01_plus = None   # TODO
    ex_dv01_minus = None  # TODO
    ex_dv01_slope = None  # TODO
    ''', "两次函数值分别是 (w+epsilon)**2、(w-epsilon)**2，再相减除以 2*epsilon。", r'''
    ex_dv01_plus = (ex_dv01_w + ex_dv01_epsilon) ** 2
    ex_dv01_minus = (ex_dv01_w - ex_dv01_epsilon) ** 2
    ex_dv01_slope = (ex_dv01_plus - ex_dv01_minus) / (2 * ex_dv01_epsilon)
    ''', r'''
    check("右侧函数值", ex_dv01_plus, 2.0001**2)
    check("左侧函数值", ex_dv01_minus, 1.9999**2)
    check("导数近似", ex_dv01_slope, 4.)
    ''')
    l.md(r'''
    导数把“小步移动”换成“函数值预计改变多少”：$f(w_0+\Delta w)\approx f(w_0)+f'(w_0)\Delta w$。这里 $w_0$ 是当前位置，$\Delta w$ 是准备走的一小步。

    图中的虚线画出这条一阶近似，它在当前点与曲线相接，往远处走就逐渐分开。后面的梯度下降正是反复利用当前位置附近的变化趋势。`np.linspace(a,b,n)` 生成从 a 到 b 的 n 个等距数，供曲线逐点计算。
    ''')
    l.code(r'''
    w_grid = np.linspace(-0.1, 1.4, 151)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(w_grid, scalar_function(w_grid), label="f(w)")
    ax.plot(w_grid, scalar_function(w0)+analytic_slope*(w_grid-w0), "--", label="First-order approximation")
    ax.scatter([w0], [scalar_function(w0)], zorder=3)
    ax.set(xlabel="w", ylabel="Function value")
    ax.legend()
    show(fig)
    ''')
    l.md(r'''
    ## 2 一条路径相乘，多条路径相加

    一次完整预测由多步组成，参数改变后，影响也沿着这些中间量传下去。用 $u=2w$、$q=u^2$ 看最短的一条路径：w 改变一点，u 改变它的 2 倍；u 的这点变化，到 q 处又乘上当前位置的斜率 $2u$。

    $$\frac{dq}{dw}=\frac{dq}{du}\frac{du}{dw}=(2u)\times2.$$

    路径上的两个比例相乘，就是链式法则。现在增加支路 $v=3w$，令输出 $f=q+v$。同一次 w 的变化会同时进入两条支路，最后的输出包含两项的变化，因此

    $$\frac{df}{dw}=\underbrace{(2u)\times2}_{\text{经过 }u\text{ 的路径}}+
                       \underbrace{1\times3}_{\text{经过 }v\text{ 的路径}}.$$

    下一格 w=0.5，经过 u 的路径贡献 4，经过 v 的路径贡献 3，总导数为 7。漏掉一条就会得到错误的变化率。神经网络反向传播也遵循同一规则：沿路径乘局部导数，多条贡献回到同一个变量时相加。
    ''')
    l.code(r'''
    branch_w = 0.5
    branch_u, branch_v = 2 * branch_w, 3 * branch_w
    through_u = (2 * branch_u) * 2
    through_v = 1 * 3
    print("前向输出：", branch_u ** 2 + branch_v)
    print("两条路径：", through_u, through_v, "总导数：", through_u + through_v)
    ''')
    l.ex("dv02", "将两条依赖路径收齐", "还是 f=(2w)²+3w，但取 w=1。填写两条路径的贡献和总导数。", r'''
    ex_dv02_w = 1.
    ex_dv02_u = 2 * ex_dv02_w
    ex_dv02_path_a = None  # TODO
    ex_dv02_path_b = None  # TODO
    ex_dv02_total = None   # TODO
    ''', "第一条是 2*u 再乘 2；第二条是 1 再乘 3；同一 w 的贡献相加。", r'''
    ex_dv02_path_a = 2 * ex_dv02_u * 2
    ex_dv02_path_b = 3.
    ex_dv02_total = ex_dv02_path_a + ex_dv02_path_b
    ''', r'''
    check("经过 u 的贡献", ex_dv02_path_a, 8.)
    check("经过 v 的贡献", ex_dv02_path_b, 3.)
    check("总导数", ex_dv02_total, 11.)
    ''')
    l.md(r'''
    ## 3 一个非线性函数及其局部导数

    上一课的输出只是输入的加权和。接上非线性函数后，神经元对输入的响应还会随所处区间改变。这里用 Sigmoid，记为 $s(z)=1/(1+e^{-z})$，输出位于 0 与 1 之间。`np.exp` 计算指数，对数组会逐元素运算。

    反向时需要知道输入 z 改变一点，s 会改变多少。对 $(1+e^{-z})^{-1}$ 一层层求导：

    $$s'(z)=-(1+e^{-z})^{-2}\times(-e^{-z})
           =\frac{e^{-z}}{(1+e^{-z})^2}=s(z)[1-s(z)].$$

    最后一步用了 $1-s(z)=e^{-z}/(1+e^{-z})$。这样，前向保存的 s(z) 就能直接用于求导。下方 z=0 的导数是 0.25，z=±2 时约为 0.105；信号处在曲线两端时，同样大小的输入变化引起的输出变化更小。
    ''')
    l.code(r'''
    def sigmoid(z):
        return 1. / (1. + np.exp(-z))
    z_demo = np.array([-2., 0., 2.])
    s_demo = sigmoid(z_demo)
    print("z：", z_demo)
    print("s(z)：", s_demo)
    print("s'(z)：", s_demo * (1 - s_demo))
    ''')
    l.md(r'''
    ## 4 认清同一个 2–2–1 网络

    把刚才的加权和与 Sigmoid 接成一个很小的网络：两个输入坐标进入两个隐藏神经元，再汇合成一个输出。接下来始终追踪这一组连接，直到每个参数都有自己的梯度。

    输入记为 $x_i$，i=0、1；隐藏输出记为 $h_j$，j=0、1；最终输出为 y，这个样本希望得到的目标值为 d。每一步写为

    $$a_j=\sum_{i=0}^{1}x_iW_{ij}+b_j,\quad h_j=s(a_j),\qquad
      z=\sum_{j=0}^{1}h_jv_j+c,\quad y=s(z),\quad L=\tfrac12(y-d)^2.$$

    $a_j$ 和 z 是激活前的加权和。W 的来源下标 i 在前、去向下标 j 在后，所以一列 W 提供一个隐藏神经元所需的输入权重。W 为 `(2,2)`，隐藏偏置 b 为 `(2,)`，输出权重 v 为 `(2,)`，输出偏置 c 为标量；输入 x、隐藏输出 h 都为 `(2,)`。

    W、b、v、c 共有九个可训练数。下面给出它们和一个样本的当前值，目标 d 取 1。此时还没有更新，所有数都能沿图代入计算。
    ''')
    l.code(r'''
    x, d = np.array([0.6, 0.2]), 1.
    W = np.array([[0.1, -0.2], [0.4, 0.3]])
    b = np.array([0.01, -0.02])
    v, c = np.array([0.2, -0.4]), 0.05
    print("x, W, b, v：", x.shape, W.shape, b.shape, v.shape)
    ''')
    l.md(r'''
    例如隐藏节点 0 收到 $0.6\times0.1+0.2\times0.4+0.01=0.15$，节点 1 收到 −0.08。经过 Sigmoid，隐藏输出约为 0.53743、0.48001；它们通过输出权重汇合后，最终 y 约为 0.49137，与目标 1 对应的半平方误差约为 0.12935。

    `forward_one` 按这个依赖顺序计算，将输出和中间量一起返回。反向要用到 h、y，保存它们就能复用。下划线 `_` 可以接收当前暂时不用的返回值。
    ''')
    l.code(r'''
    def forward_one(x, W, b, v, c):
        a = x @ W + b
        h = sigmoid(a)
        z = h @ v + c
        y = sigmoid(z)
        return y, (a, h, z)
    y, (a, h, z) = forward_one(x, W, b, v, c)
    loss = 0.5 * (y-d) ** 2
    print("a、h、z、y、L：", a, h, z, y, loss)
    ''')
    l.md(r'''
    图中每条箭头对应求和里的一项，节点显示这次前向的实际数值。沿隐藏节点 0 的两条输入箭头，可以找到刚才组成 0.15 的两个乘积；b₀ 再补上 0.01，Sigmoid 将加权和变为 h₀。
    ''')
    l.code(r'''
    def plot_small_mlp(x, h, y, W, b, v, c):
        fig, ax = plt.subplots(figsize=(9, 4))
        inputs = np.array([[0., 0.7], [0., -0.7]])
        hidden = np.array([[1., 0.7], [1., -0.7]])
        output = np.array([2., 0.])
        for i in range(2):
            for j in range(2):
                ax.annotate("", hidden[j], inputs[i], arrowprops=dict(arrowstyle="-|>", color="0.4", shrinkA=20, shrinkB=20))
                pos = 0.65 * inputs[i] + 0.35 * hidden[j]
                ax.text(pos[0], pos[1], f"$W_{{{i}{j}}}={W[i,j]:.2f}$", fontsize=9, ha="center", bbox=dict(fc="white", ec="none", pad=0.15))
        for j in range(2):
            ax.annotate("", output, hidden[j], arrowprops=dict(arrowstyle="-|>", color="0.4", shrinkA=20, shrinkB=20))
            pos = 0.5 * (hidden[j] + output)
            ax.text(pos[0], pos[1], f"$v_{j}={v[j]:.2f}$", fontsize=9, ha="center", bbox=dict(fc="white", ec="none", pad=0.15))
        nodes = [(inputs[i], f"$x_{i}$\n{x[i]:.3f}") for i in range(2)]
        nodes += [(hidden[j], f"$h_{j}$\n{h[j]:.3f}") for j in range(2)]
        nodes += [(output, f"$y$\n{y:.3f}")]
        for pos, label in nodes:
            ax.text(pos[0], pos[1], label, ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.5", fc="white", ec="0.3"))
        ax.text(1., 1.18, f"$b_0={b[0]:.2f}$", ha="center")
        ax.text(1., -1.18, f"$b_1={b[1]:.2f}$", ha="center")
        ax.text(2., -0.65, f"$c={c:.2f}$", ha="center")
        ax.set(xlim=(-0.35, 2.35), ylim=(-1.5, 1.5))
        ax.axis("off")
        show(fig)
    plot_small_mlp(x, h, y, W, b, v, c)
    ''')
    l.md(r'''
    [交互实验室：小网络前向](../../interactive/index.html#network) 可以让一条连接单独改变。观察它先影响哪个加权和，又经过哪些节点到达输出；与这里打印的 a、h、z、y 对照，前向计算就是沿这些依赖逐项代入。
    ''')
    l.ex("dv03", "独立完成一次容易手算的前向", "输入 [1,0]，隐藏权重与偏置全为 0，输出权重 [2,−2]，输出偏置 0，目标 1。先用 s(0)=0.5 手算。", r'''
    ex_dv03_x = np.array([1., 0.])
    ex_dv03_W, ex_dv03_b = np.zeros((2, 2)), np.zeros(2)
    ex_dv03_v, ex_dv03_c = np.array([2., -2.]), 0.
    ex_dv03_h = None    # TODO
    ex_dv03_y = None    # TODO
    ex_dv03_loss = None  # TODO
    ''', "隐藏层先 x @ W + b 再 sigmoid；输出对 h @ v + c 做 sigmoid。最后算半平方误差。", r'''
    ex_dv03_h = sigmoid(ex_dv03_x @ ex_dv03_W + ex_dv03_b)
    ex_dv03_y = sigmoid(ex_dv03_h @ ex_dv03_v + ex_dv03_c)
    ex_dv03_loss = 0.5 * (ex_dv03_y - 1.) ** 2
    ''', r'''
    check("两个隐藏输出", ex_dv03_h, [0.5, 0.5])
    check("输出", ex_dv03_y, 0.5)
    check("损失", ex_dv03_loss, 0.125)
    ''')
    l.md(r'''
    ## 5 从损失回到输出层

    当前输出约为 0.49137，目标是 1。输出增大一点会使损失减小，这个趋势由 $\partial L/\partial y=y-d\approx-0.50863$ 表示。

    记 $g_q=\partial L/\partial q$ 为损失对某个量 q 的偏导。从 y 回到激活前的 z，还要乘 Sigmoid 的局部导数。将这个梯度记为 $\delta^y=\partial L/\partial z$：

    $$\delta^y=(y-d)y(1-y).$$

    本次 $\delta^y\approx-0.12712$。它到达加权和 $z=h_0v_0+h_1v_1+c$，便沿着不同输入分开。每条支路乘上各自的局部变化率：

    $$\frac{\partial L}{\partial v_j}=\delta^y h_j,\qquad
      \frac{\partial L}{\partial c}=\delta^y,\qquad
      \frac{\partial L}{\partial h_j}=\delta^y v_j.$$

    以乘积 $h_jv_j$ 为例：v_j 变一点，乘数是当前 h_j；h_j 变一点，乘数则是当前 v_j。它们接到同一个下游梯度，却对应两种不同的扰动。偏置 c 在求和中的系数为 1，所以直接收到 $\delta^y$。
    ''')
    l.code(r'''
    g_y = y - d
    delta_y = g_y * y * (1-y)
    g_v = delta_y * h
    g_c = delta_y
    g_h = delta_y * v
    print("dL/dy、dL/dz：", g_y, delta_y)
    print("输出权重与偏置梯度：", g_v, g_c)
    print("回到两个隐藏输出：", g_h)
    ''')
    l.md(r'''
    两个输出权重的梯度都是负数，但两个隐藏输出的梯度一负一正。原因可以沿连接检查：v₀ 为正，增大 h₀ 会提高 y、减小当前损失；v₁ 为负，增大 h₁ 会降低 y，梯度因而为正。回传的符号已经反映了这两条连接对结果的不同作用。
    ''')
    l.ex("dv04", "区别输出权重与隐藏值的梯度", "给定 y=0.4、d=1、h=[0.5,0.25]、v=[0.2,−0.4]，填写输出误差信号以及两条梯度。", r'''
    ex_dv04_y, ex_dv04_d = 0.4, 1.
    ex_dv04_h, ex_dv04_v = np.array([0.5, 0.25]), np.array([0.2, -0.4])
    ex_dv04_delta = None  # TODO
    ex_dv04_gv = None     # TODO
    ex_dv04_gh = None     # TODO
    ''', "delta=(y-d)*y*(1-y)；gv=delta*h；gh=delta*v。", r'''
    ex_dv04_delta = (ex_dv04_y-ex_dv04_d) * ex_dv04_y * (1-ex_dv04_y)
    ex_dv04_gv = ex_dv04_delta * ex_dv04_h
    ex_dv04_gh = ex_dv04_delta * ex_dv04_v
    ''', r'''
    check("输出激活前梯度", ex_dv04_delta, -0.144)
    check("输出权重梯度", ex_dv04_gv, [-0.072, -0.036])
    check("隐藏输出梯度", ex_dv04_gh, [-0.0288, 0.0576])
    ''')
    l.md(r'''
    ## 6 穿过隐藏层，得到每一条权重的梯度

    刚才得到的是对 h_j 的梯度，它还在隐藏 Sigmoid 的输出端。乘上 $h_j(1-h_j)$，才能回到激活前的 a_j。将这个量记为 $\delta^h_j=\partial L/\partial a_j$，再沿 $a_j=\sum_i x_iW_{ij}+b_j$ 回到输入权重和偏置：

    $$\delta^h_j=(\delta^y v_j)h_j(1-h_j),\qquad
      \frac{\partial L}{\partial W_{ij}}=x_i\delta^h_j,\qquad
      \frac{\partial L}{\partial b_j}=\delta^h_j.$$

    同一个隐藏节点 j 向它的两条输入连接传回 $\delta^h_j$，两条权重分别再乘自己的输入 x_i。四条连接因此得到一个 `(2,2)` 梯度表。`x[:,None]` 为 `(2,1)`，`delta_h[None,:]` 为 `(1,2)`，广播相乘便把所有 $x_i\delta^h_j$ 排在对应位置。
    ''')
    l.code(r'''
    delta_h = g_h * h * (1-h)
    g_W = x[:, None] * delta_h[None, :]
    g_b = delta_h.copy()
    print("两个隐藏激活前梯度：", delta_h)
    print("四条输入到隐藏权重的梯度：\n", g_W)
    print("隐藏偏置梯度：", g_b)
    ''')
    l.md(r'''
    输出里的两行权重梯度相差三倍，因为 x₀=0.6、x₁=0.2，而通向同一个隐藏节点的误差信号相同。每一项数值都可以追溯到“这个输入有多大”和“这个隐藏节点收到多大梯度”。

    还可以继续走回原始输入。x_i 同时通向两个隐藏节点，它的一次扰动会影响两条支路，所以回到这个输入时要把贡献收齐：

    $$\frac{\partial L}{\partial x_i}
      =\sum_{j=0}^{1}\frac{\partial L}{\partial a_j}
                       \frac{\partial a_j}{\partial x_i}
      =W_{i0}\delta^h_0+W_{i1}\delta^h_1.$$

    矩阵写法是 `W @ delta_h`。本例 x₁ 经过隐藏节点 0 的贡献约为 −0.002528，经过节点 1 的贡献约为 0.003807，部分抵消后留下 0.001279。这种汇合也会发生在有更多分支的网络里。

    输入梯度描述输入坐标改变时的损失变化。训练这批给定样本时，x 保持固定，更新的是 W、b、v、c；计算输入梯度让我们看到模型对输入的灵敏度。
    ''')
    l.code(r'''
    g_x = W @ delta_h
    input_path_contributions = W * delta_h[None, :]
    print("每个输入经过两个隐藏节点的贡献：\n", input_path_contributions)
    print("逐行相加：", input_path_contributions.sum(axis=1))
    print("矩阵得到的输入梯度：", g_x)
    ''')
    l.md(r'''
    在[交互实验室的反向回传](../../interactive/index.html#backprop)中，可以沿一条输入路径逐项查看乘数，再将同一输入收到的贡献相加。把一条连接的正负号改掉，观察路径贡献和总梯度怎样随之改变。
    ''')
    l.ex("dv05", "收齐隐藏与输入的梯度", "使用上一练习相同的 delta、h、v，给定 x 与 W，计算隐藏激活前梯度、四条权重梯度，以及输入梯度。", r'''
    ex_dv05_delta_y = -0.144
    ex_dv05_h, ex_dv05_v = np.array([0.5, 0.25]), np.array([0.2, -0.4])
    ex_dv05_x = np.array([0.6, 0.2])
    ex_dv05_W = np.array([[0.1, -0.2], [0.4, 0.3]])
    ex_dv05_delta_h = None  # TODO
    ex_dv05_gW = None       # TODO
    ex_dv05_gx = None       # TODO
    ''', "delta_h=delta_y*v*h*(1-h)；gW 的第 ij 项为 x_i*delta_h_j；输入梯度将同一行的两个路径贡献相加。", r'''
    ex_dv05_delta_h = ex_dv05_delta_y * ex_dv05_v * ex_dv05_h * (1-ex_dv05_h)
    ex_dv05_gW = ex_dv05_x[:, None] * ex_dv05_delta_h[None, :]
    ex_dv05_gx = ex_dv05_W @ ex_dv05_delta_h
    ''', r'''
    check("隐藏梯度", ex_dv05_delta_h, [-0.0072, 0.0108])
    check("权重梯度", ex_dv05_gW, [[-0.00432, 0.00648], [-0.00144, 0.00216]])
    check("输入梯度", ex_dv05_gx, [-0.00288, 0.00036])
    ''')
    l.md(r'''
    ## 7 把已经推过的反向计算写成函数

    梯度已经逐项算出，可以像第 02 课整理测量评估器那样，把它们收进一个函数。`backward_one` 调用前向取得 h、y，沿输出、隐藏层的顺序向回计算，字典中的键标明每个梯度属于哪组参数，x 则保存输入梯度。
    ''')
    l.code(r'''
    def backward_one(x, d, W, b, v, c):
        y, (_, h, _) = forward_one(x, W, b, v, c)
        delta_y = (y-d) * y * (1-y)
        delta_h = delta_y * v * h * (1-h)
        return {"W": x[:, None] * delta_h[None, :], "b": delta_h.copy(),
                "v": delta_y * h, "c": delta_y, "x": W @ delta_h}
    gradients = backward_one(x, d, W, b, v, c)
    print(gradients)
    ''')
    l.md(r'''
    ## 8 对九个参数逐一做有限差分

    现在同一个网络有两种测量变化率的办法：反向函数沿路径计算；中心差分则实际扰动参数，重新算两次损失。它们可以逐项对照。

    将九个参数排成向量 $\theta$：按行展开 W 的四项，接上 b 的两项、v 的两项和 c。`ravel()` 按行展平，`np.concatenate` 将一维数组首尾相接。检查第 q 个参数时，只在那个位置加、减 $\varepsilon$：

    $$\frac{\partial L}{\partial\theta_q}\approx
       \frac{L(\theta+\varepsilon e_q)-L(\theta-\varepsilon e_q)}{2\varepsilon},$$

    $e_q$ 是仅第 q 项为 1 的方向。`unpack` 将向量还原成前向所需的 W、b、v、c，`*unpack(theta)` 把这四个返回值依次传给相应参数。每次使用独立副本，正负两次扰动围绕同一个原始取值展开。
    ''')
    l.code(r'''
    def unpack(theta):
        return theta[:4].reshape(2, 2), theta[4:6], theta[6:8], theta[8]
    def loss_theta(theta, x, d):
        prediction, _ = forward_one(x, *unpack(theta))
        return 0.5 * (prediction-d) ** 2
    theta = np.concatenate([W.ravel(), b, v, np.array([c])])
    analytic_gradient = np.concatenate([gradients["W"].ravel(), gradients["b"],
                                        gradients["v"], np.array([gradients["c"]])])
    ''')
    l.md(r'''
    例如 W₀₁ 的解析梯度约为 0.007615。把这条权重增加一点，差分应测到损失上升；W₀₀ 的梯度约为 −0.003792，方向相反。下一格将九个有名字的偏导并排打印，检查数值大小和符号是否都与扰动实验一致。
    ''')
    l.code(r'''
    epsilon = 1e-5
    numerical_gradient = np.zeros(len(theta))
    for q in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[q] += epsilon
        minus[q] -= epsilon
        numerical_gradient[q] = (loss_theta(plus, x, d)-loss_theta(minus, x, d)) / (2*epsilon)
    for name, exact, numeric in zip(["W00", "W01", "W10", "W11", "b0", "b1", "v0", "v1", "c"], analytic_gradient, numerical_gradient):
        print(f"{name:>3}: analytic={exact:+.8f}, difference={numeric:+.8f}")
    check("九个参数的梯度核对", analytic_gradient, numerical_gradient)
    ''')
    l.md(r'''
    九行结果能一一对上以后，也可对输入做同样的实验。参数和目标 d 固定，仅将一个输入坐标向两侧扰动，再比较损失。这样，刚才由两条隐藏路径汇合出的输入梯度，也有了独立的数值核对。
    ''')
    l.code(r'''
    numerical_input_gradient = np.zeros(len(x))
    for i in range(len(x)):
        plus_x, minus_x = x.copy(), x.copy()
        plus_x[i] += epsilon
        minus_x[i] -= epsilon
        numerical_input_gradient[i] = (loss_theta(theta, plus_x, d)-loss_theta(theta, minus_x, d)) / (2*epsilon)
    check("两个输入坐标的梯度核对", gradients["x"], numerical_input_gradient)
    ''')
    l.md(r'''
    ## 9 用同一次梯度更新所有参数

    梯度的符号已经给出每个参数在附近的增大方向。为了减小损失，更新时沿反方向走，步幅由学习率 $\eta>0$ 控制：$\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta L$。

    本例 W₀₁ 的梯度为正，因此更新使它减小；输出偏置 c 的梯度为负，更新使它增大。四组参数都使用刚才从旧参数算出的梯度，样本 x、d 继续保持原值。学习率取 0.2 后，重新前向得到的损失由约 0.12935 降到 0.12443，一次具体更新验证了这一步的效果。
    ''')
    l.code(r'''
    learning_rate = 0.2
    W_next = W - learning_rate * gradients["W"]
    b_next = b - learning_rate * gradients["b"]
    v_next = v - learning_rate * gradients["v"]
    c_next = c - learning_rate * gradients["c"]
    y_next, _ = forward_one(x, W_next, b_next, v_next, c_next)
    print("更新前、后损失：", loss, 0.5 * (y_next-d) ** 2)
    ''')
    l.ex("dv06", "给参数做一次梯度下降", "两项参数为 [1,−1]，梯度为 [0.2,−0.4]，学习率 0.1。计算新参数，保留旧参数供核对。", r'''
    ex_dv06_parameters = np.array([1., -1.])
    ex_dv06_gradient = np.array([0.2, -0.4])
    ex_dv06_updated = None  # TODO
    ''', "新参数 = 原参数 - 0.1 * 梯度。用新的变量接收结果。", r'''
    ex_dv06_updated = ex_dv06_parameters - 0.1 * ex_dv06_gradient
    ''', r'''
    check("新参数", ex_dv06_updated, [0.98, -0.96])
    check("保留的旧参数", ex_dv06_parameters, [1., -1.])
    ''')
    l.md(r'''
    ## 10 一批样本共享参数，梯度先相加再平均

    一个样本上走对一步，还不能确定同一组参数怎样同时照顾多个样本。现在把小网络用于 XOR：`[0,0]→0`、`[0,1]→1`、`[1,0]→1`、`[1,1]→0`。四行输入共享同一组参数，目标在两个输入不相同时为 1。

    记批大小为 B，X 为 `(B,2)`，隐藏输出 H 为 `(B,2)`，输出 y 与目标 d 为 `(B,)`。批损失定义为 $L_B=\frac1{2B}\sum_{i=0}^{B-1}(y_i-d_i)^2$。每个样本的输出误差信号因此包含一次 $1/B$：

    $$\Delta^y_i=\frac{(y_i-d_i)y_i(1-y_i)}{B},\qquad
      \Delta^h_{ij}=\Delta^y_i v_jH_{ij}(1-H_{ij}).$$

    一条 W 在四次前向里都被使用，所以四个样本给它的梯度要相加。这次汇合沿样本轴发生，矩阵乘法恰好把这一轴求和：

    $$\nabla_vL_B=H^\top\Delta^y,\quad \partial_cL_B=\sum_i\Delta^y_i,\qquad
      \nabla_WL_B=X^\top\Delta^h,\quad \nabla_bL_B=\sum_i\Delta^h_{i,:}.$$

    尺寸分别为 `(2,B) @ (B,) → (2,)`、`(2,B) @ (B,2) → (2,2)`。1/B 已放进 Delta_y，表示每个样本先分担平均损失中的一份，再沿共享权重汇合。把 B 取 1，就能对应回刚才逐项推导的单样本计算。
    ''')
    l.code(r'''
    X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    d_xor = np.array([0., 1., 1., 0.])
    def batch_gradients(X, d, W, b, v, c):
        H = sigmoid(X @ W + b)
        y = sigmoid(H @ v + c)
        loss = 0.5 * np.mean((y-d) ** 2)
        delta_y = (y-d) * y * (1-y) / len(X)
        delta_h = delta_y[:, None] * v[None, :] * H * (1-H)
        gradients = {"W": X.T @ delta_h, "b": delta_h.sum(axis=0),
                     "v": H.T @ delta_y, "c": delta_y.sum()}
        return loss, gradients, y
    ''')
    l.md(r'''
    ## 11 从随机参数出发，重复真正的更新

    到这里可以真正训练了：从一组初始参数出发，重复计算整批损失、求梯度、更新参数。`np.random.default_rng(seed)` 建立可复现的随机数生成器，`rng.normal(size=形状)` 按指定形状生成标准正态随机权重；两个隐藏节点从不同权重出发，偏置取 0。

    下面两次训练使用相同的 seed=0，因此从同一个网络起步，只将学习率从 0.1 换成 1.0。每次更新 12000 轮，每 100 轮保存一个损失。这样能看到步长选择怎样影响这段学习过程。

    输出用 `np.round` 保留小数位。`prediction >= 0.5` 对四行分别作判断，`.astype(int)` 将布尔值转成 0、1，与 XOR 目标逐项比较。
    ''')
    l.code(r'''
    def train_numpy_mlp(X, d, learning_rate, steps=12000, seed=0):
        rng = np.random.default_rng(seed)
        W, b = rng.normal(size=(2, 2)), np.zeros(2)
        v, c = rng.normal(size=2), 0.
        history = []
        for step in range(steps+1):
            loss, g, prediction = batch_gradients(X, d, W, b, v, c)
            if step % 100 == 0:
                history.append([step, loss])
            if step == steps:
                break
            W -= learning_rate * g["W"]
            b -= learning_rate * g["b"]
            v -= learning_rate * g["v"]
            c -= learning_rate * g["c"]
        return {"W": W, "b": b, "v": v, "c": c, "history": np.array(history),
                "prediction": prediction, "loss": loss}
    ''')
    l.code(r'''
    run_small_lr = train_numpy_mlp(X_xor, d_xor, learning_rate=0.1)
    run_large_lr = train_numpy_mlp(X_xor, d_xor, learning_rate=1.0)
    for rate, result in [(0.1, run_small_lr), (1., run_large_lr)]:
        print("learning rate:", rate, "loss:", result["loss"])
        print("prediction:", np.round(result["prediction"], 4))
    check("lr=1 的四个角点分类", (run_large_lr["prediction"] >= 0.5).astype(int), d_xor)
    ''')
    l.md(r'''
    学习率 0.1 的四个输出仍在 0.5 附近，损失约 0.12497；学习率 1.0 的输出约为 `[0.0258,0.9756,0.9708,0.0229]`，已经按目标分开，损失约 0.000330。

    两条曲线从相同参数出发，却在固定轮数内走到不同位置。纵轴改用对数刻度，才能同时看清初始误差与后来降下来的数量级。一次初始化给出了这个对照，换种子还可以继续观察结论怎样变化。
    ''')
    l.code(r'''
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for rate, result in [(0.1, run_small_lr), (1., run_large_lr)]:
        history = result["history"]
        ax.semilogy(history[:, 0], history[:, 1], label=f"Learning rate = {rate}")
    ax.set(xlabel="Update step", ylabel="Half mean squared error")
    ax.legend()
    show(fig)
    ''')
    l.md(r'''
    四个角点的数值已经对上，但小网络对中间位置也会给出输出。把一片二维网格交给训练好的参数，颜色便显示这条连续函数怎样从一类区域过渡到另一类。

    `np.meshgrid` 铺出横纵坐标网格，`ravel` 展平后，`np.column_stack` 将两列拼成 `(样本数,2)`。整批前向的输出再 reshape 回网格，四个实际训练点叠在上面。可以把某个中间位置代回公式，核对颜色对应的数值。
    ''')
    l.code(r'''
    grid_axis = np.linspace(-0.2, 1.2, 81)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    fitted = run_large_lr
    grid_prediction = sigmoid(sigmoid(grid_points @ fitted["W"] + fitted["b"]) @ fitted["v"] + fitted["c"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.contourf(grid_x, grid_y, grid_prediction.reshape(grid_x.shape), levels=np.linspace(0., 1., 11), cmap="coolwarm")
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=d_xor, cmap="coolwarm", vmin=0, vmax=1, edgecolors="black", s=70)
    ax.set(xlabel="Input coordinate 0", ylabel="Input coordinate 1")
    fig.colorbar(image, ax=ax, label="Network output")
    show(fig)
    ''')
    l.md(r'''
    ## 把小网络带到另一个任务

    试着在新代码格里重新写出前向、反向和更新循环，选一组自己的参数，用中心差分核对九个参数与两个输入。写到某条 W 的梯度时，可以在图中指出它经过的隐藏节点，并沿路径找出每个乘数。

    将四个目标改成 AND 的 `[0,0,0,1]`，相同结构就面对了另一个任务。记录它的损失和四点预测，再与 XOR 比较。也可以保留任务，只把隐藏 Sigmoid 换成恒等映射，把前向式子展开，观察两层加权和最终还能表示怎样的函数。

    差分步长和学习率值得分别试：前者改变数值梯度的估计，后者改变参数实际走过的距离。将几种步长的梯度误差、几种学习率的训练曲线各自记录下来，遇到效果变化时，便有办法追到具体计算。

    这课的“分支贡献相加”还可以直接接到残差连接。若一个分支输出为 F(x)，并与输入相加得到 $r=x+F(x)$，那么 $dr/dx=1+F'(x)$。在[残差交互实验](../../interactive/index.html#residual)里设 $F(x)=ax+b$，改变 a，观察直连贡献 1 与分支贡献 a 怎样相加。相加后若接激活，还要沿那一层继续乘局部导数；第 09 课会把它接到实际网络中。

    ## 带着问题读原文

    [《动手学深度学习》微积分](https://d2l.ai/chapter_preliminaries/calculus.html) 中的导数、偏导和链式法则，现在分别对应一段差商、九个参数检查与网络上的路径。可以选 W₀₁，将它到损失的完整计算写在纸上，逐个标出局部导数；也可以选一个输入，解释它为什么必须收齐两条隐藏支路。
    ''')
    return l


def build():
    return [python_values(), python_functions(), numpy_algebra(), derivatives_mlp()]
