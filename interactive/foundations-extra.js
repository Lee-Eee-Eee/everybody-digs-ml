/* Batch gradients and coupled output losses, using the persistent course network. */
(function () {
  'use strict';
  var BLUE = '#2c647d', ORANGE = '#be7040', GRAY = '#d1dad6', GREEN = '#3d6956', RED = '#aa4740';
  var serial = 0;
  function sub(n) { return n===1||n==='1'?'₁':'₂'; }
  function clone(s) { return JSON.parse(JSON.stringify(s)); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]); }); }
  function weightKey(key) { return { layer:key[0]==='w'?0:1,i:Number(key[1])-1,j:Number(key[2])-1,name:key[0]+sub(key[1])+sub(key[2]) }; }

  /* =========================================================================
     Batch Lab: Lesson 7 Interactive Matrix BP Visualizer
     ========================================================================= */
  function batchLab(ctx, shared) {
    var root = ctx.root, M = ctx.M, F = ctx.fmt, base = clone(shared.state);
    if (!base.d) base.d = base.target || [1, 0];
    var currentTab = 'matrix'; // 'matrix', 'samples', 'shapes'
    var selectedWeight = 'w11';

    var weights = [
      ['w11', 'w₁₁：输入 x₁ → 隐藏 h₁'],
      ['w12', 'w₁₂：输入 x₁ → 隐藏 h₂'],
      ['w21', 'w₂₁：输入 x₂ → 隐藏 h₁'],
      ['w22', 'w₂₂：输入 x₂ → 隐藏 h₂'],
      ['v11', 'v₁₁：隐藏 h₁ → 输出 y₁'],
      ['v12', 'v₁₂：隐藏 h₁ → 输出 y₂'],
      ['v21', 'v₂₁：隐藏 h₂ → 输出 y₁'],
      ['v22', 'v₂₂：隐藏 h₂ → 输出 y₂']
    ];

    root.innerHTML = `
      <div class="fe-shell">
        <style>
          .fe-shell { font-size: 15px; line-height: 1.7; color: #23323b; }
          .fe-toolbar { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 16px; align-items: flex-end; }
          .fe-toolbar > label { flex: 1 1 180px; min-width: 0; }
          .fe-tabs { display: flex; gap: 8px; border-bottom: 2px solid #e2e8e7; padding-bottom: 8px; margin-bottom: 16px; }
          .fe-tab-btn { padding: 8px 16px; border-radius: 6px; border: 1px solid #dce2e1; background: #fff; font-weight: 600; cursor: pointer; color: #69767d; }
          .fe-tab-btn.active { background: #2c647d; border-color: #2c647d; color: #fff; }
          .fe-card { padding: 16px 20px; border-radius: 8px; background: #fff; border: 1px solid #dce2e1; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
          .fe-card h3 { font-size: 16px; color: #2c647d; margin: 0 0 10px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
          .fe-grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 16px; }
          .fe-matrix-wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: center; margin: 18px 0; }
          .fe-mat-box { text-align: center; }
          .fe-mat-box .mat-title { font-weight: 700; color: #2c647d; margin-bottom: 6px; font-size: 14px; }
          .fe-table-matrix { border-collapse: separate; border-spacing: 4px; font-family: Consolas, monospace; font-size: 14px; margin: 0 auto; }
          .fe-table-matrix td { padding: 8px 12px; background: #f0f4f5; border: 2px solid transparent; border-radius: 4px; text-align: center; cursor: pointer; transition: all 0.15s; }
          .fe-table-matrix td:hover { border-color: #be7040; background: #fff2e8; }
          .fe-table-matrix td.cell-active { background: #fff0e2; border-color: #be7040; font-weight: bold; color: #be7040; }
          .fe-table-matrix td.row-active { background: #e6f1f5; border-color: #2c647d; color: #2c647d; }
          .fe-table-matrix td.col-active { background: #eaf3ed; border-color: #3d6956; color: #3d6956; }
          .fe-symbol { font-size: 20px; font-weight: bold; color: #69767d; }
          .fe-math-banner { background: #fcf6f1; border-left: 4px solid #be7040; padding: 12px 16px; border-radius: 4px; font-family: Consolas, monospace; font-size: 15px; margin: 14px 0; }
          .fe-math-banner strong { color: #be7040; }
          .fe-dim-table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 10px 0; }
          .fe-dim-table th, .fe-dim-table td { padding: 9px 12px; border-bottom: 1px solid #e2e8e7; text-align: left; }
          .fe-dim-table th { background: #f4f7f6; color: #2c647d; font-weight: 700; }
          .fe-mini-net { display: flex; gap: 16px; justify-content: space-around; flex-wrap: wrap; margin-top: 10px; }
          .fe-mini-card { flex: 1 1 240px; padding: 12px; border: 1px solid #e2e8e7; border-radius: 6px; background: #fbfcfb; font-size: 13.5px; }
          .fe-mini-card strong { color: #2c647d; }
        </style>
        <div class="fe-tabs">
          <button type="button" class="fe-tab-btn" data-tab="matrix">矩阵中的梯度</button>
          <button type="button" class="fe-tab-btn" data-tab="samples">逐样本梯度</button>
          <button type="button" class="fe-tab-btn" data-tab="shapes">形状与运算</button>
        </div>
        <div class="fe-toolbar">
          ${ctx.select('选择追踪的权重', 'fe-b-weight', weights, 'w11')}
          ${ctx.range('样本 2 输入 x₁', 'fe-b-x1', -1, 1, 0.05, 0.2)}
          ${ctx.range('样本 2 输入 x₂', 'fe-b-x2', -1, 1, 0.05, 0.8)}
          ${ctx.select('样本 2 目标 d', 'fe-b-target', [['01', 'd = [0, 1]'], ['10', 'd = [1, 0]']], '01')}
        </div>
        <div id="fe-b-content"></div>
      </div>
    `;

    function render() {
      // Update tab buttons
      root.querySelectorAll('.fe-tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.tab === currentTab);
      });

      selectedWeight = root.querySelector('#fe-b-weight').value;
      var chosen = weightKey(selectedWeight);
      var i = chosen.i, j = chosen.j, layer = chosen.layer;

      // Sample 1 (base from shared state) and Sample 2
      var s2_x1 = +root.querySelector('#fe-b-x1').value;
      var s2_x2 = +root.querySelector('#fe-b-x2').value;
      var s2_t = root.querySelector('#fe-b-target').value === '01' ? [0, 1] : [1, 0];
      var second = { x: [s2_x1, s2_x2], W0: base.W0, W1: base.W1, d: s2_t };

      var states = [base, second];
      var fs = states.map(function(s) { return M.toyForward(s); });

      // Gradients for chosen weight
      var source1 = layer === 0 ? states[0].x[i] : fs[0].h[i];
      var delta1 = layer === 0 ? fs[0].d1[j] : fs[0].d2[j];
      var grad1 = source1 * delta1;

      var source2 = layer === 0 ? states[1].x[i] : fs[1].h[i];
      var delta2 = layer === 0 ? fs[1].d1[j] : fs[1].d2[j];
      var grad2 = source2 * delta2;

      var avgGrad = (grad1 + grad2) / 2;

      // Matrices
      var Xmat = states.map(function(s, n) { return layer === 0 ? s.x : fs[n].h; }); // 2 x 2
      var XTmat = [0, 1].map(function(r) { return [0, 1].map(function(c) { return Xmat[c][r]; }); }); // 2 x 2
      var Dmat = fs.map(function(f) { return layer === 0 ? f.d1 : f.d2; }); // 2 x 2 (Delta matrix)
      var Gmat = [0, 1].map(function(r) {
        return [0, 1].map(function(c) {
          return ((layer === 0 ? fs[0].g0 : fs[0].g1)[r][c] + (layer === 0 ? fs[1].g0 : fs[1].g1)[r][c]) / 2;
        });
      });

      var container = root.querySelector('#fe-b-content');

      if (currentTab === 'matrix') {
        var matName = layer === 0 ? 'X' : 'H';
        var gradName = layer === 0 ? 'W' : 'V';
        container.innerHTML = `
          <div class="fe-card">
            <h3>${gradName} 的批量平均梯度：${matName}ᵀ × Δ / B</h3>
            <p>本例 B=2，批量损失 L=(L₁+L₂)/2。Δ 的每一行是<strong>一个样本损失对本层激活前数值的导数，尚未除以 B</strong>。点击右侧梯度矩阵中的格子，左侧会标出参与计算的行与列；点积先将两个样本的贡献相加，再除以 2。</p>
            <div class="fe-matrix-wrap">
              <div class="fe-mat-box">
                <div class="mat-title">${matName}ᵀ（行：来源节点，列：样本）</div>
                <table class="fe-table-matrix" id="mat-xt">
                  <tr>
                    <td class="${i===0?'row-active':''}" data-cell="0,0">${F(XTmat[0][0], 3)}</td>
                    <td class="${i===0?'row-active':''}" data-cell="0,1">${F(XTmat[0][1], 3)}</td>
                  </tr>
                  <tr>
                    <td class="${i===1?'row-active':''}" data-cell="1,0">${F(XTmat[1][0], 3)}</td>
                    <td class="${i===1?'row-active':''}" data-cell="1,1">${F(XTmat[1][1], 3)}</td>
                  </tr>
                </table>
              </div>
              <div class="fe-symbol">×</div>
              <div class="fe-mat-box">
                <div class="mat-title">Δ（行：样本，列：本层节点）</div>
                <table class="fe-table-matrix" id="mat-delta">
                  <tr>
                    <td class="${j===0?'col-active':''}" data-cell="0,0">${F(Dmat[0][0], 4)}</td>
                    <td class="${j===1?'col-active':''}" data-cell="0,1">${F(Dmat[0][1], 4)}</td>
                  </tr>
                  <tr>
                    <td class="${j===0?'col-active':''}" data-cell="1,0">${F(Dmat[1][0], 4)}</td>
                    <td class="${j===1?'col-active':''}" data-cell="1,1">${F(Dmat[1][1], 4)}</td>
                  </tr>
                </table>
              </div>
              <div class="fe-symbol">/ 2 =</div>
              <div class="fe-mat-box">
                <div class="mat-title">∇${gradName} (参数梯度矩阵 2×2)</div>
                <table class="fe-table-matrix" id="mat-g">
                  <tr>
                    <td class="${i===0&&j===0?'cell-active':''}" data-weight="${layer===0?'w':'v'}11">${F(Gmat[0][0], 5)}</td>
                    <td class="${i===0&&j===1?'cell-active':''}" data-weight="${layer===0?'w':'v'}12">${F(Gmat[0][1], 5)}</td>
                  </tr>
                  <tr>
                    <td class="${i===1&&j===0?'cell-active':''}" data-weight="${layer===0?'w':'v'}21">${F(Gmat[1][0], 5)}</td>
                    <td class="${i===1&&j===1?'cell-active':''}" data-weight="${layer===0?'w':'v'}22">${F(Gmat[1][1], 5)}</td>
                  </tr>
                </table>
              </div>
            </div>
            <div class="fe-math-banner">
              <strong>当前选中：${chosen.name}</strong> 位于第 ${i+1} 行、第 ${j+1} 列：<br>
              ∇${chosen.name} = (${matName}ᵀ[第 ${i+1} 行] · Δ[第 ${j+1} 列]) / 2<br>
              = ( ${F(XTmat[i][0], 4)} × ${F(Dmat[0][j], 4)} + ${F(XTmat[i][1], 4)} × ${F(Dmat[1][j], 4)} ) / 2<br>
              = ( ${F(grad1, 5)} [样本1] + ${F(grad2, 5)} [样本2] ) / 2 = <strong>${F(avgGrad, 5)}</strong>
            </div>
            <p style="font-size:13.5px;color:#69767d;margin-top:8px">
              <strong>转置把同一来源节点的 B 个样本值排在一行。</strong>${matName} 的形状为 (B×2)，Δ 为 (B×2)。${matName}ᵀ 的一行与 Δ 的一列相乘，沿样本轴求和，得到某条权重的梯度贡献和；再除以 B，得到平均梯度。结果为 (2×2)，行、列分别对应权重的来源与去向。
            </p>
          </div>
        `;

        // Bind clicks on matrix G cells
        container.querySelectorAll('#mat-g td').forEach(function(td) {
          td.addEventListener('click', function() {
            var w = td.dataset.weight;
            if (w) {
              root.querySelector('#fe-b-weight').value = w;
              render();
            }
          });
        });
      } else if (currentTab === 'samples') {
        container.innerHTML = `
          <div class="fe-card">
            <h3>同一条权重 ${chosen.name} 的两份梯度贡献</h3>
            <p>两个样本使用同一组权重，各自根据预测与目标计算损失。批量损失取两份损失的平均，因此这条权重的批量梯度也取两份单样本梯度的<strong>算术平均值</strong>：</p>
            <div class="fe-grid-2">
              <div class="fe-mini-card">
                <strong>样本 1（进入本实验时的样本）</strong><br>
                输入 x = [${base.x.join(', ')}]，目标 d = [${base.d.join(', ')}]<br>
                预测 y = [${fs[0].y.map(function(v){return F(v,3);}).join(', ')}]，损失 L₁ = ${F(fs[0].loss, 5)}<br>
                来源节点数值：${F(source1, 4)}<br>
                损失对去向节点激活前数值的导数 δ：${F(delta1, 5)}<br>
                <div style="margin-top:6px;padding:6px;background:#fff;border-radius:4px;border:1px solid #dce2e1">
                  单样本梯度 g₁ = ${F(source1, 4)} × ${F(delta1, 5)} = <strong>${F(grad1, 6)}</strong>
                </div>
              </div>
              <div class="fe-mini-card">
                <strong>样本 2（上方可调的样本）</strong><br>
                输入 x = [${second.x.join(', ')}]，目标 d = [${second.d.join(', ')}]<br>
                预测 y = [${fs[1].y.map(function(v){return F(v,3);}).join(', ')}]，损失 L₂ = ${F(fs[1].loss, 5)}<br>
                来源节点数值：${F(source2, 4)}<br>
                损失对去向节点激活前数值的导数 δ：${F(delta2, 5)}<br>
                <div style="margin-top:6px;padding:6px;background:#fff;border-radius:4px;border:1px solid #dce2e1">
                  单样本梯度 g₂ = ${F(source2, 4)} × ${F(delta2, 5)} = <strong>${F(grad2, 6)}</strong>
                </div>
              </div>
            </div>
            <div class="fe-math-banner">
              Batch 平均梯度 = (g₁ + g₂) / 2 = (${F(grad1, 5)} + ${F(grad2, 5)}) / 2 = <strong>${F(avgGrad, 6)}</strong>
            </div>
            <p style="font-size:13.5px;color:#69767d">
              调整样本 2 的输入或目标，g₂ 会随预测变化，g₁ 保持不变。两份梯度同号时相互加强，异号时部分抵消。梯度下降将这两份梯度取平均，再沿平均梯度的反方向更新权重。
            </p>
          </div>
        `;
      } else if (currentTab === 'shapes') {
        container.innerHTML = `
          <div class="fe-card">
            <h3>本例的前向与反向矩阵</h3>
            <p>每行存一个样本，批量大小为 B（本例 B=2），输入特征数 I=2，隐藏宽度 J=2，输出节点数 K=2。输入权重记为 W，输出权重记为 V；A₁、A₂ 是两层激活前的数值。本例使用 Sigmoid 与平方损失，Δ 按单样本损失定义，权重梯度显式除以 B。</p>
            <table class="fe-dim-table">
              <thead>
                <tr>
                  <th>计算阶段</th>
                  <th>数学公式</th>
                  <th>张量维度运算</th>
                  <th>计算含义</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>隐藏层前向</strong></td>
                  <td>A₁ = X W</td>
                  <td>(B×I) · (I×J) = (B×J)</td>
                  <td>每一行对应一个样本的隐藏加权和</td>
                </tr>
                <tr>
                  <td><strong>隐藏层激活</strong></td>
                  <td>H = σ(A₁)</td>
                  <td>(B×J) → (B×J)</td>
                  <td>逐元素激活，不改变矩阵形状</td>
                </tr>
                <tr>
                  <td><strong>输出层前向</strong></td>
                  <td>A₂ = H V</td>
                  <td>(B×J) · (J×K) = (B×K)</td>
                  <td>每一行对应一个样本的输出加权和</td>
                </tr>
                <tr>
                  <td><strong>输出与损失</strong></td>
                  <td>Y = σ(A₂)，L = Σₙ Lₙ / B</td>
                  <td>(B×K) → B 份损失 → 标量</td>
                  <td>Lₙ = ½ Σₖ(Yₙₖ−Dₙₖ)²，D 存各样本目标</td>
                </tr>
                <tr>
                  <td><strong>输出层误差</strong></td>
                  <td>Δ₂[n,k] = ∂Lₙ/∂A₂[n,k]</td>
                  <td>(B×K)</td>
                  <td>每行对应单样本损失的导数，尚未除以 B</td>
                </tr>
                <tr>
                  <td><strong>输出权重梯度</strong></td>
                  <td>∇V L = Hᵀ Δ₂ / B</td>
                  <td>(J×B) · (B×K) = (J×K)</td>
                  <td>沿样本轴求和并除以 B，形状与 V 相同</td>
                </tr>
                <tr>
                  <td><strong>回传至隐藏层</strong></td>
                  <td>Δ₁ = (Δ₂ Vᵀ) ⊙ σ'(A₁)</td>
                  <td>(B×K)·(K×J) ⊙ (B×J) = (B×J)</td>
                  <td>先累加各输出节点的回传贡献，再逐项乘激活导数</td>
                </tr>
                <tr>
                  <td><strong>输入权重梯度</strong></td>
                  <td>∇W L = Xᵀ Δ₁ / B</td>
                  <td>(I×B) · (B×J) = (I×J)</td>
                  <td>沿样本轴求和并除以 B，形状与 W 相同</td>
                </tr>
              </tbody>
            </table>
            <div style="background:#fff6f5;border-left:4px solid #aa4740;padding:12px 16px;border-radius:4px;margin-top:14px">
              <strong>乘法顺序对应不同的求和轴。</strong><br>
              X 为 (B×I)，Δ₁ 为 (B×J)。<code>X.T @ delta</code> 沿样本轴求和，得到 (I×J)，对应输入来源与隐藏去向。<code>X @ delta.T</code> 只有 I=J 时才可相乘，结果为 (B×B)，其两个轴都对应样本。当前小例子的维度都等于 2，更需要逐项看清每个轴代表什么。
            </div>
          </div>
        `;
      }
    }

    // Tab buttons click handler
    root.querySelectorAll('.fe-tab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        currentTab = btn.dataset.tab;
        render();
      });
    });

    ctx.controls(render);
    render();
  }

  /* =========================================================================
     Softmax Lab: Lesson 8 Interactive Softmax vs Sigmoid Visualizer
     ========================================================================= */
  function softmaxLab(ctx, shared) {
    var root = ctx.root, M = ctx.M, F = ctx.fmt, state = clone(shared.state);
    if (!state.d) state.d = state.target || [1, 0];
    var currentMode = 'softmax'; // 'softmax' or 'sigmoid'

    root.innerHTML = `
      <div class="fe-shell">
        <style>
          .fe-shell { font-size: 15px; line-height: 1.7; color: #23323b; }
          .fe-tabs { display: flex; gap: 8px; border-bottom: 2px solid #e2e8e7; padding-bottom: 8px; margin-bottom: 16px; }
          .fe-tab-btn { padding: 8px 16px; border-radius: 6px; border: 1px solid #dce2e1; background: #fff; font-weight: 600; cursor: pointer; color: #69767d; }
          .fe-tab-btn.active { background: #2c647d; border-color: #2c647d; color: #fff; }
          .fe-card { padding: 16px 20px; border-radius: 8px; background: #fff; border: 1px solid #dce2e1; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
          .fe-card h3 { font-size: 16px; color: #2c647d; margin: 0 0 10px; font-weight: 700; }
          .fe-toolbar { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 16px; align-items: flex-end; }
          .fe-toolbar > label { flex: 1 1 180px; min-width: 0; }
          .fe-bars-wrap { margin: 18px 0; }
          .fe-bar-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; font-size: 14.5px; }
          .fe-bar-label { width: 110px; font-weight: 600; text-align: right; }
          .fe-bar-track { flex: 1; height: 26px; background: #edf2f4; border-radius: 4px; overflow: hidden; position: relative; }
          .fe-bar-fill { height: 100%; border-radius: 4px; transition: width 0.2s; }
          .fe-bar-value { width: 75px; font-family: Consolas, monospace; font-weight: 600; text-align: right; }
          .fe-math-banner { background: #fcf6f1; border-left: 4px solid #be7040; padding: 12px 16px; border-radius: 4px; font-family: Consolas, monospace; font-size: 15px; margin: 14px 0; }
          .fe-math-banner strong { color: #be7040; }
          .fe-btn-row { display: flex; gap: 10px; margin: 12px 0; flex-wrap: wrap; }
          .fe-pill-btn { padding: 6px 12px; font-size: 13px; font-weight: 600; border-radius: 20px; border: 1px solid #dce2e1; background: #fff; cursor: pointer; color: #2c647d; }
          .fe-pill-btn:hover { background: #eef5f8; border-color: #2c647d; }
          .fe-callout-warning { background: #fff6f5; border-left: 4px solid #aa4740; padding: 12px 16px; border-radius: 4px; margin: 14px 0; font-size: 14px; }
        </style>
        <div class="fe-tabs">
          <button type="button" class="fe-tab-btn" data-mode="softmax">Softmax + 交叉熵</button>
          <button type="button" class="fe-tab-btn" data-mode="sigmoid">Sigmoid + 平方误差</button>
        </div>
        <div class="fe-toolbar">
          ${ctx.range('输出原始分数 z₁ (logit)', 'fe-sm-z1', -4, 4, 0.1, 1.2)}
          ${ctx.range('输出原始分数 z₂ (logit)', 'fe-sm-z2', -4, 4, 0.1, -0.3)}
          ${ctx.range('温度参数 T (仅 Softmax)', 'fe-sm-temp', 0.2, 3, 0.1, 1.0)}
          ${ctx.select('真实标签类别', 'fe-sm-target', [['0', '类别 1 (d = [1, 0])'], ['1', '类别 2 (d = [0, 1])']], '0')}
        </div>
        <div class="fe-btn-row">
          <button type="button" class="fe-pill-btn" id="btn-demo-error">设置错误预测：z₁ = −3.8，z₂ = 3.5</button>
          <button type="button" class="fe-pill-btn" id="btn-demo-balanced">设置相同分数：z₁ = z₂ = 0</button>
        </div>
        <div id="fe-sm-content"></div>
      </div>
    `;

    function render() {
      // Toggle mode tabs
      root.querySelectorAll('.fe-tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.mode === currentMode);
      });

      var z1 = +root.querySelector('#fe-sm-z1').value;
      var z2 = +root.querySelector('#fe-sm-z2').value;
      var T = +root.querySelector('#fe-sm-temp').value;
      var targetIdx = +root.querySelector('#fe-sm-target').value;
      var d = targetIdx === 0 ? [1, 0] : [0, 1];

      var isSoftmax = currentMode === 'softmax';
      var p, loss, grad;

      if (isSoftmax) {
        // Softmax with temperature
        var softmaxResult = M.softmaxLossGrad([z1, z2], d, T);
        p = softmaxResult.p;
        loss = softmaxResult.loss;
        grad = softmaxResult.grad;
      } else {
        // Independent Sigmoid + MSE
        var sig = function(x) { return 1 / (1 + Math.exp(-x)); };
        p = [sig(z1), sig(z2)];
        loss = 0.5 * (Math.pow(p[0] - d[0], 2) + Math.pow(p[1] - d[1], 2));
        // Gradient: (y - d) * y * (1 - y)
        grad = [
          (p[0] - d[0]) * p[0] * (1 - p[0]),
          (p[1] - d[1]) * p[1] * (1 - p[1])
        ];
      }

      var container = root.querySelector('#fe-sm-content');
      var correctP = p[targetIdx];
      var correctGrad = grad[targetIdx];

      container.innerHTML = `
        <div class="fe-card">
          <h3>当前输出与分数梯度</h3>
          <div class="fe-bars-wrap">
            <div class="fe-bar-row">
              <div class="fe-bar-label">类别 1 ${targetIdx===0?'(真实标签✓)':''}</div>
              <div class="fe-bar-track">
                <div class="fe-bar-fill" style="width:${p[0]*100}%;background:${targetIdx===0?ORANGE:BLUE}"></div>
              </div>
              <div class="fe-bar-value">${(p[0]*100).toFixed(1)}%</div>
            </div>
            <div class="fe-bar-row">
              <div class="fe-bar-label">类别 2 ${targetIdx===1?'(真实标签✓)':''}</div>
              <div class="fe-bar-track">
                <div class="fe-bar-fill" style="width:${p[1]*100}%;background:${targetIdx===1?ORANGE:BLUE}"></div>
              </div>
              <div class="fe-bar-value">${(p[1]*100).toFixed(1)}%</div>
            </div>
          </div>

          <div class="fe-math-banner">
            ${isSoftmax ? `
              <strong>当前配置：Softmax + 交叉熵 (Cross-Entropy)</strong><br>
              预测概率之和：p₁ + p₂ = ${(p[0]+p[1]).toFixed(4)}（Softmax 将概率和归一化为 1）<br>
              交叉熵损失 L = −log(p真实) = <strong>${F(loss, 5)}</strong><br>
              输出层梯度公式：∂L/∂zₖ = (pₖ − dₖ) / T<br>
              分数梯度向量：δʸ = ∂L/∂z = [ <strong>${F(grad[0], 5)}</strong>, <strong>${F(grad[1], 5)}</strong> ]
            ` : `
              <strong>当前配置：Sigmoid + 平方误差</strong><br>
              预测输出之和：y₁ + y₂ = ${(p[0]+p[1]).toFixed(4)}（各分数分别经过 Sigmoid，输出和不保证为 1）<br>
              平方误差损失 L = ½ Σ(yₖ − dₖ)² = <strong>${F(loss, 5)}</strong><br>
              输出层梯度公式：∂L/∂zₖ = (yₖ − dₖ) · yₖ(1 − yₖ)<br>
              分数梯度向量：δʸ = ∂L/∂z = [ <strong>${F(grad[0], 5)}</strong>, <strong>${F(grad[1], 5)}</strong> ]
            `}
          </div>

          ${isSoftmax ? `
            <div style="background:#eef6fa;border-left:4px solid #2c647d;padding:12px 16px;border-radius:4px;font-size:14px;margin-top:12px">
              <strong>真实类别概率偏低时，交叉熵怎样改变分数。</strong><br>
              输出分数的梯度为 <code>∂L/∂z = (p − d) / T</code>。在 T=1、真实类别概率为 .01 时，该类别的梯度是 <code>.01 − 1 = −.99</code>，梯度下降会增大它的分数。其他类别的正梯度则使其分数降低。继续向前层回传时，还要乘相应的局部导数与权重；实际参数更新也取决于学习率。
            </div>
          ` : `
            <div class="fe-callout-warning">
              <strong>平方误差回传时，还会乘 Sigmoid 的导数。</strong><br>
              对目标 dₖ=1，若输出 yₖ 接近 0，差值 yₖ−dₖ 接近 −1，但 <code>yₖ(1−yₖ)</code> 也接近 0。二者相乘后，传到分数 zₖ 的梯度变小。将温度设为 1，使用上方的错误预测按钮，再切换两种模式，对照同一组原始分数收到的梯度。
            </div>
          `}
        </div>

        <div class="fe-card">
          <h3>将输出导数接回矩阵回传</h3>
          <p>
            输出激活与损失共同决定 ∂L/∂z。得到这个导数以后，仿射层仍按来源值乘去向敏感度计算权重梯度，向隐藏层回传时仍需乘输出权重并累加贡献。
          </p>
          <p>
            以 Softmax 模式为例：B 个样本按行排列，P 存预测概率，D 存目标向量，每行目标和为 1。批量损失取平均 L=ΣₙLₙ/B。沿用前页的定义，Δ 每行先存单样本导数，平均因子在计算权重梯度时加入：
          </p>
          <div class="fe-math-banner">
            Δ<sub>out</sub> = (P − D) / T，形状 (B×2)<br>
            ∇<sub>V</sub>L = Hᵀ Δ<sub>out</sub> / B<br>
            Δ<sub>hidden</sub> = (Δ<sub>out</sub> Vᵀ) ⊙ σ′(A₁)<br>
            ∇<sub>W</sub>L = Xᵀ Δ<sub>hidden</sub> / B
          </div>
          <p>这里隐藏层仍用 Sigmoid。Δ<sub>out</sub> 先经 Vᵀ 累加到隐藏输出，再乘隐藏激活导数，得到激活前的 Δ<sub>hidden</sub>。两次权重梯度各对自己的样本贡献取一次平均。</p>
        </div>
      `;
    }

    // Tab buttons click handler
    root.querySelectorAll('.fe-tab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        currentMode = btn.dataset.mode;
        render();
      });
    });

    // Preset buttons
    root.querySelector('#btn-demo-error').addEventListener('click', function() {
      root.querySelector('#fe-sm-target').value = '0'; // want class 1
      root.querySelector('#fe-sm-z1').value = '-3.8'; // but predicted very negative
      root.querySelector('#fe-sm-z2').value = '3.5';
      ctx.outputs();
      render();
    });

    root.querySelector('#btn-demo-balanced').addEventListener('click', function() {
      root.querySelector('#fe-sm-z1').value = '0';
      root.querySelector('#fe-sm-z2').value = '0';
      ctx.outputs();
      render();
    });

    ctx.controls(render);
    render();
  }

  window.FoundationExtraLabs = Object.assign(window.FoundationExtraLabs || {}, {
    'foundation-batch': batchLab,
    'foundation-softmax': softmaxLab
  });
}());
