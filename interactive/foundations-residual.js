/* A scalar residual block: cached forward values and four explicit reverse steps. */
(function () {
  'use strict';

  window.FoundationResidualLab = function (ctx, lesson) {
    const { root, $, range, select, controls, outputs, value, fmt } = ctx;
    let stage = 0;
    let presetMessage = '';
    const blue = '#245c78', orange = '#a64f23', dark = '#22353c', muted = '#62747a';
    const signed = n => (n > 0 ? '+' : '') + fmt(n, 5);

    root.innerHTML = `<div class="fc-residual-shell">
      <style>
        .fc-residual-shell{font-size:15px;line-height:1.75;color:${dark}}
        .fc-residual-shell .control,.fc-residual-shell .control-top,
        .fc-residual-shell select,.fc-residual-shell button,
        .fc-residual-shell .lab-description,.fc-residual-shell .lab-note{font-size:14px;line-height:1.65}
        .fc-residual-shell .control-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}
        .fc-residual-shell .fc-residual-intro{margin:0 0 18px;font-size:15px}
        .fc-residual-shell .fc-residual-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin:16px 0 8px}
        .fc-residual-shell .fc-residual-blue{color:${blue};font-weight:650}
        .fc-residual-shell .fc-residual-orange{color:${orange};font-weight:650}
        .fc-residual-shell .fc-residual-diagram{overflow:auto;border:1px solid #dae2e2;border-radius:8px;background:#fff}
        .fc-residual-shell .fc-residual-svg{display:block;width:100%;min-width:560px;height:auto;max-height:none}
        .fc-residual-shell .fc-residual-svg text{font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:16px}
        .fc-residual-shell .fc-residual-title{font-weight:650;fill:${dark}}
        .fc-residual-shell .fc-residual-value{fill:${blue}}
        .fc-residual-shell .fc-residual-grad{fill:${orange};font-weight:650}
        .fc-residual-shell .fc-residual-pending{fill:${muted};font-size:14px}
        .fc-residual-shell .fc-residual-edge-label{font-size:15px;font-weight:650;fill:${orange}}
        .fc-residual-shell .fc-residual-current{margin:16px 0;padding:15px;border-left:4px solid ${orange};background:#fcf1e9}
        .fc-residual-shell .fc-residual-current h3{font-size:17px;margin:0 0 7px}
        .fc-residual-shell .fc-residual-current p{margin:6px 0;font-size:15px}
        .fc-residual-shell .fc-residual-equation{font-family:Consolas,"Microsoft YaHei",monospace;font-size:16px;line-height:1.9;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:8px}
        .fc-residual-shell .fc-residual-step-count{font-size:15px;font-weight:650;color:${dark}}
        .fc-residual-shell .fc-residual-parameters{border:1px solid #d7e1de;border-radius:7px;padding:14px;margin-top:16px;background:#f1f6f3}
        .fc-residual-shell .fc-residual-parameters h3{font-size:16px;margin:0 0 8px}
        .fc-residual-shell .fc-residual-parameters p{font-size:15px;margin:6px 0}
        .fc-residual-shell .fc-residual-preset-note{font-size:15px;font-weight:600;margin:9px 0;color:${orange}}
        .fc-residual-shell .buttons{gap:10px}
      </style>
      <p class="fc-residual-intro">同一个输入 x 分成两路：捷径 κx，以及残差支路 F=ax+b。两路先相加，再选择是否经过 ReLU。给定后续损失传回的 g=∂L/∂y，逐步计算损失对每个上游量的敏感度。</p>
      <div class="control-grid">
        ${range('输入 x', 'r-x', -2, 2, .05, .8)}
        ${range('残差系数 a', 'r-a', -2, 2, .05, .1)}
        ${range('残差偏置 b', 'r-b', -2, 2, .05, 0)}
        ${range('捷径系数 κ', 'r-k', -2, 2, .05, 1)}
        ${range('末端梯度 g = ∂L/∂y', 'r-g', -2, 2, .05, 1)}
        ${select('加和后处理', 'r-post', [['identity', '无后激活：y=q'], ['relu', '后接 ReLU：y=max(0,q)']], 'identity')}
      </div>
      <div class="buttons">
        <button type="button" class="quiet" id="r-zero">F=0，保留输入</button>
        <button type="button" class="quiet" id="r-cancel">两路导数抵消</button>
        <button type="button" class="quiet" id="r-block">ReLU 截断</button>
      </div>
      <div id="r-preset-note" class="fc-residual-preset-note" aria-live="polite"></div>
      <div class="fc-residual-legend">
        <span class="fc-residual-blue">蓝色 →：前向算出的数值</span>
        <span class="fc-residual-orange">橙色 →：损失对上游量的敏感度</span>
      </div>
      <p class="lab-note">蓝色显示当前前向值，橙色显示损失的局部变化率。算出这些梯度之后，便可以按学习率更新参数。</p>
      <div id="r-view" class="fc-residual-diagram"></div>
      <div class="buttons">
        <button type="button" class="quiet" id="r-prev">← 上一步</button>
        <button type="button" class="primary" id="r-next">开始反向 →</button>
        <span class="fc-residual-step-count" id="r-step-count"></span>
      </div>
      <div id="r-current" class="fc-residual-current" aria-live="polite"></div>
      <div id="r-parameters"></div>
    </div>`;

    function arrow(x1, y1, x2, y2, color, width = 2.2) {
      const angle = Math.atan2(y2 - y1, x2 - x1), h = 8;
      return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>` +
        `<path d="M${x2 - h * Math.cos(angle - .5)},${y2 - h * Math.sin(angle - .5)} L${x2},${y2} L${x2 - h * Math.cos(angle + .5)},${y2 - h * Math.sin(angle + .5)}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
    }

    function label(x, y, str, width = 110) {
      return `<rect x="${x - width / 2}" y="${y - 17}" width="${width}" height="25" rx="4" fill="white" fill-opacity=".97"/><text x="${x}" y="${y}" text-anchor="middle" class="fc-residual-edge-label">${str}</text>`;
    }

    function box(x, y, width, title, forward, gradient, ready) {
      return `<rect x="${x - width / 2}" y="${y - 45}" width="${width}" height="90" rx="8" fill="#f6f9f8" stroke="${ready ? '#c68a68' : '#c5d1d3'}" stroke-width="1.5"/>` +
        `<text x="${x}" y="${y - 21}" text-anchor="middle" class="fc-residual-title">${title}</text>` +
        `<text x="${x}" y="${y + 3}" text-anchor="middle" class="fc-residual-value">${forward}</text>` +
        `<text x="${x}" y="${y + 29}" text-anchor="middle" class="${ready ? 'fc-residual-grad' : 'fc-residual-pending'}">${ready ? gradient : '敏感度待计算'}</text>`;
    }

    function calculate() {
      const x = value('r-x'), a = value('r-a'), b = value('r-b');
      const k = value('r-k'), g = value('r-g'), relu = $('#r-post').value === 'relu';
      const u = k * x, F = a * x + b, q = u + F;
      const y = relu ? Math.max(0, q) : q, slope = relu ? (q > 0 ? 1 : 0) : 1;
      const gq = g * slope, shortcut = k * gq, residual = a * gq;
      return { x, a, b, k, g, relu, u, F, q, y, slope, gq, shortcut, residual,
        gx: shortcut + residual, ga: x * gq, gb: gq };
    }

    function diagram(s) {
      let out = '';
      // Blue arrows always connect the cached forward values.
      out += arrow(122, 160, 194, 103, blue);
      out += arrow(346, 103, 433, 160, blue);
      out += arrow(122, 226, 194, 293, blue);
      out += arrow(346, 293, 433, 226, blue);
      out += arrow(468, 237, 468, 298, blue);
      out += arrow(468, 392, 468, 443, blue);

      // Orange arrows follow the reverse dependency, with their actual numbers.
      if (stage >= 1) {
        out += arrow(343, 519, 421, 499, orange, 3);
        out += label(318, 530, `g=${signed(s.g)}`, 128);
        out += arrow(510, 443, 510, 392, orange, 3);
        out += label(487, 429, signed(s.g), 88);
        out += arrow(510, 298, 510, 237, orange, 3);
        out += label(487, 275, signed(s.gq), 88);
      }
      if (stage >= 2) {
        out += arrow(434, 143, 346, 86, orange, 3);
        out += label(396, 106, `gq=${signed(s.gq)}`, 122);
        out += arrow(434, 243, 346, 310, orange, 3);
        out += label(393, 293, `gq=${signed(s.gq)}`, 122);
      }
      if (stage >= 3) {
        out += arrow(194, 120, 122, 177, orange, 3);
        out += label(121, 115, `κgq=${signed(s.shortcut)}`, 160);
        out += arrow(194, 310, 122, 243, orange, 3);
        out += label(124, 294, `agq=${signed(s.residual)}`, 160);
      }

      out += box(70, 197, 132, '输入 x', `x=${fmt(s.x)}`, `∂L/∂x=${signed(s.gx)}`, stage >= 4);
      out += box(270, 70, 154, '捷径 u=κx', `u=${fmt(s.u)}`, `∂L/∂u=${signed(s.gq)}`, stage >= 2);
      out += box(270, 338, 154, '残差 F=ax+b', `F=${fmt(s.F)}`, `∂L/∂F=${signed(s.gq)}`, stage >= 2);
      out += box(485, 197, 132, '加和 q=u+F', `q=${fmt(s.q)}`, `∂L/∂q=${signed(s.gq)}`, stage >= 1);
      out += box(485, 346, 140, s.relu ? 'ReLU(q)' : '恒等操作 q', `输出=${fmt(s.y)}`, `局部导数=${s.slope}`, stage >= 1);
      out += box(485, 494, 132, '最终输出 y', `y=${fmt(s.y)}`, `∂L/∂y=${signed(s.g)}`, stage >= 1);
      out += `<text x="21" y="414" class="fc-residual-value">前向：先分路，再相加，最后激活。</text>`;
      if (stage >= 2) out += `<text x="21" y="444" class="fc-residual-grad">加法：两边各收一份 gq，不平分。</text>`;
      if (stage >= 3) out += `<text x="21" y="474" class="fc-residual-grad">回到 x：先保留两份有符号贡献。</text>`;
      if (stage >= 4) out += `<text x="21" y="504" class="fc-residual-grad">共享输入处：两份贡献相加。</text>`;
      return `<svg class="fc-residual-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 556" role="img" aria-label="残差块的前向数值与四步反向传播">${out}</svg>`;
    }

    function currentStep(s) {
      const text = [
        ['0 · 前向已完成，开始前先读各个数',
          '蓝色箭头按输入到输出的顺序计算。接下来固定这次前向的数值，从后续损失传来的 g 开始反向求导。',
          `u = κx = ${fmt(s.k)} × ${fmt(s.x)} = ${fmt(s.u)}\nF = ax+b = ${fmt(s.a)} × ${fmt(s.x)} + (${fmt(s.b)}) = ${fmt(s.F)}\nq = u+F = ${fmt(s.q)}；y = ${fmt(s.y)}`],
        ['1 · 从 y 穿过最后一个操作，回到 q',
          s.relu ? '后接 ReLU 时，回传梯度要乘当前 q 处的 ReLU 导数。q 为负时，这一步就截断两条上游路径。' : '加和后没有激活，y=q，局部导数为 1。后续传来的 g 原样到达加和结果 q。',
          `gq = ∂L/∂q = (∂L/∂y) × (∂y/∂q)\n   = ${signed(s.g)} × ${s.slope} = ${signed(s.gq)}`],
        ['2 · 加法节点给两条支路各一份 gq',
          'q=u+F，对 u 和 F 的局部偏导各为 1。两个输入各收到同一个 gq；并不是每边分到一半。',
          `∂L/∂u = gq × 1 = ${signed(s.gq)}\n∂L/∂F = gq × 1 = ${signed(s.gq)}`],
        ['3 · 两条支路分别把敏感度传回自己的输入 x',
          '捷径 u=κx 的局部导数是 κ；残差 F=ax+b 对 x 的局部导数是 a。两路各乘各的导数，得到两份带正负号的贡献。',
          `捷径贡献 = (∂L/∂u) × κ = ${signed(s.gq)} × (${fmt(s.k)}) = ${signed(s.shortcut)}\n残差贡献 = (∂L/∂F) × a = ${signed(s.gq)} × (${fmt(s.a)}) = ${signed(s.residual)}\n下一步在共享的 x 处把它们相加。`],
        ['4 · 在共享输入 x 相加，再求参数梯度',
          '两条路径来自同一个 x，所以损失对 x 的总敏感度是两份贡献之和。它描述 x 的微小变化会怎样影响后续损失；这里没有修改输入。',
          `∂L/∂x = κgq + agq\n       = (${signed(s.shortcut)}) + (${signed(s.residual)}) = ${signed(s.gx)}\n       = g × ${s.slope} × (κ+a)`]
      ][stage];
      let note = '';
      if (s.relu && s.q === 0 && stage >= 1) note = '<p>q=0 是 ReLU 的不可导点。本实验与常见实现一样，在该点选用 0 作为回传导数。</p>';
      return `<h3>${text[0]}</h3><p>${text[1]}</p><div class="fc-residual-equation">${text[2]}</div>${note}`;
    }

    function draw() {
      const s = calculate();
      $('#r-view').innerHTML = diagram(s);
      $('#r-current').innerHTML = currentStep(s);
      $('#r-step-count').textContent = `反向 ${stage} / 4`;
      $('#r-prev').disabled = stage === 0;
      $('#r-next').disabled = stage === 4;
      $('#r-next').textContent = stage === 0 ? '开始反向 →' : stage === 4 ? '已完成四步' : '反向下一步 →';
      $('#r-preset-note').textContent = presetMessage;
      if (stage < 4) {
        $('#r-parameters').innerHTML = '';
        return;
      }
      let conclusion = '';
      if (s.a === 0 && s.b === 0 && s.k === 1 && !s.relu)
        conclusion = 'F=0 且 κ=1、加和后无激活时，前向 y=x，输入梯度也原样传回。';
      else if (s.k + s.a === 0 && s.gq !== 0)
        conclusion = '两条输入路径相互抵消，但 a、b 的参数梯度仍可能非零。输入导数为零，不等于所有参数都学不到东西。';
      else if (s.relu && s.q < 0)
        conclusion = '后置 ReLU 在这次前向处导数为零，两路都收到零。恒等捷径也会被加和后的激活截断。';
      else if (s.k !== 1)
        conclusion = 'κ≠1 时，捷径是缩放映射；回传要乘 κ，不能把它当作导数恒为 1 的恒等捷径。';
      $('#r-parameters').innerHTML = `<div class="fc-residual-parameters"><h3>同一个 F，还要把梯度分给参数 a 与 b</h3><p>F=ax+b：对 a 的局部偏导是 x，对 b 的局部偏导是 1。</p><div class="fc-residual-equation">∂L/∂a = (∂L/∂F) × x\n       = ${signed(s.gq)} × (${fmt(s.x)}) = ${signed(s.ga)}\n∂L/∂b = (∂L/∂F) × 1 = ${signed(s.gb)}</div><p>这两项才是 a、b 的学习信号。若随后执行梯度下降，参数变化量是 −学习率 × 对应梯度。</p>${conclusion ? `<p><strong>${conclusion}</strong></p>` : ''}</div>`;
    }

    controls(() => { presetMessage = ''; draw(); });
    $('#r-prev').onclick = () => { stage = Math.max(0, stage - 1); draw(); };
    $('#r-next').onclick = () => { stage = Math.min(4, stage + 1); draw(); };
    function preset(a, k, x, post, message) {
      for (const [id, v] of [['r-a', a], ['r-k', k], ['r-x', x], ['r-b', 0], ['r-g', 1]])
        $('#' + id).value = v;
      $('#r-post').value = post;
      stage = 4;
      presetMessage = message + ' 四步结果已全部展开；可用“上一步”逐段回看。';
      outputs();
      draw();
    }
    $('#r-zero').onclick = () => preset(0, 1, .8, 'identity', '恒等预设：F=0，κ=1，g=1。');
    $('#r-cancel').onclick = () => preset(-1, 1, .8, 'identity', '抵消预设：两路对 x 的贡献为 +1 与 −1。');
    $('#r-block').onclick = () => preset(.2, 1, -1, 'relu', '截断预设：q=−1.2，后置 ReLU 把回传梯度变为 0。');
    draw();
  };
}());
