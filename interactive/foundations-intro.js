/* First encounter with the persistent 2–2–2 network: identities and scalar work. */
(function () {
  'use strict';
  var serial = 0, BLUE='#28627c', ORANGE=BLUE, GRAY='#b7c5ca';
  function sub(n){return n===1||n==='1'?'₁':'₂';}
  function weightName(layer,i,j){return (layer===0?'w':'v')+sub(i)+sub(j);}
  function esc(x){return String(x).replace(/[&<>"']/g,function(c){return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]);});}
  function css(){return '<style>.fi-lab{font-size:16px;line-height:1.8}.fi-lab .control,.fi-lab input,.fi-lab select{font-size:15px}.fi-controls{display:flex;flex-wrap:wrap;gap:15px;align-items:end;margin:15px 0}.fi-controls>label{flex:1 1 180px}.fi-card{border:1px solid #dce4e6;background:#fff;border-radius:9px;padding:16px;margin:16px 0}.fi-card h3{font-size:18px;margin:0 0 10px}.fi-card p{margin:8px 0}.fi-note{border-left:3px solid '+ORANGE+';background:#eff6f9;padding:13px 16px;margin:14px 0}.fi-eq{font-family:Consolas,"Microsoft YaHei",monospace;background:#f0f5f7;border-radius:6px;padding:13px;font-size:17px;overflow-wrap:anywhere}.fi-lab button{font-size:15px;padding:9px 13px;border:1px solid #bdcbd1;background:#fff;border-radius:6px;color:'+BLUE+'}.fi-lab button:disabled{opacity:.4}.fi-steps{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}.fi-steps button[aria-current=step]{background:'+BLUE+';color:white;border-color:'+BLUE+'}.fi-lab [data-fc-item]{cursor:pointer}.fi-lab [data-fc-item]:focus-visible{outline:3px solid '+ORANGE+';outline-offset:4px}.fi-scroll{overflow-x:auto}.fi-zoom{display:block;width:100%;min-width:700px;height:auto;max-height:none!important}.fi-zoom text{font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:17px;fill:#23343e}.fi-zoom .fi-heading{font-size:19px;font-weight:650}.fi-nav{display:flex;align-items:center;gap:12px;margin:16px 0}.fi-pending{color:#697b83}.fi-key{color:'+ORANGE+';font-weight:650}.fi-node-detail{min-height:155px}.fi-identity{display:inline-block;padding:3px 9px;border-radius:5px;background:#edf3f5;color:'+BLUE+';font-size:15px}.fi-layers{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}.fi-layers span{padding:8px 13px;background:#edf3f5;border-radius:6px}.fi-stage-copy{min-height:90px}</style>';}
  function netHTML(shared,g,options){return shared.networkHTML(g,options||{});}
  function bindItems(container,callback){container.querySelectorAll('[data-fc-item]').forEach(function(el){
    el.setAttribute('tabindex','0');el.setAttribute('role','button');
    el.addEventListener('click',function(e){e.stopPropagation();callback(el.getAttribute('data-fc-item'));});
    el.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();callback(el.getAttribute('data-fc-item'));}});
  });}
  function identity(id,g,F){
    var n=g.byId[id];if(!n)return {title:id,role:'图中元素',body:'选择另一个节点或连线查看它的身份。'};
    var m;
    if(/^x[12]$/.test(id))return {title:'输入 '+n.label,role:'给定数据',body:'这是第 0 层（l = 0）的一个输入分量。数值 '+F(n.value)+' 由当前样本给定。x₁、x₂ 是两个特征；下标 1、2 表示特征位置。训练要调整连线上的权重，本节先把输入看作已知量。'};
    if((m=/^w([01])_([12])([12])$/.exec(id))){var l=+m[1],i=m[2],j=m[3],short=weightName(l,i,j);return {title:'可学习权重 '+short,role:'网络参数',body:'这条连线从'+(l===0?'输入 x'+sub(i):'隐藏输出 h'+sub(i))+' 指向'+(l===0?'隐藏单元 h'+sub(j):'输出单元 y'+sub(j))+'，权重为 '+F(n.value)+'。本例简写为 '+short+'，完整记法为 w<sub>'+(l===0?'ij':'jk')+'</sub><sup>l='+l+'</sup>（'+(l===0?'i':'j')+'='+i+'，'+(l===0?'j':'k')+'='+j+'）。这里的上标 l='+l+' 标记层号；下标第一个 '+i+' 是来源，第二个 '+j+' 是去向。代码中对应 '+(l===0?'W0':'W1')+'['+(+i-1)+']['+(+j-1)+']。'};}
    if((m=/^h([12])$/.exec(id)))return {title:'隐藏层输出 h'+sub(m[1]),role:'算出来的中间量',body:'这是第 1 层（l = 1）的第 '+m[1]+' 个输出。先把两条入边上的“输入 × 权重”相加，得到 a'+sub(m[1])+'，再通过 Sigmoid，得到 h'+sub(m[1])+' = s(a'+sub(m[1])+')。本例 hⱼ 也可写为 x<sub>j</sub><sup>l=1</sup>；这里 j='+m[1]+'。这里的上标 l=1 标记层号；下标是这一层中的位置。当前 h'+sub(m[1])+' = '+F(n.value)+'。下一节会把这一个圆点拆开计算。'};
    if((m=/^y([12])$/.exec(id)))return {title:'网络输出 y'+sub(m[1]),role:'算出来的预测',body:'这是第 2 层（l = 2）的第 '+m[1]+' 个输出，记作 y'+sub(m[1])+'。它以两个隐藏输出为输入，先加权求和得到 z'+sub(m[1])+'，再经过 Sigmoid：y'+sub(m[1])+' = s(z'+sub(m[1])+')。当前 y'+sub(m[1])+' = '+F(n.value)+'。y 是网络算出的预测；旁边的 d 是希望它达到的目标。'};
    if((m=/^d([12])$/.exec(id)))return {title:'目标 d'+m[1],role:'样本标签／监督信号',body:'d'+m[1]+' = '+F(n.value)+' 是当前样本附带的正确目标。它只在网络已给出 y'+m[1]+' 后参与误差和损失计算，<strong>不会作为输入送进隐藏层</strong>。例如目标 [1, 0] 表示希望第一个输出接近 1、第二个接近 0。'};
    if(id==='L'||/^loss[12]$/.test(id))return {title:id==='L'?'总损失 L':'单个输出的损失 '+n.label,role:'衡量预测偏差的一个数',body:'把预测与目标的差距汇总成一个数，便于评价这次预测。当前 '+n.label+' = '+F(n.value,6)+'。连线展示它位于预测与标签之后；先认识位置，后面再逐步计算。'};
    if((m=/^a([12])_([12])$/.exec(id)))return {title:'加权和 '+(m[1]==='1'?'a':'z')+sub(m[2]),role:'激活之前的中间量',body:'这是第 '+m[1]+' 层第 '+m[2]+' 个单元的加权和，本例记作 '+(m[1]==='1'?'a':'z')+sub(m[2])+'。它把本单元的两项“来源 × 权重”相加。当前值为 '+F(n.value)+'；再通过 Sigmoid 才得到'+(m[1]==='1'?'隐藏输出 h':'预测 y')+sub(m[2])+'。'};
    return {title:n.label,role:n.role==='constant'?'固定常数':'一次标量运算的中间量',body:'此节点执行 '+esc(n.expression)+'，结果为 '+F(n.value,6)+'。它由连进来的已知量计算得到。前几节会逐个展开这些运算。'};
  }

  function mapLab(ctx,shared){
    var root=ctx.root,F=ctx.fmt,selected='x1',s=shared.state;
    var choices=[['x1','输入 x₁'],['x2','输入 x₂'],['h1','隐藏输出 h₁'],['h2','隐藏输出 h₂'],['y1','预测 y₁'],['y2','预测 y₂'],['d1','目标 d₁'],['d2','目标 d₂'],['L','总损失 L']];
    ['w0_11','w0_12','w0_21','w0_22','w1_11','w1_12','w1_21','w1_22'].forEach(function(id){var m=/^w([01])_([12])([12])$/.exec(id);choices.push([id,'权重 '+weightName(+m[1],m[2],m[3])]);});
    root.innerHTML=css()+`<section class="fi-lab fi-discovery">
      <div class="fi-discovery-head">
        <div><span class="fi-kicker">第一个实验 · LIVE</span><h2>一条连接，能改变什么？</h2>
          <p>先猜：如果输入是 0，改变这条连接的权重，输出还会变化吗？</p></div>
        <div class="fi-status"><span aria-hidden="true"></span> 本地实时计算</div>
      </div>
      <div class="fi-discovery-controls">
        ${ctx.range('输入 x₁','fi-map-x1',-2,2,.05,s.x[0])}
        ${ctx.range('权重 w₁₁','fi-map-w11',-3,3,.05,s.W0[0][0])}
      </div>
      <div class="fi-preset" role="group" aria-label="输入预设">
        <span>试试两种输入</span>
        <button type="button" id="fi-input-zero">x₁ = 0</button>
        <button type="button" id="fi-input-nonzero">x₁ = 0.6</button>
        <button type="button" id="fi-map-reset">恢复初始样例</button>
      </div>
      <div class="fi-observation" id="fi-observation"></div>
      <div class="fi-chain" id="fi-chain" role="group" aria-label="当前数值的计算路径"></div>
      <p class="fi-caption">本例没有偏置；隐藏输出 h₁ 与 h₂ 一起决定 y₁。图中的所有数值都由当前网络重新计算。</p>
      <details class="fi-map-details"><summary>展开完整网络：点击节点和连线，查看它们的身份</summary>
        <div id="fi-map-network"></div>
        <div class="fc-controls fi-controls">${ctx.select('选择图中元素','fi-map-item',choices,selected)}</div>
        <div class="fc-card fi-card fi-node-detail" id="fi-map-detail" aria-live="polite"></div>
      </details>
      <p class="fi-next-link">想亲手算清隐藏节点？<a href="#chain">下一节：乘法、求和与激活 →</a></p>
    </section>`;
    function choose(id){if(!shared.graph().byId[id])return;selected=id;var sel=root.querySelector('#fi-map-item');if(!choices.some(function(c){return c[0]===id;})){var opt=document.createElement('option');opt.value=id;opt.textContent=shared.graph().byId[id].label;sel.appendChild(opt);}sel.value=id;draw();}
    function draw(){
      var g=shared.graph(),r=g.result,detail=identity(selected,g,F),p1=s.x[0]*s.W0[0][0],p2=s.x[1]*s.W0[1][0];
      root.querySelector('#fi-observation').innerHTML=s.x[0]===0
        ? '<strong>现在试着拖动 w₁₁。</strong> x₁ = 0，所以 x₁w₁₁ 始终为 0；其他量不变时，h₁ 和 y₁ 都不会改变。'
        : '<strong>现在试着拖动 w₁₁。</strong> x₁ 不为 0，它与权重的乘积会改变；沿计算路径观察 h₁ 和 y₁。';
      root.querySelector('#fi-chain').innerHTML=
        '<div class="fi-chain-step"><span>① 当前乘积</span><strong>'+F(s.x[0],3)+' × '+F(s.W0[0][0],3)+' = '+F(p1,5)+'</strong><small>x₁ × w₁₁</small></div>'+
        '<div class="fi-chain-step"><span>② 隐藏层</span><strong>h₁ = '+F(r.h[0],6)+'</strong><small>Sigmoid('+F(p1,4)+' + '+F(p2,4)+')</small></div>'+
        '<div class="fi-chain-step"><span>③ 网络预测</span><strong>y₁ = '+F(r.y[0],6)+'</strong><small>由 h₁ 和 h₂ 共同计算</small></div>'+
        '<div class="fi-chain-step"><span>④ 与目标比较</span><strong>L = '+F(r.loss,6)+'</strong><small>两项 ½(预测−目标)² 之和</small></div>';
      root.querySelector('#fi-map-network').innerHTML=netHTML(shared,g,{known:null,selected:selected,forwardEdges:[],reverseEdges:[]});
      root.querySelector('#fi-map-detail').innerHTML='<span class="fi-identity">'+detail.role+'</span><h3>'+esc(detail.title)+'</h3><p>'+detail.body+'</p>';
      bindItems(root.querySelector('#fi-map-network'),choose);
    }
    function sync(){root.querySelector('#fi-map-x1').value=s.x[0];root.querySelector('#fi-map-w11').value=s.W0[0][0];ctx.outputs();}
    root.querySelector('#fi-input-zero').addEventListener('click',function(){s.x[0]=0;sync();draw();});
    root.querySelector('#fi-input-nonzero').addEventListener('click',function(){s.x[0]=.6;sync();draw();});
    root.querySelector('#fi-map-reset').addEventListener('click',function(){shared.reset();sync();draw();});
    ctx.controls(function(el){
      if(el.id==='fi-map-x1')s.x[0]=+el.value;
      else if(el.id==='fi-map-w11')s.W0[0][0]=+el.value;
      else if(el.id==='fi-map-item')selected=el.value;
      draw();
    });
    sync();draw();
  }

  function svgText(x,y,t,color,cls){return '<text x="'+x+'" y="'+y+'" text-anchor="middle"'+(color?' style="fill:'+color+'"':'')+(cls?' class="'+cls+'"':'')+'>'+esc(t)+'</text>';}
  function box(x,y,w,h,lines,active,pending){var out='<rect x="'+(x-w/2)+'" y="'+(y-h/2)+'" width="'+w+'" height="'+h+'" rx="8" fill="'+(active?'#e6f2f7':pending?'#f4f5f5':'#f0f6f8')+'" stroke="'+(active?ORANGE:GRAY)+'" stroke-width="'+(active?2.5:1.4)+'"/>';lines.forEach(function(t,i){out+=svgText(x,y+(i-(lines.length-1)/2)*22+6,t,active?ORANGE:pending?'#7c8b90':BLUE);});return out;}
  function arrow(id,x1,y1,x2,y2,active){return '<path d="M '+x1+' '+y1+' L '+x2+' '+y2+'" fill="none" stroke="'+(active?ORANGE:GRAY)+'" stroke-width="'+(active?2.8:1.5)+'" marker-end="url(#'+id+')"/>';}
  function circle(x,y,op,active){return '<circle cx="'+x+'" cy="'+y+'" r="25" fill="'+(active?'#e6f2f7':'white')+'" stroke="'+(active?ORANGE:GRAY)+'" stroke-width="2"/>'+svgText(x,y+6,op,active?ORANGE:BLUE);}
  function zoom(g,j,step,F,uid){
    var s=g.state,which=j+1,p1=s.x[0]*s.W0[0][j],p2=s.x[1]*s.W0[1][j],a=g.result.a1[j],h=g.result.h[j];
    var out='<div class="fi-scroll"><svg class="fi-zoom" style="min-width:900px" viewBox="0 0 1040 350" role="img" aria-label="从左到右展开两个乘法、求和a、Sigmoid与隐藏输出"><defs><marker id="'+uid+'" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="context-stroke"/></marker></defs>';
    out+=svgText(520,26,'放大 h'+sub(which)+'：输入与权重 → 两项乘积 → 加权和 → 激活后的输出',BLUE,'fi-heading');
    out+=arrow(uid,136,78,183,108,step===1)+arrow(uid,136,148,183,116,step===1)+arrow(uid,236,113,269,113,step===1);
    out+=arrow(uid,136,225,183,255,step===1)+arrow(uid,136,295,183,263,step===1)+arrow(uid,236,260,269,260,step===1);
    out+=arrow(uid,387,113,464,173,step===2)+arrow(uid,387,260,464,198,step===2)+arrow(uid,510,186,526,186,step===2);
    out+=arrow(uid,683,186,709,186,step===3)+arrow(uid,891,186,925,186,step===3);
    out+=box(76,78,120,48,['x₁ = '+F(s.x[0],3)],false,false)+box(76,148,120,48,['w₁'+sub(which)+' = '+F(s.W0[0][j],3)],false,false);
    out+=box(76,225,120,48,['x₂ = '+F(s.x[1],3)],false,false)+box(76,295,120,48,['w₂'+sub(which)+' = '+F(s.W0[1][j],3)],false,false);
    out+=circle(211,113,'×',step===1)+circle(211,260,'×',step===1);
    out+=box(329,113,116,62,['p₁ = x₁w₁'+sub(which),step>=1?F(p1,5):'尚未计算'],step===1,step<1);
    out+=box(329,260,116,62,['p₂ = x₂w₂'+sub(which),step>=1?F(p2,5):'尚未计算'],step===1,step<1);
    out+=circle(485,186,'+',step===2);
    out+=box(605,186,155,66,['a'+sub(which)+' = p₁ + p₂',step>=2?F(a,5):'尚未计算'],step===2,step<2);
    out+=box(800,186,180,66,['Sigmoid：s(a)','1 / (1 + exp(−a))'],step===3,step<3);
    out+=box(980,186,106,66,['h'+sub(which),step>=3?F(h,6):'尚未计算'],step===3,step<3);
    out+=svgText(605,266,'本例 b = 0，求和不加额外项。',BLUE)+svgText(800,302,'a'+sub(which)+' 与 h'+sub(which)+' 是激活前后的不同量。',BLUE);
    return out+'</svg></div>';
  }
  function neuronLab(ctx,shared){
    var root=ctx.root,F=ctx.fmt,j=0,step=0,uid='fi-zoom-'+(++serial),s=shared.state;
    var settings='<div class="fc-controls fi-controls">'+ctx.select('放大哪个隐藏单元','fi-neuron-choice',[['0','h₁：接收 w₁₁、w₂₁'],['1','h₂：接收 w₁₂、w₂₂']],'0')+ctx.range('给定输入 x₁','fi-neuron-x1',-2,2,'any',s.x[0])+ctx.range('给定输入 x₂','fi-neuron-x2',-2,2,'any',s.x[1])+ctx.range('从 x₁ 进入本单元的权重','fi-neuron-w1',-3,3,'any',s.W0[0][0])+ctx.range('从 x₂ 进入本单元的权重','fi-neuron-w2',-3,3,'any',s.W0[1][0])+'</div>';
    var steps='<div class="fi-steps" id="fi-neuron-steps"><button type="button" data-fi-step="0">0 · 给定量</button><button type="button" data-fi-step="1">1 · 两个乘积</button><button type="button" data-fi-step="2">2 · 求和</button><button type="button" data-fi-step="3">3 · Sigmoid</button></div>';
    var navigation='<div class="fi-nav"><button type="button" id="fi-neuron-prev">← 上一步</button><span id="fi-neuron-counter"></span><button type="button" id="fi-neuron-next">下一步 →</button><button type="button" id="fi-neuron-reset">恢复全部初始数值</button></div>';
    root.innerHTML=css()+'<section class="fi-lab"><div id="fi-neuron-zoom"></div>'+steps+navigation+'<div class="fc-card fi-card fi-stage-copy" id="fi-neuron-copy" aria-live="polite"></div>'+settings+'<div class="fc-step-note fi-note">偏置在本例中固定为 0。输入或权重一改，本节回到第 0 步，重新计算。当前修改会保留到后面的课程；h₁、h₂ 的后继输出 y₁、y₂ 暂时不计算，留到下一节继续。</div><details class="fi-card"><summary>查看这个单元在完整网络中的位置</summary><div id="fi-neuron-network"></div></details></section>';
    function sync(){root.querySelector('#fi-neuron-x1').value=s.x[0];root.querySelector('#fi-neuron-x2').value=s.x[1];root.querySelector('#fi-neuron-w1').value=s.W0[0][j];root.querySelector('#fi-neuron-w2').value=s.W0[1][j];if(ctx.outputs)ctx.outputs();}
    function draw(){var g=shared.graph(),which=j+1,known=new Set(g.nodes.filter(function(n){return n.role==='input'||n.role==='weight'||n.role==='target'||n.role==='constant';}).map(function(n){return n.id;}));
      if(step>=1){known.add('p0_1'+which);known.add('p0_2'+which);}if(step>=2)known.add('a1_'+which);if(step>=3)known.add('h'+which);
      var incoming=['w0_1'+which,'w0_2'+which],highlight=step>=3?incoming.concat(['w1_'+which+'1','w1_'+which+'2']):incoming;
      root.querySelector('#fi-neuron-zoom').innerHTML=zoom(g,j,step,F,uid);
      root.querySelector('#fi-neuron-network').innerHTML=netHTML(shared,g,{known:known,selected:'h'+which,forwardEdges:highlight,reverseEdges:[]});
      var p1=s.x[0]*s.W0[0][j],p2=s.x[1]*s.W0[1][j],a=g.result.a1[j],h=g.result.h[j],copy;
      if(step===0)copy='<h3>第 0 步：先分清给定量</h3><p>两项输入为 x₁ = '+F(s.x[0])+'、x₂ = '+F(s.x[1])+'；进入 h'+sub(which)+' 的权重为 w₁'+sub(which)+' = '+F(s.W0[0][j])+'、w₂'+sub(which)+' = '+F(s.W0[1][j])+'。还没有进行乘法，右侧结果均为“尚未计算”。</p>';
      if(step===1)copy='<h3>第 1 步：两个乘法，各算各的</h3><div class="fi-eq">p₁ = x₁ w₁'+sub(which)+' = '+F(s.x[0])+' × '+F(s.W0[0][j])+' = '+F(p1,6)+'<br>p₂ = x₂ w₂'+sub(which)+' = '+F(s.x[1])+' × '+F(s.W0[1][j])+' = '+F(p2,6)+'</div>';
      if(step===2)copy='<h3>第 2 步：只把刚才的两项相加</h3><div class="fi-eq">a'+sub(which)+' = p₁ + p₂ = '+F(p1,6)+' + ('+F(p2,6)+') = '+F(a,6)+'</div><p>a'+sub(which)+' 是隐藏层第 '+which+' 个单元的加权和；下标 '+which+' 标记该层中的位置。这里还没有通过 Sigmoid。</p>';
      if(step===3)copy='<h3>第 3 步：加权和送进 Sigmoid</h3><div class="fi-eq">h'+sub(which)+' = s(a'+sub(which)+') = 1 / [1 + exp(−('+F(a,6)+'))] = <span class="fi-key">'+F(h,6)+'</span></div><p>a'+sub(which)+' 是激活前的加权和，h'+sub(which)+' 是激活后的输出。它会沿图中的两条出边进入 y₁、y₂，但这两个输出尚未计算。</p>';
      root.querySelector('#fi-neuron-copy').innerHTML=copy;
      root.querySelector('#fi-neuron-counter').textContent=step+' / 3';root.querySelector('#fi-neuron-prev').disabled=step===0;root.querySelector('#fi-neuron-next').disabled=step===3;
      root.querySelectorAll('[data-fi-step]').forEach(function(b){b.setAttribute('aria-current',+b.getAttribute('data-fi-step')===step?'step':'false');});
      bindItems(root.querySelector('#fi-neuron-network'),function(id){if(id==='h1'||id==='h2'){j=+id[1]-1;step=0;root.querySelector('#fi-neuron-choice').value=String(j);sync();draw();}});
    }
    root.querySelectorAll('[data-fi-step]').forEach(function(b){b.addEventListener('click',function(){step=+b.getAttribute('data-fi-step');draw();});});
    root.querySelector('#fi-neuron-prev').addEventListener('click',function(){step=Math.max(0,step-1);draw();});root.querySelector('#fi-neuron-next').addEventListener('click',function(){step=Math.min(3,step+1);draw();});
    root.querySelector('#fi-neuron-reset').addEventListener('click',function(){shared.reset();step=0;sync();draw();});
    ctx.controls(function(el){if(el&&el.id==='fi-neuron-choice'){j=+el.value;step=0;sync();draw();return;}
      if(el){if(el.id==='fi-neuron-x1')s.x[0]=+el.value;if(el.id==='fi-neuron-x2')s.x[1]=+el.value;if(el.id==='fi-neuron-w1')s.W0[0][j]=+el.value;if(el.id==='fi-neuron-w2')s.W0[1][j]=+el.value;}step=0;draw();});sync();draw();
  }

  window.FoundationIntroLabs=Object.assign(window.FoundationIntroLabs||{},
    {'foundation-map':mapLab,'foundation-neuron':neuronLab});
}());
