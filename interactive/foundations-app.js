/* One persistent example: see the network, compute values, then carry sensitivities back. */
(function(){
'use strict';
const G=window.FoundationMath;
const initial={x:[.6,.2],W0:[[.1,-.2],[.4,.3]],W1:[[.2,-.3],[-.4,.1]],d:[1,0]};
const clone=s=>JSON.parse(JSON.stringify(s)),state=clone(initial);
const f=(x,n=5)=>!Number.isFinite(x)?String(x):(Math.abs(x)>0&&Math.abs(x)<1e-5?x.toExponential(3):Number(x.toFixed(n)).toString());
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sub=['₁','₂'];
const weightName=id=>(id[1]==='0'?'w':'v')+sub[+id[3]-1]+sub[+id[4]-1];
const fmtVec=a=>'['+a.map(x=>f(x)).join(', ')+']';
const val=(g,id)=>g.byId[id].value;
const sg=(x,y,s,cls='',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${esc(s)}</text>`;
const allKnown=new Set(['x1','x2','a1_1','a1_2','h1','h2','a2_1','a2_2','y1','y2','loss1','loss2','L']);
let renderSerial=0;

function networkHTML(g,opts={}){
 const known=opts.known==null?allKnown:opts.known,selected=opts.selected||'',fw=new Set(opts.forwardEdges||[]),rev=new Set(opts.reverseEdges||[]),grads=opts.gradients||{},wgrads=opts.weightGradients||{};
 const uid='fc-net-'+(++renderSerial),forward=uid+'-f',back=uid+'-b';
 const p={x1:[76,142],x2:[76,332],h1:[352,142],h2:[352,332],y1:[628,142],y2:[628,332],loss1:[824,142],loss2:[824,332],L:[976,237]};
 let content=`<defs><marker id="${forward}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker><marker id="${back}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#b66031"/></marker></defs>`;
 for(const [x,label,small]of[[76,'输入层','第 0 层 · l=0'],[352,'隐藏层','第 1 层 · l=1'],[628,'输出层','第 2 层 · l=2'],[890,'与标签比较','标签不是网络输入']])content+=sg(x,25,label,'fc-svg-head')+sg(x,48,small,'fc-svg-small');
 function edge(id,a,b,label,labelPos){
  let [x1,y1]=p[a],[x2,y2]=p[b],dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy),pad1=a.startsWith('loss')?54:37,pad2=b==='L'?39:b.startsWith('loss')?55:37;
  let ax=x1+pad1*dx/len,ay=y1+pad1*dy/len,bx=x2-pad2*dx/len,by=y2-pad2*dy/len,active=selected===id||fw.has(id),color=active?'#2c647d':'#bdc9cd';
  content+=`<g ${label?`data-fc-item="${id}" role="button" tabindex="0" aria-label="查看权重 ${label}"`:''}><path d="M${ax} ${ay}L${bx} ${by}" fill="none" stroke="${color}" stroke-width="${active?3:1.6}" marker-end="url(#${forward})"/>`;
  if(label){const [lx,ly]=labelPos;content+=`<path d="M${ax} ${ay}L${bx} ${by}" stroke="transparent" stroke-width="16"/><rect x="${lx-52}" y="${ly-18}" width="104" height="29" rx="5" fill="${selected===id?'#e3eef3':'#f7f8f6'}"/>`+sg(lx,ly+2,label,'fc-weight-label');}
  content+='</g>';
  if(rev.has(id)){content+=`<path d="M${bx} ${by+8}L${ax} ${ay+8}" fill="none" stroke="#b66031" stroke-width="3.4" marker-end="url(#${back})" class="fc-back-arrow"/>`;if(opts.reverseLabels?.[id]&&labelPos){let[lx,ly]=labelPos;content+=sg(lx,ly+(id[3]===id[4]?58:33),opts.reverseLabels[id],'fc-svg-gradient');}}
  if(wgrads[id]!==undefined){const[lx,ly]=labelPos;content+=sg(lx,ly+(id[3]===id[4]?58:29),'∂L/∂'+weightName(id)+'='+f(wgrads[id],4),'fc-svg-gradient');}
 }
 for(let l=0;l<2;l++)for(let i=0;i<2;i++)for(let j=0;j<2;j++){
  let id='w'+l+'_'+(i+1)+(j+1),a=(l?'h':'x')+(i+1),b=(l?'y':'h')+(j+1),base=l?352:76;
  let pos=i===j?[base+138,p[a][1]-27]:i===0?[base+83,223]:[base+202,247];
  edge(id,a,b,weightName(id)+'='+f(g.state['W'+l][i][j],3),pos);
 }
 for(let k=1;k<=2;k++){edge('loss'+k,'y'+k,'loss'+k);edge('total'+k,'loss'+k,'L');}
 function datum(id,x,y,label,v,kind='circle'){
  let isSel=selected===id,kn=known.has(id),value=kn?f(v,5):'待计算';
  let shape=kind==='loss'?`<rect x="${x-55}" y="${y-35}" width="110" height="70" rx="9"/>`:kind==='total'?`<rect x="${x-40}" y="${y-39}" width="80" height="78" rx="11"/>`:`<circle cx="${x}" cy="${y}" r="36"/>`;
  content+=`<g data-fc-item="${id}" role="button" tabindex="0" aria-label="查看${label}" class="fc-neuron ${isSel?'fc-selected':''} ${kn?'':'fc-pending'}">${shape}${sg(x,y-6,label,'fc-svg-node-name')}${sg(x,y+17,value,'fc-svg-value')}</g>`;
  if(grads[id]!==undefined)content+=`<rect x="${x-78}" y="${y+46}" width="156" height="28" rx="5" fill="#fbebde"/>`+sg(x,y+65,(opts.gradientLabels?.[id]||'∂L/∂'+label)+'='+f(grads[id],4),'fc-svg-gradient');
 }
 for(const id of ['x1','x2','h1','h2','y1','y2'])datum(id,...p[id],id[0]+sub[+id[1]-1],val(g,id));
 for(let j=1;j<=2;j++){
  for(const [pre,node,label]of[['a1_','h','a'],['a2_','y','z']]){let[x,y]=p[node+j],delta=opts.preGradients?.[pre+j];content+=sg(x,y-(delta===undefined?51:68),label+sub[j-1]+' = '+(known.has(pre+j)?f(val(g,pre+j),4):'?'),'fc-svg-small');if(delta!==undefined)content+=sg(x,y-48,'δ'+(node==='h'?'ʰ':'ʸ')+sub[j-1]+' = '+f(delta,5),'fc-svg-gradient');}
  datum('loss'+j,...p['loss'+j],'L'+sub[j-1],val(g,'loss'+j),'loss');
  content+=`<g data-fc-item="d${j}" role="button" tabindex="0" aria-label="查看目标d${j}"><rect x="775" y="${p['loss'+j][1]-76}" width="98" height="27" rx="5" fill="#edf1e9"/>${sg(824,p['loss'+j][1]-57,'目标 d'+sub[j-1]+'='+f(g.state.d[j-1]),'fc-svg-small')}</g><path d="M824 ${p['loss'+j][1]-48}V${p['loss'+j][1]-36}" stroke="#9baa96" marker-end="url(#${forward})"/>`;
 }
 datum('L',...p.L,'L',g.result.loss,'total');
 content+=sg(76,426,'给定数据','fc-svg-small')+sg(352,426,'乘权重 → 求和 → Sigmoid','fc-svg-small')+sg(628,426,'乘权重 → 求和 → Sigmoid','fc-svg-small')+sg(900,426,'L = L₁ + L₂','fc-svg-small');
 return `<div class="fc-network-scroll"><svg class="fc-network" viewBox="0 0 1040 445" role="img" aria-label="同一个两输入、两隐藏、两输出网络；层号0、1、2，箭头和数值对应当前步骤">${content}</svg></div><div class="fc-flow-legend"><span>圈内：前向数值</span><span class="fc-forward-key">蓝色：本步前向路径</span><span class="fc-reverse-key">橙色：本步回传的敏感度</span></div>`;
}
const shared={state,graph:()=>G.networkGraph(state),reset:()=>Object.assign(state,clone(initial)),networkHTML};
function weightOptions(){let a=[];for(let l=0;l<2;l++)for(let i=1;i<=2;i++)for(let j=1;j<=2;j++){let id='w'+l+'_'+i+j;a.push([id,weightName(id)+'：'+(l?'隐藏 ':'输入 ')+i+' → '+(l?'输出 ':'隐藏 ')+j]);}return a;}
// Shared controls expose the same sample and selected parameter in lessons 03–06.
// Changing a value recomputes the currently visible step from the same graph.
function networkControls(c){
 return `<details class="fc-card" open><summary><strong>动手改一改：输入、目标与连接权重</strong></summary>
 <p>选择一条连接并拖动权重；保留当前计算步骤，观察数值和梯度如何变化。修改会带到后续章节。</p>
 <div class="fc-controls">${c.range('输入 x₁','fc-net-x1',-2,2,.05,state.x[0])}${c.range('输入 x₂','fc-net-x2',-2,2,.05,state.x[1])}
 ${c.select('选择连接','fc-net-weight',weightOptions(),'w0_11')}${c.range('所选连接的权重','fc-net-weight-value',-3,3,.05,state.W0[0][0])}
 ${c.select('目标向量','fc-net-target',[['10','d = [1, 0]'],['01','d = [0, 1]']],state.d[0]?'10':'01')}</div></details>`;
}
function selectedNetworkWeight(c){
 const id=c.$('#fc-net-weight').value;
 return {id,layer:Number(id[1]),i:Number(id[3])-1,j:Number(id[4])-1};
}
function syncNetworkControls(c){
 const {layer,i,j}=selectedNetworkWeight(c);
 c.$('#fc-net-x1').value=state.x[0]; c.$('#fc-net-x2').value=state.x[1];
 c.$('#fc-net-weight-value').value=state['W'+layer][i][j];
 c.$('#fc-net-target').value=state.d[0]?'10':'01'; c.outputs();
}
function networkControlEvent(c,e){
 if(e.id==='fc-net-x1')state.x[0]=Number(e.value);
 else if(e.id==='fc-net-x2')state.x[1]=Number(e.value);
 else if(e.id==='fc-net-weight'){syncNetworkControls(c);return true;}
 else if(e.id==='fc-net-weight-value'){
  const {layer,i,j}=selectedNetworkWeight(c);state['W'+layer][i][j]=Number(e.value);
 }else if(e.id==='fc-net-target')state.d=e.value==='10'?[1,0]:[0,1];
 else return false;
 return true;
}
function lockNetworkControls(c,locked){
 ['fc-net-x1','fc-net-x2','fc-net-weight','fc-net-weight-value','fc-net-target'].forEach(id=>{c.$('#'+id).disabled=locked;});
}
function nav(max){return `<div class="fc-step-controls"><button class="quiet" id="fc-prev">← 上一步</button><span id="fc-count"></span><button class="primary" id="fc-next">计算下一步 →</button><button class="quiet" id="fc-reset">恢复初始样例</button></div>`;}
function setNav(c,step,max){c.$('#fc-count').textContent=step+' / '+max;c.$('#fc-prev').disabled=step===0;c.$('#fc-next').disabled=step===max;let strip=c.$('#fc-live-step');if(!strip){strip=document.createElement('div');strip.id='fc-live-step';strip.className='fc-live-step';c.root.insertBefore(strip,c.$('#fc-network'));}let title=c.$('#fc-explain h3')?.textContent||'',equations=c.$$('#fc-explain>.fc-card>.fc-equation');strip.innerHTML='<strong>'+esc(title)+'</strong>'+(equations.length?'<div class="fc-live-equation">'+esc(equations.at(-1).textContent)+'</div>':'');}
function card(title,html,kind=''){return `<section class="fc-card ${kind}"><h3>${title}</h3>${html}</section>`;}
function eq(s){return `<div class="fc-equation">${s}</div>`;}
function bindItems(c,fn){c.$$('[data-fc-item]',c.root).forEach(e=>{e.onclick=()=>fn(e.dataset.fcItem);e.onkeydown=v=>{if(v.key==='Enter'||v.key===' '){v.preventDefault();fn(e.dataset.fcItem);}};});}

function forward(c){
 const{root,$,select,controls}=c;let step=0;
 root.innerHTML=`<div class="fc-step-note">每按一次，只完成图中一项计算。尚未算到的节点保留“待计算”；目标只在最后与预测比较。</div>${networkControls(c)}${nav(8)}<div id="fc-network"></div><div id="fc-explain" aria-live="polite"></div>`;
 function draw(){let g=shared.graph(),r=g.result,known=new Set(['x1','x2']),fw=[],selected='',title='',html='';
  if(step>=1)known.add('a1_1');if(step>=2)known.add('h1');if(step>=3)['a1_2','h2'].forEach(x=>known.add(x));if(step>=4)known.add('a2_1');if(step>=5)known.add('y1');if(step>=6)['a2_2','y2'].forEach(x=>known.add(x));if(step>=7)['loss1','loss2'].forEach(x=>known.add(x));if(step>=8)known.add('L');
  const weighted=(source,W,j)=>`${f(source[0])} × (${f(W[0][j])}) + ${f(source[1])} × (${f(W[1][j])})`;
  switch(step){
   case 0:title='0 · 输入和参数给定，还没有计算隐藏值';html=`<p>输入 x=${fmtVec(state.x)}。边上是当前权重，不是需要传过去的另一个样本。下一步先算隐藏节点 h₁ 的加权和。</p>`;break;
   case 1:title='1 · 两项乘积相加，得到隐藏 1 的激活前值 a₁';fw=['w0_11','w0_21'];selected='h1';html=eq('a₁ = '+weighted(state.x,state.W0,0)+' = '+f(r.a1[0]))+'<p>现在只有加权和。h₁ 还没有经过 Sigmoid，因此仍写“待计算”。</p>';break;
   case 2:title='2 · 把 a₁ 送入 Sigmoid，得到隐藏输出 h₁';selected='h1';html=eq('h₁ = 1 / (1 + exp(−a₁)) = '+f(r.h[0]))+'<p>这就是从此节点送往下一层的数。a₁ 是激活前的和；h₁ 是激活后的输出。</p>';break;
   case 3:title='3 · 同样计算第二个隐藏节点';fw=['w0_12','w0_22'];selected='h2';html=eq('a₂ = '+weighted(state.x,state.W0,1)+' = '+f(r.a1[1]))+eq('h₂ = Sigmoid(a₂) = '+f(r.h[1]));break;
   case 4:title='4 · 输出 1 收到的是隐藏值 h₁、h₂';fw=['w1_11','w1_21'];selected='y1';html=eq('z₁ = '+weighted(r.h,state.W1,0)+' = '+f(r.a2[0]))+'<p>这一层的“输入”已经是 h，不再是原始数据 x。连接权重换成 v，但乘、加的规则相同。</p>';break;
   case 5:title='5 · 输出节点也经过 Sigmoid';selected='y1';html=eq('y₁ = Sigmoid(z₁) = '+f(r.y[0]))+'<p>到这里得到一个预测分量。标签尚未参与这些节点的计算。</p>';break;
   case 6:title='6 · 计算另一个输出分量';fw=['w1_12','w1_22'];selected='y2';html=eq('z₂ = '+weighted(r.h,state.W1,1)+' = '+f(r.a2[1]))+eq('y₂ = Sigmoid(z₂) = '+f(r.y[1]));break;
   case 7:title='7 · 预测与各自的目标比较，分开计算损失';fw=['loss1','loss2'];html=eq(`L₁ = ½(${f(r.y[0])} − ${state.d[0]})² = ${f(val(g,'loss1'),7)}`)+eq(`L₂ = ½(${f(r.y[1])} − ${state.d[1]})² = ${f(val(g,'loss2'),7)}`)+'<p>目标进入损失分支，不回到隐藏层充当输入。</p>';break;
   case 8:title='8 · 两项损失相加，完成一次前向';fw=['total1','total2'];selected='L';html=eq('L = '+f(val(g,'loss1'),7)+' + '+f(val(g,'loss2'),7)+' = '+f(r.loss,7))+'<p>接下来固定这次前向的所有数值，计算“每个数改变一点，L 会怎样变”。反向阶段不会覆盖这些缓存值。</p>';break;
  }
  $('#fc-network').innerHTML=networkHTML(g,{known,forwardEdges:fw,selected});$('#fc-explain').innerHTML=card(title,html);setNav(c,step,8);
 }
 controls(e=>{networkControlEvent(c,e);draw();});$('#fc-prev').onclick=()=>{step--;draw();};$('#fc-next').onclick=()=>{step++;draw();};$('#fc-reset').onclick=()=>{shared.reset();step=0;syncNetworkControls(c);draw();};draw();
}

function output(c){
 const{root,$,select,range,controls,value}=c;let step=0;
 root.innerHTML=`<div class="fc-step-note">每次调整后先重新前向，固定这一组数值，再从 L 开始按相反方向计算偏导。箭头传的是“L 对这个量有多敏感”，不是把预测倒着搬回去。</div>${networkControls(c)}<div class="fc-controls">${select('从哪个输出回传','fc-output',[['0','输出 1：y₁'],['1','输出 2：y₂']],'0')}${select('追踪哪个前层输入','fc-source',[['0','隐藏 1：h₁'],['1','隐藏 2：h₂']],'0')}${range('第 5 步：临时微扰 ε','fc-eps',-.05,.05,.001,.01)}</div>${nav(5)}<div id="fc-network"></div><div id="fc-explain" aria-live="polite"></div>`;
 function draw(){let g=shared.graph(),r=g.result,k=value('fc-output'),j=value('fc-source'),y=r.y[k],d=state.d[k],v=state.W1[j][k],h=r.h[j],gy=y-d,local=y*(1-y),delta=r.d2[k],gw=h*delta,gh=v*delta,rev=[],grads={},wgrads={},title='',html='',yi='y'+(k+1),hi='h'+(j+1),wi='w1_'+(j+1)+(k+1),ys='y'+sub[k],hs='h'+sub[j],zs='z'+sub[k],vs=weightName(wi);
  if(step>=1)rev.push('total'+(k+1));if(step>=2){rev.push('loss'+(k+1));grads[yi]=gy;}if(step>=4){rev.push(wi);grads[hi]=gh;wgrads[wi]=gw;}
  switch(step){
   case 0:title='0 · 先说清楚我们想求什么';html=`<p>本次追踪 ${vs}，它把 ${hs} 乘权重后送向 ${ys}。问题是：只把这条权重加一点，最终损失 L 怎样变化？这就是 ∂L/∂${vs}。</p>`+eq(`${hs} → 乘 ${vs} → ${zs} → Sigmoid → ${ys} → 平方误差 → L`);break;
   case 1:title='1 · 从损失开始，起始敏感度是 1';html=eq('∂L/∂L = 1')+eq('L = L₁ + L₂，因此 ∂L/∂L'+sub[k]+' = 1')+'<p>损失本身增加 ε，总损失就增加 ε。加法把同一份下游敏感度送给每个加数。</p>';break;
   case 2:title='2 · 穿过平方误差，先得到对预测的导数';html=eq(`L${sub[k]} = ½(${ys} − d${sub[k]})²`)+eq(`∂L/∂${ys} = 1 × (${ys} − d${sub[k]}) = ${f(gy,6)}`)+`<p>${gy<0?'导数为负：这个预测值稍增大，损失会下降。':'导数为正：这个预测值稍增大，损失会增大。'}这里还没有修改预测，只计算了一个变化率。</p>`;break;
   case 3:title='3 · 穿过 Sigmoid，把敏感度送回激活前的和';html=eq(`局部导数 ∂${ys}/∂${zs} = ${ys}(1−${ys}) = ${f(local,6)}`)+eq(`δʸ${sub[k]} ≔ ∂L/∂${zs} = ${f(gy,6)} × ${f(local,6)} = ${f(delta,6)}`)+`<p>δʸ${sub[k]} 只是这一个激活前偏导的简称：上标 y 提醒它属于输出层，下标 ${k+1} 指本层第 ${k+1} 个节点。</p>`;break;
   case 4:title='4 · 乘法向两个输入分别回传，乘的数不同';html=`<p>${zs} 包含乘积 ${hs}${vs}。对乘积的一个输入求导，局部导数是另一个输入。</p><div class="fc-grid-two">`+card('送给可训练权重 '+vs,eq(`∂${zs}/∂${vs} = ${hs} = ${f(h)}`)+eq(`∂L/∂${vs} = δʸ${sub[k]} × ${hs} = ${f(gw,7)}`))+card('送回前层数值 '+hs,eq(`∂${zs}/∂${hs} = ${vs} = ${f(v)}`)+eq(`来自本输出的贡献 = δʸ${sub[k]} × ${vs} = ${f(gh,7)}`))+'</div><p>橙色回到 '+hs+' 的数只是这条路径的贡献。另一个输出也会影响同一个 '+hs+'；下一节再把两条贡献收齐。</p>';break;
   case 5:{let eps=value('fc-eps'),perturbed=clone(state);perturbed.W1[j][k]+=eps;let r2=G.networkGraph(perturbed).result,actual=r2.loss-r.loss,predicted=gw*eps;html=eq(`${vs}：${f(v)} → ${f(v+eps)}；其他参数与输入固定`)+`<div class="fc-grid-two">`+card('重新前向得到的真实变化',eq(`L：${f(r.loss,7)} → ${f(r2.loss,7)}`)+eq(`ΔL = ${f(actual,8)}`))+card('当前梯度预测的小变化',eq(`ΔL ≈ (∂L/∂${vs}) ε`)+eq(`${f(gw,7)} × ${f(eps)} = ${f(predicted,8)}`))+'</div><p>ε 越小，局部线性预测通常越接近重新计算的变化。这个临时试探没有写回网络参数；真正更新在第 6 节。</p>';title='5 · 用同一个网络的小扰动验证梯度含义';break;}
  }
  $('#fc-network').innerHTML=networkHTML(g,{selected:step>=4?wi:yi,reverseEdges:rev,gradients:grads,weightGradients:wgrads,gradientLabels:{[hi]:'∂L'+sub[k]+'/∂'+hs},preGradients:step>=3?{['a2_'+(k+1)]:delta}:{}});$('#fc-explain').innerHTML=card(title,html,'fc-reverse-card');setNav(c,step,5);
 }
 controls(e=>{networkControlEvent(c,e);draw();});$('#fc-prev').onclick=()=>{step--;draw();};$('#fc-next').onclick=()=>{step++;draw();};$('#fc-reset').onclick=()=>{shared.reset();step=0;syncNetworkControls(c);draw();};draw();
}

function hidden(c){
 const{root,$,select,controls,value}=c;let step=0;
 root.innerHTML=`<div class="fc-step-note">这里已完成两个输出的局部反传。隐藏节点同时影响两个输出，必须先收齐两份带正负号的贡献，再穿过自己的 Sigmoid。</div>${networkControls(c)}<div class="fc-controls">${select('追踪哪个隐藏节点','fc-hidden',[['0','隐藏 1：h₁'],['1','隐藏 2：h₂']],'0')}</div>${nav(5)}<div id="fc-network"></div><div id="fc-explain" aria-live="polite"></div>`;
 function draw(){let g=shared.graph(),r=g.result,j=value('fc-hidden'),hi='h'+(j+1),hs='h'+sub[j],as='a'+sub[j],v=state.W1[j],p=v.map((w,k)=>w*r.d2[k]),gh=p[0]+p[1],dh=r.d1[j],local=r.h[j]*(1-r.h[j]),rev=[],grads={},wg={},html='',title='';
  if(step>=1)rev.push('w1_'+(j+1)+'1');if(step>=2)rev.push('w1_'+(j+1)+'2');if(step>=3)grads[hi]=gh;
  if(step>=5)for(let i=0;i<2;i++){let id='w0_'+(i+1)+(j+1);rev.push(id);wg[id]=r.g0[i][j];}
  const contribution=k=>eq(`输出 ${k+1}：δʸ${sub[k]} × v${sub[j]}${sub[k]} = (${f(r.d2[k],6)}) × (${f(v[k])}) = ${f(p[k],7)}`);
  switch(step){
   case 0:title='0 · 两个输出已经各自收到误差信号';html=eq(`δʸ₁ = ${f(r.d2[0],6)}；δʸ₂ = ${f(r.d2[1],6)}`)+`<p>两者分别表示 ∂L/∂z₁、∂L/∂z₂。现在要问：${hs} 改变一点，经两条不同路径会怎样共同改变 L？</p>`;break;
   case 1:title='1 · 先收到输出 1 的贡献';html=contribution(0)+`<p>跨过的线性运算是 z₁ = h₁v₁₁ + h₂v₂₁。对当前输入 ${hs} 求导，乘对应连接权重，而不是乘 ${hs} 本身。另一条路径还没收到，不能把这一项当作总导数。</p>`;break;
   case 2:title='2 · 再收到输出 2 的贡献';html=contribution(0)+contribution(1)+'<p>两条贡献可以同号，也可以异号。符号来自各自的输出梯度与连接权重；不能先取绝对值再加。</p>';break;
   case 3:title='3 · 共享输入处相加，得到对隐藏输出的总导数';html=contribution(0)+contribution(1)+eq(`∂L/∂${hs} = ${f(p[0],7)} + (${f(p[1],7)}) = ${f(gh,7)}`)+`<p>相加发生在同一个变量 ${hs} 上，因为改变它会同时改变两条后续路径。这不是激活函数的导数，还没回到 ${as}。</p>`;break;
   case 4:title='4 · 穿过隐藏 Sigmoid，才得到隐藏层 δ';html=eq(`局部导数 ∂${hs}/∂${as} = ${hs}(1−${hs}) = ${f(local,6)}`)+eq(`δʰ${sub[j]} ≔ ∂L/∂${as} = (${f(gh,7)}) × ${f(local,6)} = ${f(dh,7)}`)+`<p>先汇合，再乘本节点的局部导数。图中圈内仍是前向 ${hs}=${f(r.h[j])}，橙色数字才是损失敏感度。</p>`;break;
   case 5:title='5 · 同一 δ 分给入边权重与更前面的输入';html=`<div class="fc-grid-two">`+card('对参数求导：乘进入此节点的数',eq(`∂L/∂w₁${sub[j]} = δʰ${sub[j]} × x₁ = ${f(r.g0[0][j],7)}`)+eq(`∂L/∂w₂${sub[j]} = δʰ${sub[j]} × x₂ = ${f(r.g0[1][j],7)}`))+card('对输入求导：乘对应权重',eq(`传给 x₁ 的本路贡献 = δʰ${sub[j]} × w₁${sub[j]} = ${f(dh*state.W0[0][j],7)}`)+eq(`传给 x₂ 的本路贡献 = δʰ${sub[j]} × w₂${sub[j]} = ${f(dh*state.W0[1][j],7)}`))+'</div><p>原始输入还连接另一个隐藏节点；第 6 节继续把它的贡献相加，并区分敏感度与参数更新。</p>';break;
  }
  $('#fc-network').innerHTML=networkHTML(g,{selected:hi,reverseEdges:rev,gradients:grads,weightGradients:wg,reverseLabels:{['w1_'+(j+1)+'1']:'给'+hs+'：'+f(p[0],5),['w1_'+(j+1)+'2']:'给'+hs+'：'+f(p[1],5)},preGradients:step>=4?{['a1_'+(j+1)]:dh}:{}});$('#fc-explain').innerHTML=card(title,html,'fc-reverse-card');setNav(c,step,5);
 }
 controls(e=>{networkControlEvent(c,e);draw();});$('#fc-prev').onclick=()=>{step--;draw();};$('#fc-next').onclick=()=>{step++;draw();};$('#fc-reset').onclick=()=>{shared.reset();step=0;syncNetworkControls(c);draw();};draw();
}

function update(c){
 const{root,$,select,range,controls,value}=c;let step=0,committed=false,oldGraph=null,nextState=null,lr=.5;
 root.innerHTML=`<div class="fc-step-note">先完成回到原始输入的敏感度，再用同一份旧参数梯度统一更新 8 条权重。输入、标签和中间激活各有不同身份。提交更新后，本次对照固定输入与参数；恢复初始样例后可继续调整。</div>${networkControls(c)}<div class="fc-controls">${select('查看哪个原始输入','fc-input',[['0','输入 1：x₁'],['1','输入 2：x₂']],'0')}${range('学习率 λ','fc-lr',.01,2,.01,.5)}${select('核验哪条权重','fc-check-weight',weightOptions(),'w0_11')}</div>${nav(4)}<div id="fc-network"></div><div id="fc-explain" aria-live="polite"></div><div id="fc-update-actions" class="fc-step-controls"></div><div id="fc-check"></div>`;
 function draw(){let g=oldGraph||shared.graph(),r=g.result,i=value('fc-input'),xs='x'+sub[i],parts=[0,1].map(j=>g.state.W0[i][j]*r.d1[j]),rev=[],grads={},wg={},html='',title='';
  if(step>=1){for(let j=0;j<2;j++)rev.push('w0_'+(i+1)+(j+1));grads['x'+(i+1)]=r.gx[i];}
  if(step>=2){let id=$('#fc-check-weight').value;wg[id]=g.adjoints[id];}
  switch(step){
   case 0:title='0 · 上一层对输入求导，仍然要收齐所有路径';html=eq(`δʰ₁ = ${f(r.d1[0],7)}；δʰ₂ = ${f(r.d1[1],7)}`)+`<p>${xs} 同时送入两个隐藏节点。因此还需要把“来自隐藏 1”和“来自隐藏 2”的敏感度相加。</p>`;break;
   case 1:title='1 · 真正回到原始输入，但只记录敏感度';html=eq(`∂L/∂${xs} = w${sub[i]}₁δʰ₁ + w${sub[i]}₂δʰ₂`)+eq(`${f(parts[0],7)} + (${f(parts[1],7)}) = ${f(r.gx[i],7)}`)+`<p>它表示“若 ${xs} 略变，L 会怎样变”。普通监督训练固定观测样本 x 和标签 d，不用这条梯度去修改它们。隐藏值 h 是前面权重计算的结果，得到 ∂L/∂h 后还要继续追到产生 h 的参数。</p>`;break;
   case 2:{title='2 · 参数梯度已经齐备，先列账，再统一更新';let rows=[];for(let l=0;l<2;l++)for(let a=0;a<2;a++)for(let b=0;b<2;b++){let id='w'+l+'_'+(a+1)+(b+1),old=g.state['W'+l][a][b],gr=(l?r.g1:r.g0)[a][b];rows.push(`<tr><th>${weightName(id)}</th><td>${f(old,5)}</td><td>${f(gr,7)}</td><td>${f(-value('fc-lr')*gr,7)}</td><td>${f(old-value('fc-lr')*gr,7)}</td></tr>`);}html='<p>所有梯度都在当前同一组旧参数处计算。更新公式：新权重 = 旧权重 − λ × 梯度。</p><div class="fc-table-scroll"><table class="fc-table"><thead><tr><th>参数</th><th>旧值</th><th>∂L/∂参数</th><th>变化 −λg</th><th>预览新值</th></tr></thead><tbody>'+rows.join('')+'</tbody></table></div><p>输入和标签不在这张待更新列表；h、y 也不作为独立参数更新。</p>';break;}
   case 3:title='3 · 参数已更新，接着重新前向计算';html=eq('旧输入 x = '+fmtVec(g.state.x)+'；更新后 x = '+fmtVec(state.x))+eq('目标 d 仍为 '+fmtVec(state.d))+`<p>已用缓存梯度统一更新所有权重。当前图保留旧前向值供对照；按下一步，使用新参数重新算 h、y 和 L。旧梯度不是新参数位置的梯度。</p>`;break;
   case 4:{let after=G.networkGraph(nextState).result;title='4 · 输入没变，参数改变使隐藏值与预测重新算出';html=`<div class="fc-grid-two">`+card('更新前：旧参数的前向',eq('h = '+fmtVec(r.h))+eq('y = '+fmtVec(r.y))+eq('L = '+f(r.loss,7)))+card('更新后：新参数的前向',eq('h = '+fmtVec(after.h))+eq('y = '+fmtVec(after.y))+eq('L = '+f(after.loss,7)))+'</div>'+`<p>这一步损失${after.loss<r.loss?'下降':'没有下降'}。单次下降取决于步长与当前位置；这只是这个样本的一步学习。输入 x、目标 d 始终保持原值。</p>`;break;}
  }
  $('#fc-network').innerHTML=networkHTML(step===4?G.networkGraph(nextState):g,{selected:step<2?'x'+(i+1):$('#fc-check-weight').value,reverseEdges:step===4?[]:rev,gradients:step===4?{}:grads,weightGradients:step===4?{}:wg});$('#fc-explain').innerHTML=card(title,html,step<3?'fc-reverse-card':'');setNav(c,step,4);
  if(step===2){$('#fc-next').disabled=true;$('#fc-update-actions').innerHTML='<button class="primary" id="fc-commit">用这一份梯度，统一更新全部权重</button>';$('#fc-commit').onclick=()=>{lr=value('fc-lr');oldGraph=g;nextState=clone(g.state);for(let l=0;l<2;l++)for(let a=0;a<2;a++)for(let b=0;b<2;b++)nextState['W'+l][a][b]-=lr*(l?r.g1:r.g0)[a][b];Object.assign(state,clone(nextState));committed=true;syncNetworkControls(c);lockNetworkControls(c,true);step=3;draw();};}
  else $('#fc-update-actions').innerHTML='';
  $('#fc-prev').disabled=step===0||committed;
  $('#fc-lr').disabled=committed;
  if(step>=2){let id=$('#fc-check-weight').value,check=G.finiteDifference('network',g.state,1e-5).rows.find(x=>x.id===id);$('#fc-check').innerHTML=card('用独立中心差分核对 '+weightName(id),eq(`反向传播：${f(check.analytic,9)}；中心差分：${f(check.numeric,9)}`)+`<p>只临时扰动这一条权重 ±10⁻⁵，分别重新前向。两者绝对差 ${check.absoluteError.toExponential(2)}。${committed?'这里核对的是本次更新采用的旧参数梯度。':''}</p>`);}
  else $('#fc-check').innerHTML='';
 }
 controls(e=>{if(!committed)networkControlEvent(c,e);draw();});$('#fc-prev').onclick=()=>{step--;draw();};$('#fc-next').onclick=()=>{step++;draw();};$('#fc-reset').onclick=()=>{shared.reset();oldGraph=null;nextState=null;committed=false;step=0;lr=.5;$('#fc-lr').disabled=false;$('#fc-lr').value=lr;$('#fc-input').value='0';$('#fc-check-weight').value='w0_11';lockNetworkControls(c,false);syncNetworkControls(c);draw();};draw();
}

window.FoundationLesson=function(c,l){
 const handlers={...window.FoundationIntroLabs,...window.FoundationExtraLabs,'foundation-forward':forward,'foundation-output':output,'foundation-hidden':hidden,'foundation-update':update};
 if(!handlers[l.mode])throw new Error('Unknown foundational lesson '+l.mode);
 handlers[l.mode](c,shared);
};
}());
