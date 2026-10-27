/* Offline teaching interface. Every displayed experiment is recomputed locally. */
(function(){
'use strict';
const M=window.NNMath, lessons=[...(window.FOUNDATIONLESSONS||[]),...(window.NNLESSONS||[])];
const aliases={network:'neuron',backprop:'forward'};
const lessonId=hash=>aliases[hash]||hash;
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const C={blue:'#2c647d',orange:'#be7040',green:'#557363',muted:'#6b7b82',grid:'#e2e8e7'};
const fmt=(x,n=4)=>!Number.isFinite(x)?String(x):(Math.abs(x)>0&&Math.abs(x)<.0001?x.toExponential(2):Number(x.toFixed(n)).toString());
const sum=a=>a.reduce((x,y)=>x+y,0), clone=x=>JSON.parse(JSON.stringify(x));
const root=$('#experiment');
let lessonIndex=0,cleanup=()=>{},progress={done:{},last:'neuron'},mathQueue=Promise.resolve(),plotSerial=0;
try{progress=Object.assign(progress,JSON.parse(localStorage.getItem('nn-from-zero-progress')||'{}'));}catch(_){}
if(!M||!lessons){root.innerHTML='<p class="error-message">课程文件未能加载。请保留整个 neural-lab 文件夹，再打开 index.html。</p>';return;}
function save(){try{localStorage.setItem('nn-from-zero-progress',JSON.stringify(progress));}catch(_){}updateProgress();}
function typeset(){if(window.MathJax?.typesetPromise)mathQueue=mathQueue.then(()=>MathJax.typesetPromise([$('#main')])).catch(()=>{});}
function range(label,id,min,max,step,value){return `<label class="control"><span class="control-top"><span>${label}</span><output for="${id}" id="${id}-out">${fmt(value)}</output></span><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;}
function select(label,id,options,value){return `<label class="control"><span class="control-top">${label}</span><select id="${id}">${options.map(([v,t])=>`<option value="${v}" ${v==value?'selected':''}>${t}</option>`).join('')}</select></label>`;}
function value(id){return Number($('#'+id).value);}
function outputs(){ $$('input[type=range]',root).forEach(e=>{$('#'+e.id+'-out').textContent=fmt(+e.value);});}
function controls(onchange){ $$('input,select',root).forEach(e=>e.addEventListener('input',()=>{outputs();onchange(e);}));}
function svg(content,w=540,h=260,label='数值图示'){return `<svg class="lab-visual" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${content}</svg>`;}
function text(x,y,t,cls='plot-label',extra=''){return `<text x="${x}" y="${y}" class="${cls}" ${extra}>${t}</text>`;}
function line(x1,y1,x2,y2,cls='edge',extra=''){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}" ${extra}/>`;}
function node(x,y,label,val,active=false){return `<rect x="${x-45}" y="${y-27}" width="90" height="54" rx="7" class="node ${active?'active':''}"/>${text(x,y-3,label,'node-text')}${text(x,y+17,val,'node-text')}`;}
function stats(rows){return `<div class="stat-row">${rows.map(([k,v])=>`<span>${k}<strong>${v}</strong></span>`).join('')}</div>`;}
function matrix(a,label){return `<table class="matrix"><caption>${label}</caption><tbody>${a.map(row=>'<tr>'+row.map(v=>`<td>${fmt(v,3)}</td>`).join('')+'</tr>').join('')}</tbody></table>`;}
function plot(series,{width=540,height=230,xLabel='x',yLabel='y',xMin,xMax,yMin,yMax,contours}={}){
  let points=series.flatMap(s=>s.points).filter(p=>p.every(Number.isFinite));
  if(!points.length)return '';
  xMin=xMin??Math.min(...points.map(p=>p[0]));xMax=xMax??Math.max(...points.map(p=>p[0]));
  yMin=yMin??Math.min(0,...points.map(p=>p[1]));yMax=yMax??Math.max(...points.map(p=>p[1]));
  if(xMax===xMin)xMax=xMin+1;if(yMax===yMin)yMax=yMin+1;
  if(yMax>yMin){const pad=(yMax-yMin)*.08;yMin-=pad;yMax+=pad;}
  let l=52,r=18,t=16,b=42, X=x=>l+(x-xMin)/(xMax-xMin)*(width-l-r),Y=y=>height-b-(y-yMin)/(yMax-yMin)*(height-t-b),out='';
  for(let k=0;k<5;k++){let y=yMin+(yMax-yMin)*k/4;out+=line(l,Y(y),width-r,Y(y),'plot-grid')+text(l-7,Y(y)+4,fmt(y,2),'plot-label','text-anchor="end"');}
  for(let k=0;k<5;k++){let x=xMin+(xMax-xMin)*k/4;out+=text(X(x),height-b+20,fmt(x,2),'plot-label','text-anchor="middle"');}
  out+=line(l,t,l,height-b,'axis')+line(l,height-b,width-r,height-b,'axis');
  out+=text(width/2,height-3,xLabel,'plot-label','text-anchor="middle"')+text(l,12,yLabel);
  const clipId='plot-clip-'+(++plotSerial);out+=`<defs><clipPath id="${clipId}"><rect x="${l}" y="${t}" width="${width-l-r}" height="${height-t-b}"/></clipPath></defs><g clip-path="url(#${clipId})">`;
  if(contours)for(const level of [.025,.1,.25,.5,1,2,4,8])out+=`<ellipse cx="${X(0)}" cy="${Y(0)}" rx="${Math.sqrt(2*level/contours[0])*(width-l-r)/(xMax-xMin)}" ry="${Math.sqrt(2*level/contours[1])*(height-t-b)/(yMax-yMin)}" fill="none" stroke="#d1dad6" stroke-width="1"/>`;
  for(const s of series){let p=s.points.filter(p=>p.every(Number.isFinite));out+=`<polyline points="${p.map(p=>`${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="${s.color||C.blue}" stroke-width="2.2" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
    if(s.dots)out+=p.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="${s.color||C.blue}"/>`).join('');}
  return svg(out+'</g>',width,height,`${xLabel}与${yLabel}关系`);
}
function legend(items){return '<div class="legend">'+items.map(([name,color])=>`<span style="--swatch:${color}">${name}</span>`).join('')+'</div>';}
function updateProgress(){
  $('#completion').textContent=`${Object.keys(progress.done).filter(k=>progress.done[k]).length} / ${lessons.length} 节完成检查`;
  $$('.chapter').forEach((b,i)=>{b.classList.toggle('active',i===lessonIndex);b.classList.toggle('done',!!progress.done[lessons[i].id]);$('.chapter-number',b).textContent=progress.done[lessons[i].id]?'✓':String(i+1).padStart(2,'0');});
}
function renderLesson(index){
  cleanup();cleanup=()=>{};lessonIndex=Math.max(0,Math.min(lessons.length-1,index));
  const l=lessons[lessonIndex];progress.last=l.id;save();
  document.body.classList.toggle('foundation-mode',l.mode.startsWith('foundation-')||l.mode==='residual');
  if(window.MathJax?.typesetClear)MathJax.typesetClear([$('#main')]);
  $('#lesson-kicker').textContent=`第 ${lessonIndex+1} 节 / ${lessons.length} · ${lessonIndex<6?'看清局部计算':lessonIndex<13?'让网络学起来':'理解结构与训练'}`;
  $('#lesson-title').textContent=l.title;$('#lesson-intro').textContent=l.intro;
  $('#lesson-body').innerHTML=l.body;$('#task-text').innerHTML=l.task;
  if(location.hostname.endsWith('.github.io') || window.EDML_STANDALONE) {
    document.querySelectorAll('a[href^="../lessons/"],a[href="../README.md"]').forEach(link=>{
      link.href='https://github.com/Lee-Eee-Eee/everybody-digs-ml/blob/main/'+link.getAttribute('href').slice(3);
    });
  }
  $('#lesson-map').innerHTML=lessons.map((x,i)=>`<button class="chapter-dot ${i===lessonIndex?'current':''} ${progress.done[x.id]?'complete':''}" title="${i+1}. ${x.title}" aria-label="跳到${x.title}" data-lesson="${i}"></button>`).join('');
  $('#quiz-question').textContent=l.quiz.question;
  const options=l.quiz.options.map((s,i)=>({s,i})),rotate=lessonIndex%options.length;
  $('#quiz-options').innerHTML=options.slice(rotate).concat(options.slice(0,rotate)).map(({s,i})=>`<label class="quiz-option"><input type="radio" name="answer" value="${i}"><span>${s}</span></label>`).join('');
  $('#quiz-feedback').textContent=progress.done[l.id]?'这一节的检查已完成。你仍可以重新作答。':'';
  $('#previous').disabled=lessonIndex===0;$('#next').disabled=lessonIndex===lessons.length-1;$('#chapter-counter').textContent=`${lessonIndex+1} / ${lessons.length}`;
  $('#syllabus').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');
  root.innerHTML='';
  const context={root,$,$$,M,fmt,sum,svg,plot,range,select,stats,legend,controls,outputs,value,text,line,node,setCleanup:fn=>cleanup=fn};
  try{if(l.mode.startsWith('foundation-'))window.FoundationLesson(context,l);
    else if(l.mode==='residual')window.FoundationResidualLab(context,l);
    else ({neuron:neuronLab,graph:graphLab,backprop:backpropLab,batch:batchLab,softmax:softmaxLab,gradient:gradientLab,train:trainLab,optimizer:optimizerLab,residual:residualLab,deep:deepLab,conv:convLab,regularize:regularizeLab}[l.mode]||neuronLab)(l);
  }catch(e){root.innerHTML=`<p class="error-message">实验遇到问题：${e.message}</p>`;console.error(e);}
  typeset();
}
function navigate(i){location.hash=lessons[i].id;}
$('#chapter-list').innerHTML=lessons.map((l,i)=>`${[0,6,13].includes(i)?`<div class="nav-group">${i===0?'Ⅰ · 从数值到偏导':i===6?'Ⅱ · 从梯度到学习':'Ⅲ · 从深度到残差'}</div>`:''}<button class="chapter" data-lesson="${i}"><span class="chapter-number">${i+1}</span><span>${l.short||l.title}</span></button>`).join('');
document.addEventListener('click',e=>{const b=e.target.closest('[data-lesson]');if(b)navigate(+b.dataset.lesson);});
$('#previous').onclick=()=>navigate(lessonIndex-1);$('#next').onclick=()=>navigate(lessonIndex+1);
$('#menu-toggle').onclick=()=>{$('#syllabus').classList.toggle('open');$('#menu-toggle').setAttribute('aria-expanded',$('#syllabus').classList.contains('open'));};
$('#quiz-form').onsubmit=e=>{e.preventDefault();let selected=$('input[name=answer]:checked');if(!selected){$('#quiz-feedback').textContent='先选出你的判断，再来核对。';return;}const q=lessons[lessonIndex].quiz,good=+selected.value===q.correct;$('#quiz-feedback').className=good?'feedback-good':'feedback-retry';$('#quiz-feedback').innerHTML=(good?'判断正确。':'再想一步。')+q.explanation;if(good){progress.done[lessons[lessonIndex].id]=true;save();}typeset();};
window.addEventListener('hashchange',()=>{let i=lessons.findIndex(l=>l.id===lessonId(location.hash.slice(1)));renderLesson(i<0?0:i);window.scrollTo({top:0,behavior:'instant'});});

function neuronLab(){
 root.innerHTML=`<p class="lab-description">改变输入、权重或偏置，观察加权和与局部导数一起变化。</p><div class="control-grid">${range('输入 x₁','n-x1',-2,2,.05,.6)}${range('输入 x₂','n-x2',-2,2,.05,.2)}${range('权重 w₁','n-w1',-5,5,.05,1)}${range('权重 w₂','n-w2',-5,5,.05,-1)}${range('偏置 b','n-b',-6,6,.1,0)}${select('激活函数','n-act',[['sigmoid','Sigmoid'],['relu','ReLU'],['tanh','Tanh'],['linear','线性']], 'sigmoid')}</div><div id="n-view"></div><div id="n-result" class="numeric-block"></div>`;
 function draw(){let x1=value('n-x1'),x2=value('n-x2'),w1=value('n-w1'),w2=value('n-w2'),b=value('n-b'),act=$('#n-act').value,z=x1*w1+x2*w2+b,a=M.activate(z,act),s=M.derivative(z,a,act);
  let graph=line(95,60,245,115)+line(95,175,245,115)+line(290,115,450,115)+node(60,60,'x₁',fmt(x1))+node(60,175,'x₂',fmt(x2))+node(270,115,'加权和',fmt(z),true)+node(480,115,'输出 h',fmt(a));
  graph+=text(145,65,`× ${fmt(w1)}`)+text(145,172,`× ${fmt(w2)}`)+text(231,177,`+ b = ${fmt(b)}`);
  const low=Math.min(-6,z-1),high=Math.max(6,z+1),xs=Array.from({length:101},(_,i)=>low+i*(high-low)/100);
  $('#n-view').innerHTML=svg(graph,540,225,'一个神经元的加权求和与激活')+plot([{points:xs.map(x=>[x,M.activate(x,act)]),color:C.blue},{points:[[z,a]],dots:true,color:C.orange}],{xLabel:'激活前加权和 a',yLabel:'神经元输出',height:200,xMin:low,xMax:high});
  $('#n-result').textContent=`a = (${fmt(x1)})×(${fmt(w1)}) + (${fmt(x2)})×(${fmt(w2)}) + (${fmt(b)}) = ${fmt(z)}\nh = ${fmt(a)}；s′(a) = ${fmt(s)}\n∂h/∂w₁ = s′(a)×x₁ = ${fmt(s*x1)}；∂h/∂b = ${fmt(s)}`;
 }controls(draw);draw();
}
function graphLab(lesson){
 const paths=lesson.id==='paths';let step=0;
 root.innerHTML=`<p class="lab-description">${paths?'x 同时影响两条支路。分别计算，再把回到 x 的导数相加。':'同一个函数，前向传数值，反向传“损失对这个数的敏感度”。'}</p><div class="control-grid">${range(paths?'输入 x':'参数 w','g-x',-2,2,.05,.4)}${range(paths?'路径系数 a':'目标 d','g-a',-2,2,.05,paths?1.2:.8)}${paths?range('直接路径系数 c','g-c',-2,2,.05,.5):''}</div><div id="g-view"></div><div class="buttons"><button class="quiet" id="g-prev">← 上一步</button><button class="primary" id="g-next">下一步 →</button><span id="g-stage" class="lab-note"></span></div><div id="g-result" class="numeric-block"></div>`;
 function draw(){let x=value('g-x'),a=value('g-a'),c=paths?value('g-c'):0,u=paths?a*x:2*x,v=u*u,f=paths?v+c*x:.5*(v-a)**2;
  let labels=paths?[['x',x],['u=ax',u],['v=u²',v],['f=v+cx',f]]:[['w',x],['u=2w',u],['v=u²',v],['f=½(v−d)²',f]];
  let out='';labels.forEach(([s,v],i)=>{if(i)out+=line(45+(i-1)*148+40,95,45+i*148-40,95,step>=i?'edge active':'edge');out+=node(45+i*148,95,s,fmt(v),step===i);});
  if(paths)out+=`<path d="M45 128 V190 H489 V125" class="edge ${step===3?'active':''}"/>`+text(205,184,`直接路径：c = ${fmt(c)}`);
  $('#g-view').innerHTML=svg(out,540,paths?220:165,'计算图与反向链式求导');
  $('#g-stage').textContent=['先读输入','计算 u','计算 v','到达损失，再向回看'][step];
  $('#g-prev').disabled=step===0;$('#g-next').disabled=step===3;
  $('#g-result').textContent=paths?`f = (a x)² + c x = ${fmt(f)}\n路径 1：∂f/∂v × ∂v/∂u × ∂u/∂x = 1×${fmt(2*u)}×${fmt(a)} = ${fmt(2*u*a)}\n路径 2：∂(c x)/∂x = ${fmt(c)}\n两路相加：∂f/∂x = ${fmt(2*u*a+c)}`:`f = ${fmt(f)}\n∂f/∂v = v−d = ${fmt(v-a)}\n∂v/∂u = 2u = ${fmt(2*u)}；∂u/∂w = 2\n∂f/∂w = ${fmt(v-a)}×${fmt(2*u)}×2 = ${fmt((v-a)*2*u*2)}`;
 }$('#g-prev').onclick=()=>{step--;draw();};$('#g-next').onclick=()=>{step++;draw();};controls(draw);draw();
}
function backpropLab(lesson){
 let state={x:[.6,.2],W0:[[.1,-.2],[.4,.3]],W1:[[.2,-.3],[-.4,.1]],d:[1,0]},stage=lesson.id==='output'?3:lesson.id==='hidden'?4:0,updateNote='';
 let weights=[];for(let l=0;l<2;l++)for(let i=0;i<2;i++)for(let j=0;j<2;j++)weights.push([`${l},${i},${j}`,`w${l===0?'ᶫ⁼⁰':'ᶫ⁼¹'} [${i+1},${j+1}]`]);
 root.innerHTML=`<p class="lab-description">2 个输入 → 2 个 Sigmoid 隐藏节点 → 2 个 Sigmoid 输出。无偏置，平方误差，沿图逐步计算。点击连线也可选择权重。</p><div class="control-grid">${select('当前权重','b-edge',weights,lesson.id==='hidden'?'0,0,0':'1,0,0')}${range('权重数值','b-weight',-3,3,.01,lesson.id==='hidden'?.1:.2)}${range('输入 x₁','b-x1',0,1,.05,.6)}${range('输入 x₂','b-x2',0,1,.05,.2)}${range('学习率 λ','b-lr',.01,5,.01,.5)}${select('目标向量','b-target',[['0','d=(1,0)'],['1','d=(0,1)']],'0')}</div><div id="b-view"></div><div class="buttons"><button class="quiet small-button" id="b-prev">← 计算上一步</button><button class="primary small-button" id="b-next">计算下一步 →</button><button class="quiet small-button" id="b-step">更新全部权重一次</button></div><div id="b-stage" class="lab-note"></div><div id="b-result" class="numeric-block"></div><div id="b-check" class="selected-detail"></div><div id="b-update" class="lab-note" aria-live="polite"></div>`;
 function selected(){return $('#b-edge').value.split(',').map(Number);}
 function sync(){let[l,i,j]=selected();$('#b-weight').value=state['W'+l][i][j];outputs();}
 function draw(){let f=M.toyForward(state),[l,i,j]=selected(),grad=(l?f.g1:f.g0)[i][j];let out='';
  for(let q=0;q<2;q++)for(let a=0;a<2;a++)for(let b=0;b<2;b++){
   let active=q===l&&a===i&&b===j,x1=60+q*210,x2=x1+210,y1=65+a*130,y2=65+b*130;
   out+=line(x1+35,y1,x2-35,y2,active?'edge active':'edge',`data-edge="${q},${a},${b}" style="cursor:pointer" tabindex="0" role="button" aria-label="选择权重 ${q},${a+1},${b+1}"`);
   if(active)out+=text((x1+x2)/2,(y1+y2)/2-8,`w=${fmt(state['W'+q][a][b])}`,'value-text','text-anchor="middle"');
  }
  for(let a=0;a<2;a++)out+=node(60,65+a*130,`x${a+1}`,fmt(state.x[a]),stage===0)+node(270,65+a*130,`h${a+1}`,stage>=1?fmt(f.h[a]):'?',stage===1)+node(480,65+a*130,`y${a+1}`,stage>=2?fmt(f.y[a]):'?',stage===2);
  if(stage>=3)for(let a=0;a<2;a++)out+=text(443,105+a*130,`δ₂=${fmt(f.d2[a])}`);
  if(stage>=4)for(let a=0;a<2;a++)out+=text(233,105+a*130,`δ₁=${fmt(f.d1[a])}`);
  $('#b-view').innerHTML=svg(out,540,250,'可选权重的两层神经网络');
  $('#b-stage').textContent=['0 · 输入已知','1 · 加权和 → 隐藏激活','2 · 输出激活 → 损失','3 · 输出 δ=(y−d)y(1−y)','4 · 隐藏 δ=h(1−h)Σwδ','5 · 输入×下一层 δ = 权重梯度'][stage];
  $('#b-prev').disabled=stage===0;$('#b-next').disabled=stage===5;
  $('#b-result').textContent=`隐藏加权和 a₁ = [${f.a1.map(x=>fmt(x)).join(', ')}]\n隐藏输出 h = [${f.h.map(x=>fmt(x)).join(', ')}]\n预测 y = [${f.y.map(x=>fmt(x)).join(', ')}]；目标 d = [${state.d}]\n损失 L = ${fmt(f.loss,7)}\n当前权重梯度 ∂L/∂w = ${fmt(grad,7)}\n建议变化量 −λ∂L/∂w = ${fmt(-value('b-lr')*grad,7)}`;
  const eps=1e-5,plus=clone(state),minus=clone(state);plus['W'+l][i][j]+=eps;minus['W'+l][i][j]-=eps;
  const numeric=(M.toyForward(plus).loss-M.toyForward(minus).loss)/(2*eps);
  $('#b-check').textContent=`另一条计算路线：中心差分 = ${fmt(numeric,8)}；与解析梯度绝对差 ${Math.abs(numeric-grad).toExponential(2)}。差分只扰动当前这一个权重。`;
  $('#b-update').textContent=updateNote;
  $$('[data-edge]',root).forEach(e=>{const pick=()=>{$('#b-edge').value=e.dataset.edge;sync();draw();};e.onclick=pick;e.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();pick();}};});
 }controls(e=>{if(e.id==='b-edge')sync();else if(e.id==='b-weight'){let[l,i,j]=selected();state['W'+l][i][j]=+e.value;}else if(e.id==='b-x1')state.x[0]=+e.value;else if(e.id==='b-x2')state.x[1]=+e.value;else if(e.id==='b-target')state.d=+e.value?[0,1]:[1,0];updateNote='';draw();});
 $('#b-prev').onclick=()=>{stage--;draw();};$('#b-next').onclick=()=>{stage++;draw();};$('#b-step').onclick=()=>{let before=M.toyForward(state).loss;state=M.toyStep(state,value('b-lr'));stage=5;let after=M.toyForward(state).loss;updateNote=`统一更新一次：损失 ${fmt(before,6)} → ${fmt(after,6)}（${after<before?'下降':'未下降，可试更小学习率'}）。`;sync();draw();};sync();draw();
}
function batchLab(){
 root.innerHTML=`<p class="lab-description">每行一个样本。选定 W₀[1,1]，查看每个样本对它的梯度贡献。损失取样本平均。</p><div class="control-grid">${range('batch 大小 B','ba-size',1,6,1,3)}</div><div id="ba-view"></div>`;
 function draw(){let B=value('ba-size'),xs=Array.from({length:B},(_,n)=>[.2+.1*n,.7-.08*n]),results=xs.map((x,n)=>M.toyForward({x,W0:[[.1,-.2],[.4,.3]],W1:[[.2,-.3],[-.4,.1]],d:n%2?[0,1]:[1,0]}));
 let ds=results.map(f=>f.d1.map(x=>x/B)),g=sum(results.map((f,n)=>xs[n][0]*f.d1[0]))/B;
 $('#ba-view').innerHTML=`<div class="matrix-row">${matrix(xs,`X：${B}×2`)}<span>→</span>${matrix(results.map(f=>f.h),`H：${B}×2`)}</div><div class="matrix-row">${matrix(xs[0].map((_,j)=>xs.map(x=>x[j])),'Xᵀ：2×'+B)}<span>×</span>${matrix(ds,'Δ₁：'+B+'×2')}<span>=</span>${matrix([[g,sum(xs.map((x,n)=>x[0]*ds[n][1]))],[sum(xs.map((x,n)=>x[1]*ds[n][0])),sum(xs.map((x,n)=>x[1]*ds[n][1]))]],'∇W₀：2×2')}</div><div class="numeric-block">∂L/∂W₀[1,1] = Σₙ X[n,1] Δ₁[n,1]\n${xs.map((x,n)=>`${fmt(x[0],2)} × (${fmt(ds[n][0],6)})`).join(' + ')}\n= ${fmt(g,7)}\n这里 Δ 已经除过 B，不再重复除。</div>`;
 }controls(draw);draw();
}
function softmaxLab(){
 root.innerHTML=`<p class="lab-description">三个类别就足以看清概率耦合。温度 T 改变概率的尖锐程度；对原始 logit 的导数还要除以 T。</p><div class="control-grid">${range('得分 z₁','s-z1',-4,5,.1,1.2)}${range('得分 z₂','s-z2',-4,5,.1,-.3)}${range('得分 z₃','s-z3',-4,5,.1,.8)}${range('温度 T','s-temp',.2,3,.1,1)}${select('真实类别','s-target',[['0','类别 1'],['1','类别 2'],['2','类别 3']],'2')}</div><div id="s-view"></div><div id="s-result" class="numeric-block"></div>`;
 function draw(){let logits=[value('s-z1'),value('s-z2'),value('s-z3')],target=value('s-target'),T=value('s-temp'),r=M.softmaxLossGrad(logits,target,T),out='';
  for(let i=0;i<3;i++){let y=32+i*62;out+=text(5,y+15,`类别 ${i+1}${i===target?' ✓':''}`)+`<rect x="100" y="${y}" width="${r.p[i]*340}" height="25" fill="${i===target?C.orange:C.blue}"/>`+text(100+r.p[i]*340+8,y+17,(r.p[i]*100).toFixed(1)+'%');}
  $('#s-view').innerHTML=svg(out,540,215,'三个类别的Softmax概率');$('#s-result').textContent=`概率和 = ${fmt(sum(r.p),8)}\n交叉熵 L = −log p真实类 = ${fmt(r.loss,6)}\n∂L/∂z = (p−d)/T\n= [${r.grad.map(x=>fmt(x,6)).join(', ')}]\n负梯度会提高真实类得分、降低其他类得分。`;
 }controls(draw);draw();
}
function gradientLab(lesson){
 root.innerHTML=`<p class="lab-description">可手算的标量链：hₗ₊₁=s(g·hₗ+b)。这里 g 是每层相同的标量权重，不是随机矩阵初始化的方差。局部导数包含 g。</p><div class="control-grid">${select('激活函数','v-act',[['sigmoid','Sigmoid'],['relu','ReLU'],['tanh','Tanh'],['linear','线性']],'sigmoid')}${range('层数 L','v-depth',1,30,1,12)}${range('每层权重 g','v-gain',0,3,.05,1)}${range('每层偏置 b','v-bias',-2,2,.1,0)}${range('输入 h₀','v-input',-2,2,.1,.6)}</div><div id="v-view"></div><div id="v-result" class="numeric-block"></div>`;
 function draw(){let act=$('#v-act').value,L=value('v-depth'),g=value('v-gain'),b=value('v-bias'),h=value('v-input'),values=[[0,h]],local=[],back=1,grads=[];
  for(let k=0;k<L;k++){let z=g*h+b;h=M.activate(z,act);local.push(g*M.derivative(z,h,act));values.push([k+1,h]);}
  for(let k=L-1;k>=0;k--){back*=local[k];grads.unshift([k,back]);}
  let logs=grads.map(([k,v])=>[k,v===0?-16:Math.max(-16,Math.log10(Math.abs(v)))]);
  $('#v-view').innerHTML=plot([{points:values,color:C.blue,dots:true}],{xLabel:'层号',yLabel:'前向数值 h',height:190})+plot([{points:logs,color:C.orange,dots:true}],{xLabel:'反传回到的层号',yLabel:'log₁₀ |∂hL/∂h层|',height:205,yMax:Math.max(0,...logs.map(p=>p[1]))});
  $('#v-result').textContent=`各层局部导数（前 8 个）：[${local.slice(0,8).map(x=>fmt(x,3)).join(', ')}${L>8?', …':''}]\n从末端反传回输入的导数 = ${fmt(back,7)}\n图的 −16 位置合并表示零或小于 10⁻¹⁶ 的量，并不把它们当作相同的精确值。`;
 }controls(draw);draw();
}
function dataset(kind,count,seed){let rng=M.makeRng(seed),xs=[],ys=[];for(let i=0;i<count;i++){let x=(rng()*2-1),y=(rng()*2-1);xs.push([x,y]);ys.push(kind==='circles'?(x*x+y*y>.45?1:0):x*y>0?1:0);}return{xs,ys};}
function trainLab(lesson){
 let net,train,valid,history=[],epoch=0,running=false,frame,disposed=false;
 root.innerHTML=`<p class="lab-description">在浏览器里真实训练一个小网络。背景是模型概率，实心点是训练样本，空心点是独立生成的验证样本；验证点不参与更新。</p><div class="control-grid">${select('数据形状','t-data',[['xor','XOR 四象限'],['circles','内外圆']],'xor')}${range('隐藏层数','t-depth',0,4,1,2)}${range('每层节点数','t-width',2,12,1,6)}${select('隐藏激活','t-act',[['tanh','Tanh'],['sigmoid','Sigmoid'],['relu','ReLU'],['linear','线性']],'tanh')}${range('学习率','t-lr',.001,.15,.001,.03)}${select('优化器','t-opt',[['adam','Adam'],['sgd','SGD'],['momentum','Momentum']],'adam')}</div><div class="buttons"><button class="primary" id="t-play">开始训练</button><button class="quiet" id="t-step">训练 20 步</button><button class="quiet" id="t-reset">按当前设置重新初始化</button></div><div id="t-status" class="train-status" aria-live="polite"></div><canvas id="t-canvas" class="decision-canvas" width="420" height="420" role="img" aria-label="实时分类边界，横轴x1纵轴x2，范围均为负1到1"></canvas><div id="t-curve"></div><div id="t-metrics"></div><div id="t-note" class="lab-note"></div>`;
 function errors(data){let loss=0,err=0;data.xs.forEach((x,i)=>{let r=M.softmaxLossGrad(net.forward(x).output,data.ys[i]);loss+=r.loss;if((r.p[1]>.5?1:0)!==data.ys[i])err++;});return{loss:loss/data.xs.length,error:err/data.xs.length};}
 function draw(){const can=$('#t-canvas'),ctx=can.getContext('2d'),N=32,cell=360/N;ctx.clearRect(0,0,420,420);
  for(let j=0;j<N;j++)for(let i=0;i<N;i++){let p=net.predict([-1+2*(i+.5)/N,1-2*(j+.5)/N])[1];let lo=[226,237,241],hi=[242,218,201];ctx.fillStyle=`rgb(${lo.map((v,k)=>Math.round(v*(1-p)+hi[k]*p)).join(',')})`;ctx.fillRect(35+i*cell,15+j*cell,cell+1,cell+1);}
  ctx.strokeStyle='#71838b';ctx.strokeRect(35,15,360,360);ctx.fillStyle=C.muted;ctx.font='12px Segoe UI';ctx.fillText('−1',27,395);ctx.fillText('1',390,395);ctx.fillText('x₁',208,413);ctx.fillText('x₂',8,195);ctx.fillText('1',12,25);ctx.fillText('−1',9,375);
  for(const [data,fill] of [[train,true],[valid,false]])for(let i=0;i<data.xs.length;i++){let[x,y]=data.xs[i];ctx.beginPath();ctx.arc(35+(x+1)*180,15+(1-y)*180,fill?3.8:3,0,Math.PI*2);ctx.fillStyle=data.ys[i]?C.orange:C.blue;ctx.strokeStyle=ctx.fillStyle;if(fill)ctx.fill();else{ctx.lineWidth=1.2;ctx.stroke();}}
  const a=errors(train),b=errors(valid);$('#t-metrics').innerHTML=stats([['更新步',epoch],['训练错率',(a.error*100).toFixed(1)+'%'],['验证错率',(b.error*100).toFixed(1)+'%']]);
  $('#t-note').textContent=`参数 ${net.parameterCount()} 个；每步使用全部 ${train.xs.length} 个训练样本的平均梯度。深度为 0 时是线性 Softmax 分类器。动画数据是教学合成数据，不是 MNIST。`;
  if(!history.length||history.at(-1).epoch!==epoch)history.push({epoch,train:a.loss,valid:b.loss});
  $('#t-curve').innerHTML=legend([['训练交叉熵',C.blue],['验证交叉熵',C.orange]])+plot([{points:history.map(r=>[r.epoch,r.train]),color:C.blue},{points:history.map(r=>[r.epoch,r.valid]),color:C.orange}],{height:185,xLabel:'参数更新步',yLabel:'交叉熵'});
 }
 function stop(message){running=false;cancelAnimationFrame(frame);$('#t-play').textContent='继续训练';$('#t-status').textContent=message||`已暂停在第 ${epoch} 步。`;}
 function step(count){try{for(let k=0;k<count;k++){let r=net.lossGrad(train.xs,train.ys);net.step(r.grads,value('t-lr'),$('#t-opt').value);epoch++;if(!Number.isFinite(r.loss)||r.loss>1e6)throw new Error('损失过大');}draw();}catch(e){stop('数值不稳定，请减小学习率再重新初始化。');}}
 function loop(){if(!running||disposed)return;step(8);if(epoch>=1600){stop('已完成 1600 步，可改设置再比较。');return;}frame=requestAnimationFrame(loop);}
 function reset(){stop();train=dataset($('#t-data').value,100,23);valid=dataset($('#t-data').value,60,97);net=new M.DenseNet({input:2,width:value('t-width'),depth:value('t-depth'),output:2,activation:$('#t-act').value,seed:42});epoch=0;history=[];$('#t-status').textContent='已初始化。先预测边界能否把样本分开。';$('#t-play').textContent='开始训练';draw();}
 controls(e=>{if(['t-lr','t-opt'].includes(e.id)){stop('优化设置已改变；点击继续训练，或重新初始化做对照。');}else reset();});
 $('#t-play').onclick=()=>{if(running)stop();else{running=true;$('#t-play').textContent='暂停';$('#t-status').textContent='正在训练；切换章节会自动暂停。';loop();}};$('#t-step').onclick=()=>{stop();step(20);$('#t-status').textContent=`已训练到第 ${epoch} 步。`;};$('#t-reset').onclick=reset;cleanup=()=>{disposed=true;cancelAnimationFrame(frame);};reset();
}
function optimizerLab(){
 root.innerHTML=`<p class="lab-description">同一个二次曲面 L=(0.1x²+c y²)/2，同一起点 (3,2)。调节两个方向的曲率差，比较优化器怎样使用梯度。</p><div class="control-grid">${range('共同学习率','o-lr',.01,.8,.01,.12)}${range('显示更新步数','o-steps',1,100,1,40)}${range('y 方向曲率 c','o-curvature',.1,10,.1,2)}${range('Momentum 动量 μ','o-momentum',0,.99,.01,.9)}</div><div id="o-view"></div><div id="o-result"></div>`;
 function draw(){let lr=value('o-lr'),steps=value('o-steps'),curvature=value('o-curvature'),momentum=value('o-momentum'),paths=[],report=[],colors=[C.blue,C.orange,C.green];
 for(const [idx,name] of ['SGD','Momentum','Adam'].entries()){let x=[3,2],m=[0,0],v=[0,0],points=[[...x]],bad=false;for(let t=1;t<=steps;t++){let g=[.1*x[0],curvature*x[1]];for(let j=0;j<2;j++){if(name==='SGD')x[j]-=lr*g[j];else if(name==='Momentum'){m[j]=momentum*m[j]+g[j];x[j]-=lr*m[j];}else{m[j]=.9*m[j]+.1*g[j];v[j]=.999*v[j]+.001*g[j]*g[j];x[j]-=lr*(m[j]/(1-.9**t))/(Math.sqrt(v[j]/(1-.999**t))+1e-8);}}if(!x.every(Number.isFinite)||Math.max(...x.map(Math.abs))>1e4){bad=true;break;}points.push([...x]);}paths.push({points:points.filter(p=>p.every(v=>Math.abs(v)<=4)),color:colors[idx]});report.push([name,fmt(.5*(.1*x[0]**2+curvature*x[1]**2),5),bad?'发散':`(${fmt(x[0],3)}, ${fmt(x[1],3)})`]);}
 $('#o-view').innerHTML=legend([['SGD',C.blue],['Momentum',C.orange],['Adam',C.green]])+plot(paths,{xMin:-4,xMax:4,yMin:-3,yMax:3,xLabel:'参数 x',yLabel:'参数 y',height:300,contours:[.1,curvature]});
 $('#o-result').innerHTML=`<table class="table-small"><tr><th>优化器</th><th>最终损失</th><th>最终位置</th></tr>${report.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')}</table><p class="lab-note">浅灰曲线表示等损失线，中心 (0,0) 为最小值。显示窗口为 x∈[−4,4]、y∈[−3,3]；跑出窗口的路径不画，但表中保留实际损失。同一学习率只是一种实验设置，不是最优超参数比较。</p>`;
 }controls(draw);draw();
}
function residualLab(){
 root.innerHTML=`<p class="lab-description">标量残差支路 F(x)=ax+b，捷径 κx。把“恒等捷径”“投影捷径”“加和后激活”拆开观察。</p><div class="control-grid">${range('输入 x','r-x',-2,2,.05,.8)}${range('残差系数 a','r-a',-2,2,.05,.1)}${range('残差偏置 b','r-b',-2,2,.05,0)}${range('捷径系数 κ','r-k',0,2,.05,1)}${select('加和后处理','r-post',[['identity','保持线性'],['relu','再接 ReLU']],'identity')}</div><div class="buttons"><button class="quiet small-button" id="r-zero">F=0，保留输入</button><button class="quiet small-button" id="r-cancel">两路导数抵消</button><button class="quiet small-button" id="r-block">ReLU 截断</button></div><div id="r-view"></div><div id="r-result" class="numeric-block"></div>`;
 function draw(){let x=value('r-x'),a=value('r-a'),b=value('r-b'),k=value('r-k'),post=$('#r-post').value,z=k*x+a*x+b,y=post==='relu'?Math.max(0,z):z,mask=post==='relu'?(z>0?1:0):1;
 let out=line(90,140,230,140)+line(310,140,420,140)+node(50,140,'输入',fmt(x))+node(270,140,'F(x)',fmt(a*x+b),true)+node(465,140,'加和输出',fmt(y))+`<path d="M50 106 V45 H465 V105" class="edge active"/>`+text(205,35,`捷径 κx = ${fmt(k*x)}`)+text(131,193,`主支路局部导数 a = ${fmt(a)}`);
 $('#r-view').innerHTML=svg(out,540,225,'主支路和捷径的前向与梯度')+stats([['捷径贡献',fmt(mask*k)],['主支路贡献',fmt(mask*a)],['总导数',fmt(mask*(k+a))]]);
 $('#r-result').textContent=`加和前 z = κx+F(x) = ${fmt(z)}\n${post==='relu'?`ReLU′(z) = ${mask}${z===0?'（零点按0处理）':''}`:'加和后无激活：导数为1'}\n∂y/∂x = ${post==='relu'?'ReLU′(z) × ':''}(κ+a) = ${fmt(mask*(k+a))}\n${k===1?'κ=1，捷径是恒等映射。':'κ≠1，这里是标量投影捷径。'} ${a===-k?'两路相加抵消，总梯度可以为0。':''}`;
 }controls(draw);function preset(a,k,x,post){$('#r-a').value=a;$('#r-k').value=k;$('#r-x').value=x;$('#r-b').value=0;$('#r-post').value=post;outputs();draw();}
 $('#r-zero').onclick=()=>preset(0,1,.8,'identity');$('#r-cancel').onclick=()=>preset(-1,1,.8,'identity');$('#r-block').onclick=()=>preset(.2,1,-1,'relu');draw();
}
function deepLab(lesson){
 let plain,res,history=[],iteration=0,frame,running=false,disposed=false;
 root.innerHTML=`<p class="lab-description">同一目标：f(x)=x+0.25sin(3x)。普通网络与等宽残差网络真实反向训练。残差模型含输入嵌入、L 个 h←h+0.3s(hW+b) 块及线性输出。</p><div class="control-grid">${range('普通隐藏层 / 残差块数 L','d-depth',1,12,1,6)}${range('隐藏宽度','d-width',3,12,1,6)}${range('训练样本数','d-count',5,45,1,45)}${select('激活函数','d-act',[['tanh','Tanh'],['sigmoid','Sigmoid'],['relu','ReLU']],'tanh')}${range('学习率','d-lr',.001,.05,.001,.01)}</div><div class="buttons"><button class="primary" id="d-play">开始对照训练</button><button class="quiet" id="d-step">各训练 20 步</button><button class="quiet" id="d-reset">重新初始化</button></div><div id="d-status" class="train-status" aria-live="polite"></div><div id="d-view"></div><div id="d-metrics"></div><div id="d-grad"></div><p class="lab-note">相同训练样本、更新步数和优化器；虚线使用 44 个未训练的区间中点。样本数改变会重新初始化两种模型，比较时一次只改变一个设置。残差模型额外有输入嵌入，参数量并不相等；0.3 是本实验的残差缩放。</p>`;
 const target=x=>x+.25*Math.sin(3*x),vx=Array.from({length:81},(_,i)=>[-1.2+i*2.4/80]),vy=vx.map(([x])=>target(x));
 const validX=Array.from({length:44},(_,i)=>[-1.2+(i+.5)*2.4/44]),validY=validX.map(([x])=>target(x));
 let xs=[],ys=[];
 function validationMSE(net){return sum(validX.map((x,i)=>(net.predict(x)-validY[i])**2))/validX.length;}
 function draw(){let a=plain.lossGrad(xs,ys),b=res.lossGrad(xs,ys),av=validationMSE(plain),bv=validationMSE(res);if(!history.length||history.at(-1).step!==iteration)history.push({step:iteration,a:2*a.loss,b:2*b.loss,av,bv});
  $('#d-view').innerHTML=legend([['目标函数','#26343c'],['普通网络',C.blue],['残差网络',C.orange]])+plot([{points:vx.map((x,i)=>[x[0],vy[i]]),color:'#26343c',dash:true},{points:vx.map(x=>[x[0],plain.predict(x)]),color:C.blue},{points:vx.map(x=>[x[0],res.predict(x)]),color:C.orange}],{height:245,xLabel:'输入 x',yLabel:'预测函数值'})+plot([{points:history.map(r=>[r.step,Math.log10(Math.max(1e-10,r.a))]),color:C.blue},{points:history.map(r=>[r.step,Math.log10(Math.max(1e-10,r.b))]),color:C.orange},{points:history.map(r=>[r.step,Math.log10(Math.max(1e-10,r.av))]),color:C.blue,dash:true},{points:history.map(r=>[r.step,Math.log10(Math.max(1e-10,r.bv))]),color:C.orange,dash:true}],{height:190,xLabel:'参数更新步',yLabel:'log₁₀ 均方误差'})+'<p class="lab-note">曲线颜色区分模型：实线为训练误差，虚线为区间内未参与训练的中点验证误差。</p>';
  $('#d-metrics').innerHTML=stats([['更新步',iteration],['普通训练 MSE',fmt(2*a.loss,6)],['残差训练 MSE',fmt(2*b.loss,6)],['普通验证 MSE',fmt(av,6)],['残差验证 MSE',fmt(bv,6)]]);
  let ag=a.layerGradRms||a.gradRms,bg=b.layerGradRms||b.gradRms;
  $('#d-grad').innerHTML=`<div class="numeric-block">普通参数 ${plain.parameterCount()}；残差参数 ${res.parameterCount()}\n层输出反传信号 RMS（含线性输出层；输入 → 输出）：\n普通 [${ag.map(x=>fmt(x,3)).join(', ')}]\n残差 [${bg.map(x=>fmt(x,3)).join(', ')}]</div>`;
 }
 function stop(msg){running=false;cancelAnimationFrame(frame);$('#d-play').textContent='继续对照训练';$('#d-status').textContent=msg||`已暂停在第 ${iteration} 步。`;}
 function step(n){try{for(let i=0;i<n;i++){let a=plain.lossGrad(xs,ys),b=res.lossGrad(xs,ys);plain.step(a.grads,value('d-lr'),'adam');res.step(b.grads,value('d-lr'),'adam');iteration++;if(!Number.isFinite(a.loss+b.loss)||a.loss+b.loss>1e7)throw new Error('unstable');}draw();}catch(e){stop('数值不稳定，请降低学习率或重新初始化。');}}
 function loop(){if(!running||disposed)return;step(3);if(iteration>=600){stop('已完成 600 步，可修改结构再比较。');return;}frame=requestAnimationFrame(loop);}
 function reset(){stop();let count=value('d-count');xs=Array.from({length:count},(_,i)=>[-1.2+Math.round(i*44/(count-1))*2.4/44]);ys=xs.map(([x])=>target(x));let config={input:1,width:value('d-width'),depth:value('d-depth'),output:1,activation:$('#d-act').value,seed:12,task:'regression'};plain=new M.DenseNet(config);res=new M.DenseNet({...config,residual:true});iteration=0;history=[];$('#d-play').textContent='开始对照训练';$('#d-status').textContent='已初始化。先看两个网络初始输出与梯度尺度。';draw();}
 controls(e=>{if(e.id==='d-lr')stop('学习率已改变。');else reset();});$('#d-play').onclick=()=>{if(running)stop();else{running=true;$('#d-play').textContent='暂停';$('#d-status').textContent='两个网络正在使用相同样本与更新次数训练。';loop();}};$('#d-step').onclick=()=>{stop();step(20);$('#d-status').textContent=`两个模型均已更新 ${iteration} 步。`;};$('#d-reset').onclick=reset;cleanup=()=>{disposed=true;cancelAnimationFrame(frame);};reset();
}
function convLab(){
 let kernel=[[0,-1,0],[-1,4,-1],[0,-1,0]];
 const initialImage=()=>Array.from({length:7},(_,y)=>Array.from({length:7},(_,x)=>x===3||y===3?1:0));
 let image=initialImage();
 const cells=Array.from({length:9},(_,i)=>[String(i),`第 ${Math.floor(i/3)+1} 行，第 ${i%3+1} 列`]);
 root.innerHTML=`<p class="lab-description">选中输出位置，查看它实际读取的输入窗口；点击输入像素可画出自己的图形。修改一个核权重，观察共享核如何同时改变整张输出。这里采用深度学习常用的互相关约定，核不翻转。</p>
 <div class="control-grid">${select('卷积核预设','c-kind',[['edge','边缘'],['blur','均值平滑'],['identity','中心取值']],'edge')}${select('调整哪个核权重','c-k-index',cells,'4')}${range('所选核权重','c-k-value',-5,5,.01,4)}${range('步长 stride','c-stride',1,2,1,1)}${range('零填充 padding','c-pad',0,1,1,0)}${range('输出行（从 0 开始）','c-row',0,4,1,0)}${range('输出列（从 0 开始）','c-col',0,4,1,0)}</div>
 <div class="buttons"><button type="button" class="quiet" id="c-reset-image">恢复十字输入</button><button type="button" class="quiet" id="c-clear-image">清空输入</button></div>
 <p class="lab-note">点击或用键盘激活像素，切换 0 / 1；蓝框表示当前窗口读取到的有效输入，边界外用零填充。均值核使用精确的 1/9；移动权重滑块后该权重按 0.01 步长改变。</p><div id="c-view"></div><div id="c-result" class="numeric-block" aria-live="polite"></div>`;
 function pixelTable(row,col,stride,pad){
  const y0=row*stride-pad,x0=col*stride-pad;
  return `<table class="matrix"><caption>可编辑输入 7×7；蓝框为当前窗口</caption><tbody>${image.map((line,y)=>'<tr>'+line.map((v,x)=>{
   let active=y>=y0&&y<y0+3&&x>=x0&&x<x0+3;
   return `<td class="${active?'selected':''}" style="padding:2px"><button type="button" data-c-pixel="${y},${x}" aria-label="输入第 ${y+1} 行第 ${x+1} 列，当前 ${v}，点击切换" aria-pressed="${v===1}" style="min-width:28px;min-height:28px;padding:2px 4px;border:0;border-radius:3px;background:${v?'#2c647d':'transparent'};color:${v?'white':'#405b67'}">${v}</button></td>`;
  }).join('')+'</tr>').join('')}</tbody></table>`;
 }
 function syncKernelControl(){let i=+$('#c-k-index').value;$('#c-k-value').value=kernel[Math.floor(i/3)][i%3];outputs();}
 function draw(){let stride=value('c-stride'),pad=value('c-pad'),r=M.convolution(image,kernel,stride,pad);
  for(const id of ['c-row','c-col']){$('#'+id).max=r.length-1;if(value(id)>=r.length)$('#'+id).value=r.length-1;}outputs();
  let row=value('c-row'),col=value('c-col'),ky=Math.floor(+$('#c-k-index').value/3),kx=(+$('#c-k-index').value)%3;
  $('#c-k-value-out').textContent=fmt(kernel[ky][kx],5);
  let patch=kernel.map((ks,y)=>ks.map((_,x)=>image[row*stride+y-pad]?.[col*stride+x-pad]??0)),products=patch.map((p,y)=>p.map((v,x)=>v*kernel[y][x]));
  let iy=row*stride+ky-pad,ix=col*stride+kx-pad,inside=iy>=0&&iy<7&&ix>=0&&ix<7;
  $('#c-view').innerHTML=`<div class="matrix-row">${pixelTable(row,col,stride,pad)}${matrix(kernel,'共享核 3×3')}</div><div class="matrix-row">${matrix(patch,'当前局部窗口')}<span>⊙</span>${matrix(kernel,'共享卷积核')}<span>→</span>${matrix([[r[row][col]]],'求和输出')}</div><div class="matrix-row">${matrix(r,`完整输出 ${r.length}×${r[0].length}`)}</div>`;
  $('#c-result').textContent=`输出大小 = floor((7 + 2×${pad} − 3)/${stride}) + 1 = ${r.length}\n当前位置的逐元素乘积：\n${products.map(p=>p.map(x=>fmt(x,3)).join('  ')).join('\n')}\n求和 = ${fmt(sum(products.flat()),5)}\n所选核格 (${ky+1}, ${kx+1}) 读取 ${inside?`输入 (${iy+1}, ${ix+1})`:'边界外的补零'}，其当前乘积 ${fmt(patch[ky][kx],3)} × ${fmt(kernel[ky][kx],3)} = ${fmt(products[ky][kx],5)}。\n仅对当前输出求导：∂Y/∂K = 对应输入 = ${fmt(patch[ky][kx],3)}。完整损失对共享 K 的梯度还要累加所有输出位置的贡献。`;
  $$('[data-c-pixel]',root).forEach(button=>button.onclick=()=>{let [y,x]=button.dataset.cPixel.split(',').map(Number);image[y][x]=1-image[y][x];draw();root.querySelector(`[data-c-pixel="${y},${x}"]`).focus();});
 }
 controls(e=>{if(e.id==='c-kind'){
   kernel=e.value==='edge'?[[0,-1,0],[-1,4,-1],[0,-1,0]]:e.value==='blur'?Array.from({length:3},()=>[1/9,1/9,1/9]):[[0,0,0],[0,1,0],[0,0,0]];syncKernelControl();
  }else if(e.id==='c-k-index')syncKernelControl();
  else if(e.id==='c-k-value'){let i=+$('#c-k-index').value;kernel[Math.floor(i/3)][i%3]=+e.value;}
  draw();});
 $('#c-reset-image').onclick=()=>{image=initialImage();draw();};$('#c-clear-image').onclick=()=>{image=Array.from({length:7},()=>Array(7).fill(0));draw();};draw();
}
function regularizeLab(){
 let seed=1;const data=[[1,3,2],[2,5,4],[3,7,6],[4,9,8]],features=[.2,.5,.8,1.1,.9,.4,.7,1.3];
 root.innerHTML=`<p class="lab-description">先观察批归一化如何按特征计算均值与方差，再用 γ 和 β 调整输出；下方独立实验倒置 Dropout。预测阶段使用示例中预先固定的统计量。</p><div class="control-grid">${range('Dropout 丢弃概率 p','z-drop',0,.8,.1,.5)}${select('运行阶段','z-mode',[['train','训练'],['eval','预测']],'train')}${range('BN 缩放 γ','z-gamma',-2,2,.1,1)}${range('BN 平移 β','z-beta',-2,2,.1,0)}${range('当前 batch 整体平移','z-shift',-3,3,.5,0)}${range('仅第一样本的第一特征变化','z-first',-3,3,.5,0)}</div><div class="buttons"><button class="quiet" id="z-mask">换一次随机掩码</button><button class="quiet" id="z-average">计算 200 次掩码平均</button></div><div id="z-view"></div><div id="z-result" class="numeric-block" aria-live="polite"></div><div id="z-mean" class="lab-note" aria-live="polite"></div>`;
 function draw(){let q=value('z-drop'),isTrain=$('#z-mode').value==='train',shift=value('z-shift'),first=value('z-first'),gamma=value('z-gamma'),beta=value('z-beta'),batch=data.map((row,n)=>row.map((v,j)=>v+shift+(n===0&&j===0?first:0))),base=M.batchNorm(data),current=M.batchNorm(batch),mean=isTrain?current.mean:base.mean,variance=isTrain?current.variance:base.variance,normalized=batch.map(row=>row.map((v,j)=>gamma*(v-mean[j])/Math.sqrt(variance[j]+1e-5)+beta)),drop=isTrain?M.dropout(features,q,seed):{values:features.slice(),mask:features.map(()=>1)};
 $('#z-mask').disabled=!isTrain;$('#z-average').disabled=!isTrain;
 $('#z-view').innerHTML=`<div class="matrix-row">${matrix(batch,'输入 batch：4样本×3特征')}<span>→</span>${matrix(normalized,isTrain?'训练 BN：按列归一化并缩放平移':'预测 BN：使用固定统计')}</div><div class="numeric-block">均值 μ=[${mean.map(x=>fmt(x,3)).join(', ')}]\n方差 σ²=[${variance.map(x=>fmt(x,3)).join(', ')}]</div><p class="canvas-caption">Dropout 前后，同一组 8 个特征</p>`+plot([{points:features.map((v,i)=>[i+1,v]),color:C.blue,dots:true},{points:drop.values.map((v,i)=>[i+1,v]),color:C.orange,dots:true}],{height:200,xLabel:'特征编号',yLabel:'特征数值'})+legend([['原始特征',C.blue],['Dropout 输出',C.orange]]);
 $('#z-result').textContent=`第一样本的第一特征：\nŷ = γ × (x − μ) / √(σ² + ε) + β\n  = ${fmt(gamma,2)} × (${fmt(batch[0][0],3)} − ${fmt(mean[0],3)}) / √(${fmt(variance[0],3)} + 0.00001) + (${fmt(beta,2)}) = ${fmt(normalized[0][0],4)}\n每列各自计算 μ 和 σ²，ε 防止除以零。γ、β 是同一特征内共享的可学习参数。\n${isTrain?'训练时使用当前 batch 的统计量；单独修改一个样本也会改变其他样本的归一化结果。':'预测时使用固定统计量；当前 batch 不再重新估计均值方差。'}\n\n${isTrain?'训练时保留特征乘以 1/(1−p)，在随机掩码的期望上保持原值。':'预测时关闭 Dropout，所有特征按原值传递。'}\n${isTrain?'缩放掩码 m/(1−p)':'预测阶段的乘数'}=[${drop.mask.map(x=>fmt(x,2)).join(', ')}]\n当前保留 ${drop.mask.filter(x=>x>0).length}/8 个。单次掩码平均不必等于原值。`;
 $('#z-mean').textContent='';
 }controls(draw);$('#z-mask').onclick=()=>{seed++;draw();};$('#z-average').onclick=()=>{let means=features.map(()=>0),p=value('z-drop');for(let i=0;i<200;i++){let r=M.dropout(features,p,500+i);r.values.forEach((v,j)=>means[j]+=v/200);}$('#z-mean').textContent=`固定输入，训练掩码独立采样 200 次：平均输出 [${means.map(v=>fmt(v,3)).join(', ')}]；原值 [${features.join(', ')}]。有限次平均通常接近原值，不要求逐项完全相等。`;};draw();
}
let initial=lessons.findIndex(l=>l.id===lessonId(location.hash.slice(1)||progress.last));renderLesson(initial<0?0:initial);
})();
