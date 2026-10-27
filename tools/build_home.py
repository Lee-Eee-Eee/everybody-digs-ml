"""Create the web entry point from the lesson manifest."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/Lee-Eee-Eee/neural-networks-from-zero'

def main():
    lessons = json.loads((ROOT/'course_manifest.json').read_text(encoding='utf-8'))
    rows = ''.join(f'<a class="lesson" href="{REPO}/blob/main/{escape(row["path"])}"><span>{escape(row["slug"][:2])}</span>{escape(row["title"][3:])}</a>' for row in lessons)
    exercise_count = sum(row['exercises'] for row in lessons)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>从Python到神经网络</title>
<style>:root{color:#243740;background:#f7f6f1;font:17px/1.85 Arial,"Microsoft YaHei",sans-serif}*{box-sizing:border-box}body{margin:0}main{max-width:1060px;margin:auto;padding:58px 30px}a{color:#285c72;text-underline-offset:4px}h1{font-size:clamp(30px,5vw,49px);line-height:1.35;font-weight:650;margin:12px 0 25px}h2{font-size:23px;margin:42px 0 17px}p{max-width:790px}.lead{font-size:19px}.label{font-size:13px;color:#63777c;letter-spacing:.09em}.actions{display:flex;gap:14px;flex-wrap:wrap;margin:30px 0 36px}.actions a{padding:11px 21px;border:1px solid #285c72;border-radius:5px;text-decoration:none}.actions a:first-child{background:#285c72;color:white}.route{display:grid;grid-template-columns:1fr 1fr;gap:10px}.lesson{display:flex;align-items:baseline;gap:15px;padding:13px 17px;border:1px solid #dce2de;background:#fff;text-decoration:none;border-radius:4px;font-size:15px}.lesson span{font:15px Georgia,serif;color:#829197}.lesson:hover{border-color:#285c72}.note{color:#63777c;font-size:14px;margin-top:33px}code{font-size:.9em;background:#eef0ec;padding:2px 5px}@media(max-width:700px){main{padding:34px 19px}.route{grid-template-columns:1fr}}</style>
<main><div class="label">动手学习 · PYTHON / PYTORCH</div><h1>从几条读数，走到会学习的网络</h1>
<p class="lead">一张手写数字先变成一张数值表，一段文本先变成一串编号。模型怎样从这些数里作判断，又怎样从错误中改变自己？</p>
<p>在这里，计算会从一个具体例子展开。点一条边、改一个数，观察输出与梯度，再回到学习本，把它写成自己的代码。</p>
<div class="actions"><a href="interactive/index.html#network">打开交互学习室 →</a><a href="REPO#开始学习">学习本与运行方法</a></div>
<h2>先把一个小网络看清</h2><p>交互室从同一个小网络逐步展开：读取输入、计算预测、沿边反传、更新权重。接着观察真实训练、卷积与残差。每个显示的数值都在浏览器重新计算。</p>
<h2>再将这些动作写成程序</h2><p>00–07连成基础路线。随后进入图像、记忆、注意力、图与连续时间网络，最后做一项能解释的小研究。点击下列课节可在GitHub阅读学习本；下载后可以直接运行和填写。</p>
<div class="route">ROWS</div><p class="note">学习本含EXERCISE_COUNT个填写练习；交互室含18节实验。<a href="REPO">查看源码与参考材料</a>。</p></main></html>'''.replace('REPO',REPO).replace('ROWS',rows).replace('EXERCISE_COUNT', str(exercise_count))
    (ROOT/'index.html').write_text(html,encoding='utf-8')

if __name__=='__main__':
    main()
