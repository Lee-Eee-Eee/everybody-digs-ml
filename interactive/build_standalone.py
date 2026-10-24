"""Embed this course's scripts, styles and MathJax for offline use."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def main():
    html = (ROOT/'index.html').read_text(encoding='utf-8')
    for filename in re.findall(r'<link rel="stylesheet" href="([^"]+)">', html):
        html = html.replace(f'<link rel="stylesheet" href="{filename}">', '<style>\n'+(ROOT/filename).read_text(encoding='utf-8')+'\n</style>')
    for filename in re.findall(r'<script defer src="([^"]+)"></script>', html):
        source = (ROOT/filename).read_text(encoding='utf-8').replace('</script','<\\/script')
        if filename == 'app.js':
            source = "document.addEventListener('DOMContentLoaded',function(){\n"+source+'\n});'
        html = html.replace(f'<script defer src="{filename}"></script>', '<script>\n'+source+'\n</script>')
    assert not re.search(r'<script[^>]+src=', html)
    notice = '<script type="text/plain" id="mathjax-license">\n'+(ROOT/'vendor/MathJax-LICENSE.txt').read_text(encoding='utf-8')+'\n</script>'
    html = html.replace('</body>',notice+'\n</body>')
    path = ROOT/'neural-network-lab.html'
    path.write_text(html,encoding='utf-8')
    print(path.name, path.stat().st_size, 'bytes')

if __name__ == '__main__':
    main()
