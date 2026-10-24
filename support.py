"""Exercise feedback and figure styling; model code stays in the lessons."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
check_results = []


def _array(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def check(name, got, expected, atol=1e-6):
    import numpy as np
    if got is None or got is Ellipsis:
        check_results.append((name, 'pending'))
        print(f'待填写：{name}')
        return
    def equal(a, b):
        if isinstance(b, dict):
            return isinstance(a, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in b)
        if isinstance(b, str):
            return isinstance(a, str) and a == b
        actual, target = _array(a), _array(b)
        if actual.shape != target.shape:
            return False
        if actual.dtype.kind in 'OUS' or target.dtype.kind in 'OUS':
            return bool(np.array_equal(actual, target))
        return bool(np.allclose(actual, target, atol=atol, rtol=1e-5))
    try:
        ok = equal(got, expected)
    except (TypeError, ValueError, AttributeError):
        ok = False
    check_results.append((name, 'passed' if ok else 'failed'))
    print(('通过：' if ok else '还需修改：') + name)
    if not ok:
        print('你的结果：', got, '\n对照结果：', expected)


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'figure.dpi': 110, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'mathtext.fallback': 'stix',
    })


def show(fig):
    import matplotlib.pyplot as plt
    from IPython.display import display
    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')
    display(fig)
    plt.close(fig)


def block_diagram():
    """The explicit Pre-LN block used in lesson 13, with both addition ports."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    fig,ax=plt.subplots(figsize=(11.6,3.4),layout='constrained')
    ax.set(xlim=(-.2,12),ylim=(-.3,3));ax.axis('off')
    def box(x,y,width,title,detail):
        ax.add_patch(FancyBboxPatch((x-width/2,y-.36),width,.72,boxstyle='round,pad=.06',
                                  fc='#EDF3F4',ec='#37798A',lw=1.2))
        ax.text(x,y+.10,title,ha='center',va='center',fontsize=11)
        ax.text(x,y-.16,detail,ha='center',va='center',fontsize=9)
    def arrow(a,b,curve='arc3,rad=0',color='#496773'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.25,
                                    connectionstyle=curve,color=color))
    for x in [5.,10.3]:
        ax.add_patch(Circle((x,2),.19,fc='white',ec='#496773',lw=1.4))
        ax.text(x,2,r'$\Sigma$',ha='center',va='center',fontsize=14)
    box(1.9,.8,1.,'LayerNorm','over C')
    box(3.7,.8,1.65,'Attention','Q, K, V; heads')
    box(6.8,.8,1.,'LayerNorm','over C')
    box(8.8,.8,1.65,'Feed-forward',r'$C\to4C\to C$')
    arrow((.1,2),(4.81,2));arrow((5.19,2),(10.11,2));arrow((10.49,2),(11.6,2))
    arrow((.8,2),(1.34,.8),'angle3,angleA=-90,angleB=0')
    arrow((2.46,.8),(2.815,.8))
    arrow((4.585,.8),(5.,1.81),'angle3,angleA=0,angleB=90')
    arrow((5.8,2),(6.24,.8),'angle3,angleA=-90,angleB=0')
    arrow((7.36,.8),(7.915,.8))
    arrow((9.685,.8),(10.3,1.81),'angle3,angleA=0,angleB=90')
    ax.text(.1,2.42,r'$H\ (B,T,C)$',fontsize=12)
    ax.text(5.25,2.42,r'$H+\mathrm{Attn}(\mathrm{LN}(H))$',fontsize=11)
    ax.text(10.65,2.42,'Output',fontsize=11)
    ax.text(2.6,2.16,'identity path',ha='center',fontsize=10)
    ax.text(8.,2.16,'identity path',ha='center',fontsize=10)
    ax.text(5.8,-.09,'Backward at each sum: send g to both inputs; add the returning path contributions.',
            ha='center',fontsize=10,color='#8E5135')
    ax.set_title('One Pre-LN Transformer block: every branch returns (B, T, C)',pad=5)
    return fig
