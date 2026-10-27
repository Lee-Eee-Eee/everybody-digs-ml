"""Execute course notebooks with saved outputs, or verify all reference answers."""
from pathlib import Path
import os
import atexit
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import ast
import json
import socket
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]


def execute(source,answers=False):
    notebook=nbformat.read(source,4)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            if answers and 'exercise' in cell.metadata.get('tags',[]):
                cell.source=cell.metadata['starter_source']+'\n\n'+cell.metadata['reference_solution']
            cell.outputs=[];cell.execution_count=None
    if answers:
        notebook.cells.append(nbformat.v4.new_code_cell(
            "from support import check_results\n"
            "unpassed=[(name,status) for name,status in check_results if status != 'passed']\n"
            "assert not unpassed, unpassed\nprint('Verified checks:',len(check_results))"))
    target=ROOT/'.build'/'reference_answers'/source.name if answers else source
    target.parent.mkdir(parents=True,exist_ok=True)
    kernel_root=ROOT/'.build'/'kernels';kernel_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lesson-',dir=kernel_root) as temp:
        spec=Path(temp)/'course';spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-X','utf8','-B','-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Neural networks course','language':'python'}),encoding='utf-8')
        sockets=[socket.socket() for _ in range(5)]
        for connection in sockets:connection.bind(('127.0.0.1',0))
        ports=[connection.getsockname()[1] for connection in sockets]
        for connection in sockets:connection.close()
        manager=KernelManager(kernel_name='course',kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]),
                              ip='127.0.0.1',shell_port=ports[0],iopub_port=ports[1],stdin_port=ports[2],control_port=ports[3],hb_port=ports[4])
        client=NotebookClient(notebook,km=manager,timeout=600,startup_timeout=120,
                              resources={'metadata':{'path':str(source.parent)}})
        try:
            client.execute()
        except Exception:
            failed=ROOT/'.build'/'failures'/('answers_' if answers else 'student_')
            failed.mkdir(parents=True,exist_ok=True)
            nbformat.write(notebook,failed/source.name)
            raise
        finally:
            if client.km is not None:
                client._cleanup_kernel()
            atexit.unregister(client._cleanup_kernel)
    nbformat.validate(notebook);nbformat.write(notebook,target)
    images=sum('image/png' in o.get('data',{}) for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
    return dict(lesson=source.stem,mode='answers' if answers else 'student',cells=len(notebook.cells),figures=images,status='passed')


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers',action='store_true')
    parser.add_argument('--pattern',default='*.ipynb')
    parser.add_argument('--jobs',type=int,default=1)
    args=parser.parse_args()
    sources=sorted((ROOT/'lessons').rglob(args.pattern))
    if not sources:parser.error('No matching lesson notebooks.')
    results=[];failures=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(execute,path,args.answers):path for path in sources}
        for future in as_completed(futures):
            try:
                row=future.result();results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            except Exception as error:
                path=futures[future]
                failures.append(dict(lesson=path.stem,error=str(error)))
                print('FAILED '+path.stem+'\n'+str(error),flush=True)
    report=ROOT/'.build'/('answer_validation.json' if args.answers else 'execution_validation.json')
    report.write_text(json.dumps(dict(results=results,failures=failures),ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
