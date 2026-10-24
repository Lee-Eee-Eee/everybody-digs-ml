"""Check saved lesson structure, executable syntax and local learning links."""
from pathlib import Path
import ast
import json
import re
from urllib.parse import unquote, urlsplit
import nbformat

ROOT=Path(__file__).resolve().parents[1]

def local_links(text,base):
    for target in re.findall(r'\]\(([^\s)]+)\)',text):
        parsed=urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        destination=(base/unquote(parsed.path)).resolve()
        assert destination.exists(), (base,target)

def main():
    manifest=json.loads((ROOT/'course_manifest.json').read_text(encoding='utf-8'))
    figures=exercises=0
    for row in manifest:
        path=ROOT/row['path']; notebook=nbformat.read(path,4)
        nbformat.validate(notebook)
        assert notebook.cells[0].cell_type=='markdown'
        assert notebook.cells[1].cell_type=='markdown', (path,'problem should precede setup')
        for cell in notebook.cells:
            if cell.cell_type=='markdown':
                local_links(cell.source,path.parent)
                continue
            ast.parse(cell.source)
            assert cell.execution_count is not None, (path,cell.id,'missing execution')
            if 'exercise' in cell.metadata.get('tags',[]):
                exercises+=1
                ast.parse(cell.metadata['reference_solution'])
            for output in cell.outputs:
                assert output.output_type!='error',(path,output)
                figures+='image/png' in output.get('data',{})
        print('PASS',row['slug'])
    for path in [ROOT/'README.md', ROOT/'THIRD_PARTY.md',ROOT/'references/SOURCES.md',ROOT/'references/TEACHING_NOTES.md']:
        local_links(path.read_text(encoding='utf-8'),path.parent)
    assert len(manifest)==19 and exercises==113,(len(manifest),exercises)
    assert (ROOT/'interactive/neural-network-lab.html').is_file()
    print(f'{len(manifest)} lessons, {exercises} exercises, {figures} saved figures; local links passed.')

if __name__=='__main__':
    main()
