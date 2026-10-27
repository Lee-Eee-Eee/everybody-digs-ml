"""Check saved lesson structure, executable syntax and local learning links."""
from pathlib import Path
import ast
import json
import re
from urllib.parse import unquote, urlsplit
import nbformat

ROOT=Path(__file__).resolve().parents[1]

def local_links(text,base):
    targets = re.findall(r'\]\(([^\s)]+)\)', text)
    targets += re.findall(r'''(?:src|href)=["']([^"']+)["']''', text)
    for target in targets:
        parsed=urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        destination=(base/unquote(parsed.path)).resolve()
        assert destination.is_relative_to(ROOT), (base, target, 'outside repository')
        assert destination.exists(), (base,target)

def main():
    manifest=json.loads((ROOT/'course_manifest.json').read_text(encoding='utf-8'))
    figures=exercises=0
    assert len({row['slug'] for row in manifest}) == len(manifest), 'duplicate lesson slug'
    for row in manifest:
        path=ROOT/row['path']; notebook=nbformat.read(path,4)
        nbformat.validate(notebook)
        assert path.stem == row['slug'], (path, row['slug'])
        assert len(notebook.cells) == row['cells'], (path, 'cell count')
        actual_exercises = sum('exercise' in c.metadata.get('tags', []) for c in notebook.cells)
        assert actual_exercises == row['exercises'], (path, 'exercise count')
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
    for path in [*ROOT.glob('*.md'), *(ROOT/'docs').glob('*.md'), ROOT/'references/SOURCES.md', ROOT/'references/TEACHING_NOTES.md']:
        local_links(path.read_text(encoding='utf-8'),path.parent)
    assert exercises == sum(row['exercises'] for row in manifest), 'manifest exercise total'
    # The interactive lessons contain HTML links inside JavaScript strings.
    # Check both their source files and the generated standalone page.
    for path in [ROOT/'interactive/lessons.js', ROOT/'interactive/foundations-lessons.js',
                 ROOT/'interactive/neural-network-lab.html']:
        assert path.is_file(), path
        for target in re.findall(r'href=[\"\'](\.\./lessons/[^\"\']+\.ipynb)[\"\']',
                                 path.read_text(encoding='utf-8')):
            assert (path.parent/target).resolve().is_file(), (path, target)
    print(f'{len(manifest)} lessons, {exercises} exercises, {figures} saved figures; local links passed.')

if __name__=='__main__':
    main()
