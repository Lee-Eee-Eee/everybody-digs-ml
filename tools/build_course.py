"""Generate the complete course without overwriting filled exercise answers."""
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    records=[]
    for name in ['orientation_research','foundations','torch_vision','attention','time_graphs','evaluation']:
        module=importlib.import_module('chapters.'+name)
        for lesson in module.build():
            path=lesson.write()
            records.append(dict(slug=lesson.slug,title=lesson.title,path=path.relative_to(ROOT).as_posix(),
                                cells=len(lesson.cells),exercises=sum('exercise' in c.metadata.get('tags',[]) for c in lesson.cells)))
    records.sort(key=lambda r:r['slug'])
    assert len({r['slug'] for r in records})==len(records)
    (ROOT/'course_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Built {len(records)} lessons, {sum(r['exercises'] for r in records)} exercises")
    for row in records: print(row['slug'],row['cells'],row['exercises'])


if __name__=='__main__':
    main()
