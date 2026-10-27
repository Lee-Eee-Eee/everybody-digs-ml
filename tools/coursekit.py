"""Notebook authoring utilities for the course."""
from pathlib import Path
import hashlib
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]

# Keep lesson numbers stable while grouping notebooks by their main learning goal.
LESSON_DIRECTORIES = {
    **{n: 'python_math' for n in range(0, 4)},
    **{n: 'neural_network_basics' for n in range(4, 8)},
    **{n: 'computer_vision' for n in range(8, 10)},
    **{n: 'sequence_models' for n in range(10, 12)},
    **{n: 'llm' for n in range(12, 16)},
    **{n: 'other_networks' for n in range(16, 18)},
    **{n: 'experiments' for n in range(18, 20)},
    **{n: 'classical_ml' for n in range(20, 22)},
    22: 'other_networks',
    23: 'computer_vision',
}


def lesson_path(slug):
    try:
        group = LESSON_DIRECTORIES[int(slug.split('_', 1)[0])]
    except (ValueError, KeyError) as error:
        raise ValueError(f'No course directory registered for {slug!r}') from error
    return ROOT / 'lessons' / group / (slug + '.ipynb')


class Lesson:
    def __init__(self, slug, title, goals=None, prereq=''):
        self.slug, self.title, self.cells = slug, title, []
        self.goals, self.prereq = goals or [], prereq
        self.md(f'# {title}')
        self.code('from support import ROOT, check, show, style')
        self.cells[-1].metadata.update({'tags':['course-setup'], 'jupyter':{'source_hidden':True}})

    def md(self, text):
        self.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(self, text):
        self.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    def ex(self, identifier, title, prompt, starter, hint, solution, checks):
        self.md(f'### 练习 {identifier}：{title}\n\n' + textwrap.dedent(prompt).strip())
        self.code(starter)
        self.cells[-1].metadata.update({'tags': ['exercise'], 'exercise_id': identifier,
                                       'starter_source': self.cells[-1].source,
                                       'reference_solution': textwrap.dedent(solution).strip()})
        self.code('# 填写上一格后，运行检查。\n' + textwrap.dedent(checks).strip())
        self.cells[-1].metadata.update({'tags': ['exercise-check'], 'exercise_id': identifier})
        self.md('<details>\n<summary>提示</summary>\n\n' + textwrap.dedent(hint).strip() +
                '\n\n</details>\n\n<details>\n<summary>参考答案</summary>\n\n```python\n' +
                textwrap.dedent(solution).strip() + '\n```\n\n</details>')

    exercise = ex

    def reading(self, title, url, prompt):
        self.md(f'## 带着问题读原文\n\n[{title}]({url})\n\n{prompt}')

    def notebook(self):
        # Let the concrete problem precede the housekeeping imports.
        first_body = next((i for i,c in enumerate(self.cells[1:],1) if c.cell_type=='markdown'), None)
        if first_body is not None and first_body != 1:
            self.cells.insert(1, self.cells.pop(first_body))
        for i, cell in enumerate(self.cells):
            cell['id'] = 'lesson-' + hashlib.sha1((self.slug + str(i) + cell.source).encode()).hexdigest()[:12]
        notebook = nbformat.v4.new_notebook(cells=self.cells)
        notebook.metadata.update({'lesson_id': self.slug, 'learning_goals':self.goals, 'prerequisite':self.prereq,
            'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python (neural_networks_from_zero)'},
            'language_info': {'name': 'python', 'version': '3.12'}})
        nbformat.validate(notebook)
        return notebook

    def write(self):
        path = lesson_path(self.slug)
        notebook = self.notebook()
        if path.exists():
            old = nbformat.read(path, 4)
            answers = {c.metadata['exercise_id']: c for c in old.cells
                       if 'exercise' in c.metadata.get('tags', []) and
                       c.source != c.metadata.get('starter_source', c.source)}
            for cell in notebook.cells:
                identifier = cell.metadata.get('exercise_id')
                if 'exercise' in cell.metadata.get('tags', []) and identifier in answers:
                    cell.source = answers[identifier].source
            previous_code = [c for c in old.cells if c.cell_type=='code']
            current_code = [c for c in notebook.cells if c.cell_type=='code']
            if [c.source for c in previous_code] == [c.source for c in current_code]:
                for previous,current in zip(previous_code,current_code):
                    current.outputs = previous.outputs
                    current.execution_count = previous.execution_count
            elif (len(previous_code)==len(current_code) and
                  [c.source for c in previous_code[1:]]==[c.source for c in current_code[1:]] and
                  'course-setup' in previous_code[0].metadata.get('tags',[])):
                # The shared display-helper import was simplified; examples are unchanged.
                for previous,current in zip(previous_code,current_code):
                    current.outputs = previous.outputs
                    current.execution_count = previous.execution_count
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, path)
        return path
