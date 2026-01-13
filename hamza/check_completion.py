import json
import sys

def check_notebook(nb_path):
    with open(nb_path, 'r') as f:
        nb_data = json.load(f)

    print(f"=== {nb_path} ===")

    # Find experiment sections
    current_exp = None
    experiments = {}

    for i, cell in enumerate(nb_data['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])

            # Look for experiment markers
            if '## Experiment' in source or '## Exp ' in source:
                lines = source.split('\n')
                for line in lines:
                    if line.strip().startswith('## Exp'):
                        current_exp = line.strip('# ').strip()
                        experiments[current_exp] = {'start': i, 'code_cells': 0, 'output_cells': 0}
                        break

        elif cell['cell_type'] == 'code' and current_exp:
            experiments[current_exp]['code_cells'] += 1
            if cell.get('outputs', []):
                experiments[current_exp]['output_cells'] += 1

    # Print summary
    if experiments:
        for exp_name, info in experiments.items():
            if info['code_cells'] > 0:
                completion = info['output_cells'] / info['code_cells'] * 100
                status = "✅" if completion > 80 else "⚠️" if completion > 30 else "❌"
                print(f"{status} {exp_name}: {info['output_cells']}/{info['code_cells']} cells ({completion:.0f}%)")
            else:
                print(f"📝 {exp_name}: (no code cells)")
    else:
        print("  No experiments found with ## Experiment markers")

    print()

if __name__ == '__main__':
    for nb in sys.argv[1:]:
        check_notebook(nb)
