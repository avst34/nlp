import csv
import sys

def parse_psseval(psseval_path):
    with open(psseval_path, 'r') as f:
        rows = [x.strip().split('\t') for x in f.readlines()]
    return {
        'ALL': {
            'ID': {
                'P': rows[3][6],
                'R': rows[3][7],
                'F': rows[3][8]
            },
            'Role': {
                'P': rows[3][10],
                'R': rows[3][11],
                'F': rows[3][12]
            },
            'Fxn': {
                'P': rows[3][14],
                'R': rows[3][15],
                'F': rows[3][16]
            },
            'Role,Fxn': {
                'P': rows[3][18],
                'R': rows[3][19],
                'F': rows[3][20]
            },
        },
        'MWE': {
            'ID': {
                'P': rows[8][6],
                'R': rows[8][7],
                'F': rows[8][8]
            },
            'Role': {
                'P': rows[8][10],
                'R': rows[8][11],
                'F': rows[8][12]
            },
            'Fxn': {
                'P': rows[8][14],
                'R': rows[8][15],
                'F': rows[8][16]
            },
            'Role,Fxn': {
                'P': rows[8][18],
                'R': rows[8][19],
                'F': rows[8][20]
            },
        },
        'MWP': {
            'ID': {
                'P': rows[13][6],
                'R': rows[13][7],
                'F': rows[13][8]
            },
            'Role': {
                'P': rows[13][10],
                'R': rows[13][11],
                'F': rows[13][12]
            },
            'Fxn': {
                'P': rows[13][14],
                'R': rows[13][15],
                'F': rows[13][16]
            },
            'Role,Fxn': {
                'P': rows[13][18],
                'R': rows[13][19],
                'F': rows[13][20]
            },
        }
    }


def collect_results(results_dir):
    stypes = ['train', 'dev', 'test']
    gtypes = ['ALL', 'MWE', 'MWP']
    sstypes = ['ID', 'Role', 'Fxn', 'Role,Fxn']
    scs = ['P', 'R', 'F']
    tasks = [idt + '.' + syn for idt in ['autoid', 'goldid'] for syn in ['autosyn', 'goldsyn']]
    d = {}
    for stype in stypes:
        for task in tasks:
            evl = parse_psseval(results_dir + '/' + task + '/' + task + '.' + stype + '.psseval.tsv')
            d[task] = d.get(task) or {}
            d[task][stype] = evl

    with open(results_dir + '/eval.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([''] + ['Train'] + ['']*(4*3**2 - 1) + ['Dev'] + ['']*(4*3**2 - 1) + ['Test'] + ['']*(4*3**2 - 1))
        writer.writerow([''] + (['ALL'] + ['']*(4*3**1 - 1) + ['MWE'] + ['']*(4*3**1 - 1) + ['MWP'] + ['']*(4*3**1 - 1)) * 3)
        writer.writerow([''] + (['ID'] + ['']*(3**1 - 1) + ['Role'] + ['']*(3**1 - 1) + ['Fxn'] + ['']*(3**1 - 1) + ['Role,Fxn'] + ['']*(3**1 - 1)) * 9)
        writer.writerow([''] + scs * 4*3**2)
        for task in tasks:
            writer.writerow([task] + [
                d[task][stype][gtype][sstype][sc] for stype in stypes for gtype in gtypes for sstype in sstypes for sc in scs
            ])

if __name__ == '__main__':
    collect_results(sys.argv[1])