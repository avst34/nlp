import json

import sys


def create_embeddings_txt(orig_txt_path, target_txt_path, max_line=3072):
    dim = None
    d = {}
    with open(target_txt_path, 'wb') as out_f:
        with open(orig_txt_path) as f:
            for ind, line in enumerate(f):
                line = line[:-1]
                if dim is None:
                    dim = int(line.split()[1])
                    l = "%d %d" % (max_line, dim)
                else:
                    l = line
                    w = ' '.join(line.split()[:-dim])
                    d[w] = ind
                lb = l.encode('utf8')
                lb += b"\x00" * (max_line - len(lb))
                assert len(lb) == max_line
                out_f.write(lb)
    with open(target_txt_path+'.inds', 'w') as out_f:
        json.dump(d, out_f)


if __name__ == '__main__':
    create_embeddings_txt(sys.argv[1], sys.argv[2])