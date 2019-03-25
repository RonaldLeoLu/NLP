# create load data function
# define some constrant path variables
import pickle
from pathlib import Path
from collections import Counter
from HuffmanTree import HuffmanTree

data_path = Path(Path.cwd()).parent / 'data'
cache_path = Path(Path.cwd()).parent / 'cache'

def read_dataset(w2i_dict, filename, separate=" "):
    with open(str(data_path / filename / 'train.txt'), 'r') as f:
        for line in f:
            yield [w2i_dict[x] for x in line.lower().strip().split(separate)]

def get_HuffmanCodePath(filename, separate=" "):
    nodes = filename + '_nodes.pkl'
    codename = filename + '_code.pkl'
    pathname = filename + '_path.pkl'
    filepath = data_path / filename / 'train.txt'
    nodecache = cache_path / nodes
    codecache = cache_path / codename
    pathcache = cache_path / pathname

    if (codecache.exists()) and (pathcache.exists()) and (nodecache.exists()):
        huffman_nodes = pickle.load(open(str(nodecache), 'rb'))
        huffman_codes = pickle.load(open(str(codecache), 'rb'))
        huffman_paths = pickle.load(open(str(pathcache), 'rb'))
    else:
        wordlist = []
        with open(str(filepath), 'r') as f:
            for line in f:
                wordlist += [word for word in line.lower().strip().split(separate)]

        wordlist = Counter(wordlist)

        chars_weights = list(wordlist.items())

        tree = HuffmanTree(chars_weights)
        huffman_codes = tree.huffman_code
        huffman_paths = tree.huffman_path
        huffman_nodes = tree.root._name

        pickle.dump(huffman_codes, open(str(codecache), 'wb'))
        pickle.dump(huffman_paths, open(str(pathcache), 'wb'))
        pickle.dump(huffman_nodes, open(str(nodecache), 'wb'))

    return huffman_nodes, huffman_codes, huffman_paths

if __name__ == "__main__":
    a,b,c = get_HuffmanCodePath('ptb')
    print(list(b.keys())[0:20])