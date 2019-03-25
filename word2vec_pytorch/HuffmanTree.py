# Huffman Tree Structure
from heapq import heapify, heappop, heappush

# leaf node
class Node:
    # init to create a leaf node
    def __init__(self, name=None, value=None, depth=0):
        self._name = name
        self._value = value
        self._left_child = None
        self._right_child = None
        self._code = None
        self._depth = depth
        self._path = []

# Huffman tree
class HuffmanTree:
    def __init__(self, chars_weights):
        '''
        chars_weights : a list of tuples which consists of 
                        the chars and their freqs
        '''
        # create the series of leaf node by chars and their frequencies
        self.series = [(wgt, 0, chars, Node(chars, wgt)) for chars,wgt in chars_weights]
        heapify(self.series)
        # used to create the name of non-leaf node
        current_node_no = -1
        # structure the Huffman Tree
        while len(self.series) != 1:
            # Get the least two freq
            RValue, Rdepth, Rname, Rnode = heappop(self.series)
            Rnode._code = [0]
            Lvalue, Ldepth, Lname, Lnode = heappop(self.series)
            Lnode._code = [1]
            new_value = Lvalue+RValue
            # a new node created by two nodes of which values are the least two.
            current_node_no += 1
            new_name = [current_node_no]
            new_node = Node(name=new_name, value=new_value)
            # update the new node
            new_node._left_child = Lnode
            new_node._right_child = Rnode
            new_node._depth = max([Ldepth, Rdepth]) + 1
            # add the new node to the list
            heappush(self.series, (new_value, new_node._depth, new_name, new_node))
        # set the root of the tree
        self.root = self.series[0][-1]
        # create a dict to savehuffman code
        self.huffman_code = {}
        self.huffman_path = {}
        # get huffman code
        self.get_huffman_code_path(self.root)

    def get_huffman_code_path(self, current_node):
        '''return a dict of huffman code'''
        if current_node._depth == 0:
            current_node._code.reverse()
            self.huffman_code[current_node._name] = current_node._code
            self.huffman_path[current_node._name] = current_node._path
        else:
            if current_node._code is not None:
                current_node._left_child._code += current_node._code
                current_node._right_child._code += current_node._code
            current_node._left_child._path += current_node._path + current_node._name
            current_node._right_child._path += current_node._path + current_node._name

        if current_node._left_child is not None:
            self.get_huffman_code_path(current_node._left_child)
        if current_node._right_child is not None:
            self.get_huffman_code_path(current_node._right_child)


if __name__ == '__main__':
    test_data = [('a',3), ('b', 5), ('c', 7), ('d', 4), ('e', 11)]
    tree = HuffmanTree(test_data)
    hcodes = tree.huffman_code
    for i,j in hcodes.items():
        print('Huffman code of {} is: {}'.format(i,j))
    hpath = tree.huffman_path
    for i,j in hpath.items():
        print('Huffman path of {} is: {}'.format(i,j))
    print('Tree depth:',tree.root._depth)
 