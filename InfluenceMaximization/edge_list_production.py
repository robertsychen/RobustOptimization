import numpy as np

def produce_wiki_edge_list():
    '''
    Formats edge data from Wikipedia graph into Python object.
    Relabels vertices to use integers 0 to V-1, where V = number of vertices.
    '''
    index_array = np.zeros(10000)

    raw_edge_list = []
    with open('Wiki-Vote.txt') as f:
        header_skip_counter = 0
        for line in f:
            if header_skip_counter < 4:
                header_skip_counter += 1
                continue
            raw_edge_list.append([int(line.split()[0]), int(line.split()[1])])
            index_array[int(line.split()[0])] = 1
            index_array[int(line.split()[1])] = 1

    vertex_index_dict = {}
    avail_index = 0
    for i in xrange(10000):
        if index_array[i] == 1.0:
            vertex_index_dict[i] = avail_index
            avail_index += 1

    edge_list = []
    for elem in raw_edge_list:
        edge_list.append([vertex_index_dict[elem[0]], vertex_index_dict[elem[1]]])
    return edge_list