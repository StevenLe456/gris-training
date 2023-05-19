import gensim.downloader
import networkx as nx
from nltk.tokenize import word_tokenize
import numpy as np
import os
import random
import scipy as sp
from sklearn.decomposition import IncrementalPCA

def create_transition_matrix(vertex_adj):
    edge_index = np.nonzero(vertex_adj.todense())
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.sparse.csr_matrix((data, (row_index, col_index)),
               shape=(vertex_adj.todense().shape[0], num_edge))

    return T

def word_vecs():
    return gensim.downloader.load("word2vec-google-news-300")

def pca():
    return IncrementalPCA(n_components=512)

def train_test_random(dir, train_fraction):
    nonpsychotic_dir = dir + "/nonpsychotic_graphs"
    psychotic_dir = dir + "/psychotic_graphs"
    nonpsychotic_files = [nonpsychotic_dir + "/" + f for f in os.listdir(nonpsychotic_dir) if f.endswith(".gml")]
    nonpsychotic_files = random.sample(nonpsychotic_files, int(len(nonpsychotic_files) * 0.5))
    psychotic_files = [psychotic_dir + "/" + f for f in os.listdir(psychotic_dir) if f.endswith(".gml")]
    psychotic_files = random.sample(psychotic_files, int(len(psychotic_dir) * 0.5))
    f = nonpsychotic_files + psychotic_files
    random.shuffle(f)
    return f[0:int(len(f) * train_fraction)], f[int(len(f) * train_fraction):int(len(f))]

def embeddings(w2v, words):
    dummy = np.zeros((len(words), 1500))
    for k, w in enumerate(words):
        word_list = word_tokenize(w)
        temp = np.zeros((5, 300))
        for i in range(5):
            try:
                temp[i] = w2v[word_list[i]] if word_list[i] in w2v else np.zeroes((1, 300))
            except:
                break
        temp = temp.flatten()
        dummy[k] = temp
    return dummy

def process_data(file_path, w2v):
    g = nx.read_gml(file_path)
    d = {}
    if not nx.is_empty(g):
        d["node_adjacency_matrix"] = nx.adjacency_matrix(g)
        d["edge_adjacency_matrix"] = nx.adjacency_matrix(nx.line_graph(g))
        d["n_e"] = embeddings(w2v, g.nodes)
        d["e_e"] = embeddings(w2v, [a[2] for a in g.edges.data("relation")])
        d["t"] = create_transition_matrix(d["node_adjacency_matrix"])
        if file_path.startswith("data/nonpsychotic_graphs/"):
            d["label"] = 0
        elif file_path.startswith("data/psychotic_graphs"):
            d["label"] = 1
        else:
            raise ValueError
    return d