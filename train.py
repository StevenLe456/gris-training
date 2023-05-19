import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_preprocessing import embeddings, pca, process_data, train_test_random, word_vecs
import intel_extension_for_pytorch as ipex
import json
from model import GCN
import networkx as nx
import numpy as np
import os
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_dtype(torch.float64)

def accuracy(output, labels):
    output = [0 if n < 0.5 else 1 for n in output]
    acc = [1 if a == b else 0 for a, b in zip(output, labels)]
    return sum(acc) / len(acc)

print("Creating pipeline metadata file...")
if not os.path.exists("pipeline.json"):
    with open("pipeline.json", "w") as f:
        data = {"module": 0}
        json.dump(data, f)

print("Reading pipeline metadata...")
data = json.load(open("pipeline.json", "r"))

print("Reading in graph directory and splitting into train/test datasets...")
train_files, test_files = train_test_random("data", 0.8)

print("Creating word vectors object...")
word_vectors = word_vecs()

if data["module"] == 0:
    print("Creating temp directories...")
    os.makedirs("temp/n", exist_ok=True)
    os.makedirs("temp/e", exist_ok=True)

    print("Creating PCA object...")
    node_pca = pca()
    edge_pca = pca()

    print("Collecting word embeddings...")
    node_emb = []
    edge_emb = []
    data_files = train_files + test_files
    i = 0
    j = 0
    for f in data_files:
        g = nx.read_gml(f)
        nes = np.asarray(embeddings(word_vectors, g.nodes))
        ees = np.asarray(embeddings(word_vectors, [a[2] for a in g.edges.data("relation")]))
        for ne in nes:
            np.save("temp/n/" + str(i) + "_n.npy", ne)
            i += 1
        for ee in ees:
            np.save("temp/e/" + str(j) + "_e.npy", ee)
            j += 1

    print("Training PCA models...")        
    n_filenames = ["temp/n/" + f for f in os.listdir("temp/n") if f.endswith("_n.npy")]
    e_filenames = ["temp/e/" + f for f in os.listdir("temp/e") if f.endswith("_e.npy")]
    for n in range(0, len(n_filenames), 750):
        n_arr = []
        n_subset = n_filenames[n : n + 750 if n + 750 < len(n_filenames) else len(n_filenames)]
        for ns in n_subset:
            n_arr.append(np.load(ns).reshape(1, -1))
        n_arr = np.concatenate(tuple(n_arr), axis=0)
        try:
            node_pca.partial_fit(n_arr)
        except:
            break
    for e in range(0, len(e_filenames), 750):
        e_arr = []
        e_subset = e_filenames[e : e + 750 if e + 750 < len(e_filenames) else len(e_filenames)]
        for es in e_subset:
            e_arr.append(np.load(es).reshape(1, -1))
        e_arr = np.concatenate(tuple(e_arr), axis=0)
        try:
            edge_pca.partial_fit(e_arr)
        except:
            break

    print("Deleting temp directory...")
    shutil.rmtree("temp")

    with open("node_pca.pickle", "wb") as p:
        pickle.dump(node_pca, p)

    with open("edge_pca.pickle", "wb") as p:
        pickle.dump(edge_pca, p)

    data["module"] = 1
    with open("pipeline.json", "w") as f:
        json.dump(data, f)

print("Creating model...")
model = GCN(0.4)
model = model.to(memory_format=torch.channels_last)
optimizer = optim.Adam(model.parameters(), lr=0.005)
model, optimizer = ipex.optimize(model, optimizer=optimizer)
criteria = F.binary_cross_entropy
acc_measure = accuracy

def train(epoch, nf, ef, eadj, nadj, t, label):
    tim = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(nf, ef, eadj, nadj, t)
    """loss_train = criteria(torch.tensor(output), torch.LongTensor([[j for i in range(output.shape[0])] for j in label]))
    acc_train = acc_measure(torch.tensor(output), torch.LongTensor([[j for i in range(output.shape[0])] for j in label]))"""
    x = []
    y = []
    for i, j in zip(output, label):
        x.append(i)
        for k in range(i.shape[0]):
            y.append(j)
    x = torch.cat(x).squeeze()
    y = torch.DoubleTensor(y)
    loss_train = criteria(x, y)
    loss_train.backward()
    optimizer.step()
    acc_train = acc_measure(x, y)
    print("Epoch: ", epoch, 'time: {:.4f}s'.format(time.time() - tim))
    """
    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'time: {:.4f}s'.format(time.time() - tim))
    """
    return loss_train, acc_train

def test(nf, ef, eadj, nadj, t, label):
    model.eval()
    output = model(nf, ef, eadj, nadj, t)
    """loss_test = criteria(output, torch.LongTensor([[j for i in range(output.shape[0])] for j in label]))
    acc_test = acc_measure(output, torch.LongTensor([[j for i in range(output.shape[0])] for j in label]))"""
    x = []
    y = []
    for i, j in zip(output, label):
        x.append(i)
        for k in range(i.shape[0]):
            y.append(j)
    x = torch.cat(x).squeeze()
    y = torch.DoubleTensor(y)
    loss_test = criteria(x, y)
    acc_test = acc_measure(x, y)
    return loss_test, acc_test

print("Loading PCA models...")
node_pca = pickle.load(open("node_pca.pickle", "rb"))
edge_pca = pickle.load(open("edge_pca.pickle", "rb"))

log = open("log.txt", "w")

print("Training...")
t_total = time.time()
train_loss_test = []
train_acc_test = []
test_loss_test = []
test_acc_test = []
for epoch in range(10):
    train_loss_test = []
    train_acc_test = []
    test_loss_test = []
    test_acc_test = []
    train_num = 0
    test_num = 0
    for f in range(0, len(train_files), 500):
        data = []
        for fil in train_files[f : f + 500 if f + 500 < len(train_files) else len(train_files)]:
            dat = process_data(fil, word_vectors)
            data.append(dat)
        h_v0 = []
        h_e0 = []
        eam = []
        nam = []
        t = []
        label = []
        for d in data:
            if "n_e" in d:
                h_v0.append(torch.from_numpy(node_pca.transform(d["n_e"])))
                h_e0.append(torch.from_numpy(edge_pca.transform(d["e_e"])))
                eam.append(d["edge_adjacency_matrix"])
                nam.append(d["node_adjacency_matrix"])
                t.append(d["t"])
                label.append(d["label"])
        l, a = train(epoch, h_v0, h_e0, eam, nam, t, label)
        train_loss_test.append(l)
        train_acc_test.append(a)
        train_num += 1
    print("Train set results:",
    "loss= {:.4f}".format(sum(train_loss_test)),
    "accuracy= {:.4f}".format(sum(train_acc_test) / len(train_acc_test)))
    log.write("Train set results: " +
            "loss= {:.4f} ".format(sum(train_loss_test)) +
            "accuracy= {:.4f}".format(sum(train_acc_test) / len(train_acc_test)) + "\n")
    for f in range(0, len(test_files), 500):
        data = []
        for fil in test_files[f : f + 500 if f + 500 < len(test_files) else len(test_files)]:
            dat = process_data(fil, word_vectors)
            data.append(dat)
        h_v0 = []
        h_e0 = []
        eam = []
        nam = []
        t = []
        label = []
        for d in data:
            if "n_e" in d:
                h_v0.append(torch.from_numpy(node_pca.transform(d["n_e"])))
                h_e0.append(torch.from_numpy(edge_pca.transform(d["e_e"])))
                eam.append(d["edge_adjacency_matrix"])
                nam.append(d["node_adjacency_matrix"])
                t.append(d["t"])
                label.append(d["label"])
        l, a = test(h_v0, h_e0, eam, nam, t, label)
        test_loss_test.append(l)
        test_acc_test.append(a)
        test_num += 1
    print("Test set results:",
            "loss= {:.4f}".format(sum(test_loss_test)),
            "accuracy= {:.4f}".format(sum(test_acc_test) / len(test_acc_test)))
    log.write("Test set results: " +
            "loss= {:.4f} ".format(sum(test_loss_test)) +
            "accuracy= {:.4f}".format(sum(test_acc_test) / len(test_acc_test)) + "\n")

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "model.pt")
log.close()