import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg as la
import numpy as np


def load_graph():
    with open('grafuri.pickle', 'rb') as f:
        return pickle.load(f)


def obtain_graph(graph_set):
    for graph in graph_set:
        print(nx.convert_matrix.from_numpy_array(graph))


def draw_graphs(graph_set):
    for i, graph in enumerate(graph_set):
        graph = nx.from_numpy_array(graph)
        plt.figure(i)
        nx.draw(graph, with_labels=True)
    plt.show()


def get_eigenvalues(graph_set):
    for graph in graph_set:
        print(la.eigvals(graph))


def complete_graph(graph_set):
    for i, graph in enumerate(graph_set):
        eigenvalues = la.eigvals(graph)
        # graph is complete -> it has 2 eigenvalues
        if len(set(np.round(eigenvalues,6))) == 2:
            print(f"Graph {i+1} is complete")
        else:
            print(f"Graph {i+1} is not complete")


def bipartite_graph(graph_set):
    for i, graph in enumerate(graph_set):
        eigenvalues = la.eigvals(graph)
        # graph is bipartite -> λmin = - λmax
        # print(min(eigenvalues), max(eigenvalues))
        if np.round(min(eigenvalues), 6) == np.round(-max(eigenvalues), 6):
            print(f"Graph {i+1} is bipartite")
        else:
            print(f"Graph {i+1} is not bipartite")


def maximum_clique_graph(graph_set):
    for i, graph in enumerate(graph_set):
        # graph = nx.from_numpy_array(graph)
        # maximum_clique = nx.algorithms.approximation.max_clique(graph)

        # maximum_clique = λmax + 1
        maximum_clique = int(np.round(max(la.eigvals(graph).real) + 1))
        print(f"Graph {i + 1} has maximum clique = {maximum_clique}")


if __name__ == '__main__':
    graphs = load_graph()

    print("---------------------\na)")
    obtain_graph(graphs)

    print("---------------------\nb) Check the plots")
    draw_graphs(graphs)

    print("---------------------\nc)")
    get_eigenvalues(graphs)

    print("---------------------\nd)")
    print("=====================\nIS THE GRAPH COMPLETE?")
    complete_graph(graphs)

    print("=====================\nIS THE GRAPH BIPARTITE?")
    bipartite_graph(graphs)

    print("=====================\nWHAT IS THE MAXIMUM CLIQUE?")
    maximum_clique_graph(graphs)
