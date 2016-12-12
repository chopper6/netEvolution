import networkx as nx

def main():
    G=nx.Graph()
    nodez = [6, 'Dr.Derp', 3.3]
    G.add_nodes_from(nodez)
    H=nx.Graph()
    H.add_node(777)
    G.add_nodes_from(H) #diff from G.add_node(H)
    G.add_edge(6, 777)
    #print (G[0])
    print (G.number_of_nodes())

    G=nx.gnp_random_graph(50,.5)
    H=nx.gnp_random_graph(500,.5)
    I=nx.gnp_random_graph(30,.5)
    print (nx.clustering(G))
    print ("\n", H.order())
    print(I.degree())

    return


main()