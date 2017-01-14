"""
Kruskal

Kruskal's algorithm for minimum spanning trees.
"""
# Packages for timing algorithm and file input
from timeit import Timer
import csv

# Document Name
file_in = 'stdin'

class Kruskal:
    """
    Kruskal
    - __init__
    - make_set
    - find
    - union
    - kruskal
    """

    def __init__(self):
        self.node = {}
        self.rank = {}

    def make_set(vertice):
        self.node[vertice] = vertice
        self.rank[vertice] = 0

    def find(vertice):
        # Finds the parent nodes vertices for union
        if self.node[vertice] != vertice:
            self.node[vertice] = find( self.node[vertice] )
        return self.node[vertice]

    def union(vertice1, vertice2):
        # Finds smallest rank
        root1 = find(vertice1)
        root2 = find(vertice2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.node[root2] = root1
            else:
                self.node[root1] = root2
                if self.rank[root1] == self.rank[root2]: self.rank[root2] += 1

    def kruskal(graph):
        """
        Kruskal Algorithm
        Takes graph with routes set as:
        - wieght, vertices (a,b)
        - returns minimum routes 
        """
        for vertice in graph['vertices']:
            make_set(vertice)

        minimum_spanning_tree = set()
        edges = list(graph['edges'])
        edges.sort()
        for edge in edges:
            weight, vertice1, vertice2 = edge
            if find(vertice1) != find(vertice2):
                union(vertice1, vertice2)
                minimum_spanning_tree.add(edge)
        return minimum_spanning_tree

#############################################
#           Create Graph from file
#############################################

def readin_graph( file_name ):
    """
    Read in file as graph
    """
    num_cities = 0
    poss_routes = 0

    graph = {}
    with open ( file_name + '.csv', 'rb') as file_in:
        read = csv.reader(file_in)
        count =0
        for row in read:
            # retrieve first to integers
            if count == 0:
                num_cities = int(row[0])
                graph['vertices'] = [x for x in range(0,8)]
            elif count == 1:
             poss_routes = int(row[0])
             edges = []
            # store all routes
            else:
                edges.append((float(row[-1]),int(row[0]),int(row[1])))

            count = count+1
    graph['edges'] = set(edges)
    return graph

#############################################
#           Time Algorithm
#############################################
k = Kruskal()
g = readin_graph( file_in )
print 'g',g
kru = k.kruskal( g )

_sum = 0
for k in kru:
    _sum = _sum + k[0]
    print 'k', k
print _sum

t = Timer(lambda:  k.kruskal( readin_graph( file_in ) )
print 'kruskal', t.timeit(number=1000)