"""
Prim

Prim's algorithm for minimum spanning trees.
Uses Adjcent Matrix.
"""
import csv

file_in = 'stdin'

####################################################################################
#           Prim Algorithm
####################################################################################

def prim(graph, vertices):

    # initialize the MST and the set X
    T = set();
    X = set();

    # select an arbitrary vertex to begin with
    X.add(0);

    while len(X) != vertices:
        crossing = set();
        # for each element x in X, add the edge (x, k) to crossing if
        # k is not in X
        for x in X:
            for k in range(vertices):
                if k not in X and graph[x][k] != 0:
                    crossing.add((x, k))
        # find the edge with the smallest weight in crossing
        edge = sorted(crossing, key=lambda e:graph[e[0]][e[1]])[0];
        # add this edge to T
        T.add(edge)
        # add the new vertex to X
        X.add(edge[1])
    return T, graph


####################################################################################
#           Time Algorithm 
####################################################################################

def readin_graph( file_name ):
    """
    Read in file as graph
    """
    # open the file and get read to read data    
    vertices,num_cities,poss_routes = 0,0,0
    graph = []

    matrix_in = []
    with open (file_name + '.csv', 'rb') as file_in:
        read = csv.reader(file_in)
        count =0
        for row in read:
            # retrieve first to integers
            if count == 0:
                num_cities = int(row[0])
                vertices = num_cities
                # initialize the graph
                graph = [[0]*vertices for _ in range(vertices)]
            elif count == 1:
             poss_routes = int(row[0])
            # store all routes
            else:
                # populate the graph
                u, v, weight = int(row[0]), int(row[1]), float(row[-1])
                #print 'row[0]', u, v, weight
                graph[u][v] = weight
                graph[v][u] = weight
            count = count+1
    return graph, vertices

print "____"*20
g,v = readin_graph( file_in )
T, graph = prim( g,v )

_sum = 0
for edge in T:
    _sum = _sum + graph[ edge[0] ][ edge[-1] ]
    print edge[0], edge[-1], graph[ edge[0] ][ edge[-1] ]
print _sum

from timeit import Timer

t = Timer(lambda:  prim( g,v ) )
print 'prim', t.timeit(number=1000)
