"""
Floyd Warshall

Floyd Warshall's algorithm for finding the shortest paths in a weighted graph.
Uses Adjcent Matrix.
"""

def create_matrices(graph):
    # Convert the directed graph to an adjacency matrix.
    # Make all unkown distance infinite.
    vertices = graph.keys()
    return {vert_1: {vert_2: 0 if vert_1 == vert_2 else graph [ vert_1 ].get(vert_2, float('inf'))
                 for vert_2 in vertices}
            for vert_1 in vertices}

def Floyd_Warshall(graph):
    vertices = graph.keys()
    dis_graph = graph
    for vert_2 in vertices:
        # Add smallest disance 
        dis_graph = { vert_1: { vert_3: min(dis_graph [ vert_1 ][vert_3], dis_graph[ vert_1 ][ vert_2 ] + dis_graph[ vert_2 ][vert_3] )
                 for vert_3 in vertices }
             for vert_1 in vertices }
    return dis_graph

def short_dist(vert_1, vert_2, graph):
    # Find the shorest distance in FW for given vertices
    dis = 0
    fw_dis = Floyd_Warshall( create_matrices(graph) )
    routes = fw_dis[ vert_1 ][ vert_2 ]
    print routes

#################
# Test
#################

STDIN = raw_input()
STDIN = STDIN.split(' ')
n = int(STDIN[0])
e = int(STDIN[-1])

graph = {}
for edge in range(1,e+1):
    # read all the 
    row = raw_input()
    row = row.split(' ')
    vert_1, vert_2, w = int(row[0]), int(row[1]), int(row[-1])
    # 
    if vert_1 in graph: graph[ vert_1 ][ vert_2 ] = w        
    else: graph[ vert_1 ]={vert_2:w}
        
    # Negative routes
    if vert_2 in graph: graph[ vert_2 ][ vert_1 ] = w
    else: graph[ vert_2 ]={vert_1: w}

    
dis = Floyd_Warshall( create_matrices(graph) )

# Questions
q = raw_input()
for row in range(0, int(q)):
    row = raw_input()
    row = row.split(' ')
    vert_1, vert_2 = int(row[0]), int(row[-1])

    short_dist( vert_1, vert_2, graph )