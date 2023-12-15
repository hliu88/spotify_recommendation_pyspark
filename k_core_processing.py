import pandas as pd
import networkx as nx

# Read in data
df = pd.read_csv('all_data_raw.csv')
df.drop(['artist_name', 'trackid'], axis=1, inplace=True)

# Create graph
G = nx.Graph()
for i in range(len(df)):
    if i % 1000000 == 0:
        print(i / 1000000)
    G.add_edge(df['track_name'][i], df['pid'][i])

# k-core decomposition
# k_core = nx.k_core(G, k=50)
k_core = nx.k_core(G, k=25)

k_core_edges = k_core.edges()
k_core_edges_df = pd.DataFrame(k_core_edges)
k_core_edges_np = k_core_edges_df.to_numpy()

# some edges are flipped, we'd like pid to be the first column
for i in range(len(k_core_edges_np)):
    if type(k_core_edges_np[i][0]) == str:
        k_core_edges_np[i][0], k_core_edges_np[i][1] = k_core_edges_np[i][1], k_core_edges_np[i][0]

# sort by pid and save to csv
k_core_edges_np = k_core_edges_np[k_core_edges_np[:, 0].argsort()]
k_core_edges_df = pd.DataFrame(k_core_edges_np)
k_core_edges_df.to_csv('k_core_50_edges.csv', index=False)