import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import re

# Read the data from the file
with open('/Users/ahae/Documents/GenePioneer/CHD1L_DPF2_Results_Complete.txt', 'r') as f:
    content = f.read()

# Parse the modules and pathways with genes, p-values, enrichments
modules = []
pathways = []
pathway_data = []  # list of dicts with pval, enrich, genes
lines = content.split('\n')
in_chd1l = False
current_module = None
current_pathways = []
current_pathway_data = []
parsing_pathway = False
current_path = None
for line in lines:
    if '🧬 CHD1L MODULES' in line:
        in_chd1l = True
    elif '🧬 DPF2 MODULES' in line:
        in_chd1l = False
    elif in_chd1l and line.startswith('Module'):
        if current_module is not None:
            modules.append(current_module)
            pathways.append(current_pathways)
            pathway_data.append(current_pathway_data)
        module_str = line.split(': ')[1]
        current_module = eval(module_str)
        current_pathways = []
        current_pathway_data = []
        parsing_pathway = False
        current_path = None
    elif in_chd1l and 'HALLMARK_' in line.strip():
        if current_path is not None:
            current_pathway_data.append(current_data)
        current_path = line.strip().split(' (')[0]
        current_pathways.append(current_path)
        current_data = {'pathway': current_path, 'pval': None, 'enrich': None, 'genes': []}
        parsing_pathway = True
    elif in_chd1l and parsing_pathway and 'p-value:' in line:
        pval = line.split('p-value: ')[1].strip()
        current_data['pval'] = pval
    elif in_chd1l and parsing_pathway and 'Enrichment:' in line:
        enrich = line.split('Enrichment: ')[1].strip()
        current_data['enrich'] = enrich
    elif in_chd1l and parsing_pathway and 'Genes:' in line:
        genes_str = line.split('Genes: ')[1]
        genes = [g.strip() for g in genes_str.split(',')]
        current_data['genes'] = genes
        parsing_pathway = False
    elif in_chd1l and 'Significant pathways:' in line:
        pass  # Skip
    elif in_chd1l and line.strip() == '':
        pass
if current_path is not None:
    current_pathway_data.append(current_data)
if current_module is not None:
    modules.append(current_module)
    pathways.append(current_pathways)
    pathway_data.append(current_pathway_data)

# Create a combined figure with all modules
fig, axes = plt.subplots(1, 5, figsize=(30, 6))  # Larger figure for better visibility
# fig.suptitle('CHD1L Modules Network Visualizations', fontsize=18, fontweight='bold')

for i, (module, path_list, path_data_list) in enumerate(zip(modules, pathways, pathway_data)):
    ax = axes[i]
    
    G = nx.Graph()
    
    # Add central module node
    module_node = f'Module {i+1}'
    G.add_node(module_node, type='module')
    
    # Add gene nodes
    gene_nodes = []
    for gene in module:
        G.add_node(gene, type='gene')
        G.add_edge(module_node, gene)
        gene_nodes.append(gene)
    
    # Add pathway nodes and edges directly to module
    pathway_nodes = []
    for data in path_data_list:
        path = data['pathway']
        pval = data['pval']
        # Shorten pathway name more
        short_path = path.replace('HALLMARK_', '').replace('_', ' ').title()
        if len(short_path) > 10:
            short_path = short_path[:8] + '...'
        # Label with pathway and p-value, but shorter
        label = f'{short_path}\n{pval}'
        G.add_node(label, type='pathway', full_name=path, pval=pval)
        pathway_nodes.append(label)
        # Connect directly to module
        G.add_edge(module_node, label)
    
    # print(f"Module {i+1}: {len(pathway_nodes)} pathways added")
    
    num_genes = len(gene_nodes)
    num_paths = len(pathway_nodes)
    
    # Custom layout
    pos = {}
    import math
    
    # Center for module
    pos[module_node] = (0, 0)
    
    # Genes in a circle
    radius_gene = 1
    for j, gene in enumerate(gene_nodes):
        angle = 2 * math.pi * j / num_genes if num_genes > 0 else 0
        pos[gene] = (radius_gene * math.cos(angle), radius_gene * math.sin(angle))
    
    # Pathways in a larger circle around module
    radius_path = 2.5  # Further reduced radius
    for j, path in enumerate(pathway_nodes):
        angle = 2 * math.pi * j / num_paths if num_paths > 0 else 0
        pos[path] = (radius_path * math.cos(angle), radius_path * math.sin(angle))
    
    # Refine with spring layout
    pos = nx.spring_layout(G, pos=pos, fixed=[module_node], seed=42)
    
    # Node shapes: all circles
    node_shapes = ['o'] * len(G.nodes())  # All circles
    
    # Better colors with significance for pathways
    node_colors = []
    for node in G.nodes():
        if node == module_node:
            node_colors.append('#C0392B')  # Darker red for module
        elif G.nodes[node]['type'] == 'gene':
            node_colors.append('#2980B9')  # Darker blue for genes
        else:
            # Color pathways based on p-value
            pval_str = G.nodes[node]['pval']
            try:
                pval_num = float(pval_str)
                if pval_num < 0.01:
                    node_colors.append('#27AE60')  # Darker green for significant
                else:
                    node_colors.append('#E67E22')  # Darker orange for less significant
            except:
                node_colors.append('#7F8C8D')  # Gray default
    
    # Node sizes
    node_sizes = [3000 if node == module_node else (2000 if G.nodes[node]['type'] == 'pathway' else 1500) for node in G.nodes()]
    
    # Labels
    labels = {node: node for node in G.nodes()}
    
    # Draw the graph without borders
    for shape, color, size, node in zip(node_shapes, node_colors, node_sizes, G.nodes()):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_shape=shape, node_size=size, ax=ax, linewidths=0, edgecolors='none')
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#34495E', alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6, font_weight='bold')
    
    # Remove axes borders
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', markersize=10, label='Module'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9', markersize=10, label='Gene'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', markersize=10, label='Pathway (p<0.01)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22', markersize=10, label='Pathway (p≥0.01)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=5)
    
    # Add module score to title
    ax.set_title(f'Module {i+1}', fontsize=14)

plt.tight_layout()
plt.savefig('/Users/ahae/Documents/GenePioneer/figures/CHD1L_All_Modules_Network_Beautiful.png', dpi=600, bbox_inches='tight')
plt.close()

print("Beautiful combined network figure saved.")
