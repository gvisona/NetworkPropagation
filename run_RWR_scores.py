import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sknetwork.ranking import PageRank
from sknetwork.data import convert_edge_list, load_edge_list


diseases = ["asthma", "autism", "schizophrenia"]

network_files = {
    "BioPlex3": "processed_data/networks/BioPlex3_shared/edges_list_ncbi.csv",
    "HumanNet": "processed_data/networks/HumanNetV3/edges_list_ncbi.csv",
    "PCNet": "processed_data/networks/PCNet/edges_list_ncbi.csv",
    "ProteomeHD": "processed_data/networks/ProteomeHD/edges_list_ncbi.csv",
    "STRING": "processed_data/networks/STRING/edges_list_ncbi.csv",
}

alpha_array = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

def main():

    for disease in diseases:
        print("Analysing {} \n\n".format(disease))
        data = pd.read_csv("processed_data/gwas_gene_pvals/{}/filtered_ncbi_PEGASUS_{}_gwas_data.csv".format(disease, disease))
        data = data[~data["NCBI_id"].isna()]
        data["NCBI_id"] = data["NCBI_id"].astype(str)
        min_pval = data["Pvalue"][data["Pvalue"]>0].min()
        pegasus_scores = {}
        for i, row in data.iterrows():
            pv = np.maximum(min_pval, np.minimum(1, row["Pvalue"]))
            pegasus_scores[row["NCBI_id"]] = np.maximum(0, -np.log10(pv))
        pegasus_ncbi_genes = set(pegasus_scores.keys())
        ncbi2gene = dict(zip(data.NCBI_id, data.Gene))

        for network in network_files.keys():
            # Load Graph
            print("Loading {} graph\n".format(network))
            df = pd.read_csv(network_files[network])[["node1", "node2"]].astype(str)
            graph_nodes = set(df["node1"]).union(set(df["node2"]))
            edge_list = list(df.itertuples(index=False))
            graph = convert_edge_list(edge_list)
            node2idx = {n: i for i, n in enumerate(graph.names)}
            idx2node = {v: k for k, v in node2idx.items()}
            adjacency = graph.adjacency

            pagerank_seeds = {}
            for node in graph["names"]:
                if node in pegasus_ncbi_genes:
                    pagerank_seeds[node2idx[node]] = pegasus_scores[node]
                else:
                    pagerank_seeds[node2idx[node]] = 0

            seeds_vector = np.array(list(pagerank_seeds.values()))
            print(np.isnan(seeds_vector).any())
            nonzero_genes = [idx2node[g] for g in pagerank_seeds.keys() if pagerank_seeds[g]>0]
            max_val = np.max(seeds_vector[~np.isinf(seeds_vector)])
            

            def process_rwr_results(scores):
                rwr_results = []
                for i, node in enumerate(graph["names"]):
                    row = {}

                    row["Idx"] = node2idx[node]
                    row["Gene NCBI ID"] = node
                    row["Symbol"] = ncbi2gene[node] if node in ncbi2gene.keys() else "-"


                #     row[""]
                    if node in nonzero_genes:
                        init_score = pagerank_seeds[node2idx[node]]
                        if np.isinf(init_score):
                            init_score = max_val+1
                    else:
                        init_score = 0

                    row["Initial Score"] = init_score
                    row["Final Score"] = scores[i]

                    rwr_results.append(row)

                rwr_results = pd.DataFrame(rwr_results).sort_values(by="Final Score", ascending=False)
                return rwr_results

            for alpha in alpha_array:
                print("Alpha: ", alpha)
                pagerank = PageRank(damping_factor=alpha)
                rwr_scores = pagerank.fit_transform(adjacency, pagerank_seeds)
                rwr_results = process_rwr_results(rwr_scores)
                rwr_results.to_csv("outputs/RWRs_scores/{}_{}_alpha{}_results.csv".format(disease, network, alpha), index=False)
            
            print("\n")




if __name__=="__main__":
    main()