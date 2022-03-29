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
        with open("processed_data/gene_seeds/{}_ncbi_seeds.json".format(disease), "r") as f:
            disease_gene_seeds = json.load(f)
        disease_gene_seeds = set([str(s) for s in disease_gene_seeds])

        # Use PEGASUS for the NCBI->Gene mapping
        data = pd.read_csv("processed_data/gwas_gene_pvals/{}/filtered_ncbi_PEGASUS_{}_gwas_data.csv".format(disease, disease))
        data = data[~data["NCBI_id"].isna()]
        data["NCBI_id"] = data["NCBI_id"].astype(str)
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
                if node in disease_gene_seeds:
                    pagerank_seeds[node2idx[node]] = 1
                else:
                    pagerank_seeds[node2idx[node]] = 0
                        

            def process_rwr_results(scores):
                rwr_results = []
                for i, node in enumerate(graph["names"]):
                    row = {}

                    row["Idx"] = node2idx[node]
                    row["Gene NCBI ID"] = node
                    row["Symbol"] = ncbi2gene[node] if node in ncbi2gene.keys() else "-"


                #     row[""]
                    if node in disease_gene_seeds:
                        init_score = 1
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
                rwr_results.to_csv("outputs/RWRs_gene_seeds/{}_{}_alpha{}_results.csv".format(disease, network, alpha), index=False)
            
            print("\n")




if __name__=="__main__":
    main()