#!/usr/bin/env python

import pandas as pd
import numpy as np


def parse_gmt(gmt_file):
    """
    Parse gmt file into a dictionary of "gene_set_name: {gene_set}".

    :gmt_file: str
    :returns: dict
    """
    try:
        with open(gmt_file, "r") as handle:
            lines = handle.readlines()
    except IOError("Could not open %s" % gmt_file) as e:
        raise e

    # parse into dict
    gene_sets = dict()
    for line in lines:
        line = line.strip().split("\t")
        line.pop(1)  # remove url
        set_id = line.pop(0)

        gene_sets[set_id] = set(line)

    return gene_sets


def pair_gene_sets(gene_sets, suffixes=("UP", "DN")):
    """
    Given a dictionary of sets of genes, match sets based on key,
    if two sets have identically keys, only differing in the suffixes.
    Unpaired gene sets are returned additionally.
    """
    # get sets into nice dataframe
    sets = pd.Series(gene_sets, name="gene_sets")
    sets.index.name = "set_name"
    sets = sets.reset_index()

    # # get only sets with UP and DN (paired sets)
    unpaired = sets[~sets["set_name"].str.contains("_{}$|_{}$".format(*suffixes))]
    paired = sets[sets["set_name"].str.contains("_{}$|_{}$".format(*suffixes))]
    # label paired sets with direction
    paired["direction"] = np.nan
    paired.loc[paired["set_name"].str.endswith("_{}".format(suffixes[0])), "direction"] = "up"
    paired.loc[paired["set_name"].str.endswith("_{}".format(suffixes[1])), "direction"] = "down"
    # rename sets to "consensus"
    paired["set_name"] = paired["set_name"].str.replace("_{}$|_{}$".format(*suffixes), "")

    # pair sets in multindex dataframe
    paired = paired.set_index(['set_name', 'direction'])
    return (paired, unpaired)


def get_gene_set_values(matrix, paired):
    """
    Given a genes vs samples matrix (genes as index),
    return the distribution of mean (sample-wise) values on UP and DN genes of gene_set.
    Genes in set not in matrix will be ignored.
    """
    return paired['gene_sets'].apply(lambda x: matrix.ix[x].dropna().mean())


def compute_metrics(paired_values, paired):
    """
    Given two continuous distributions, compute measures of di/similarity,
    correlation and perform statistical tests between the two.
    """
    from scipy.stats import ks_2samp
    from statsmodels.sandbox.stats.multicomp import multipletests
    from scipy.stats import pearsonr

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    metrics = pd.DataFrame()
    for idx in paired_values.index.levels[0]:
        if len(paired_values.ix[idx]) == 2:
            series = pd.Series(name=idx)
            # length of gene sets
            series["size_up"] = len(paired.ix[idx, 'up']['gene_sets'])
            series["size_down"] = len(paired.ix[idx, 'down']['gene_sets'])
            # difference of means
            series["mean_difference"] = np.log2(paired_values.ix[idx, 'up']).mean() - np.log2(paired_values.ix[idx, 'down']).mean()
            # Pearson correlation
            series["r"] = pearsonr(paired_values.ix[idx, 'up'], paired_values.ix[idx, 'down'])[0]
            # RMSE
            series["error"] = rmse(paired_values.ix[idx, 'up'], paired_values.ix[idx, 'down'])
            # KS-test
            series["p_values"] = ks_2samp(paired_values.ix[idx, 'up'], paired_values.ix[idx, 'down'])[1]

            metrics = metrics.append(series)

    # correct p-values
    metrics["corrected_p_values"] = multipletests(metrics["p_values"])[1]
    return metrics


"http://software.broadinstitute.org/gsea/msigdb/download_file.jsp?filePath=/resources/msigdb/5.1/msigdb.v5.1.symbols.gmt"
gmt_file = "msigdb.v5.1.symbols.gmt"

gene_sets = parse_gmt(gmt_file)

(paired, unpaired) = pair_gene_sets(gene_sets)


df = pd.read_csv("/home/afr/Documents/workspace/cll-patients/data/cll_peaks.coverage_qnorm.log2.annotated.tsv", sep="\t")
df = df.set_index("gene_name")
df2 = df.drop([
    'chrom', 'start', 'end', 'genomic_region', 'chromatin_state',
    'support', 'mean', 'variance', 'std_deviation',
    'dispersion', 'qv2', 'fold_change'], axis=1)


# reduce values to mean per sample
set_matrix = get_gene_set_values(df2, paired)

# Get metrics of gene set comparisons
metrics = compute_metrics(set_matrix)

# Rank system

# Save


# Some plots

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

# rank vs mean_differenceerence
plt.scatter(metrics["mean_difference"].rank(), metrics["mean_difference"])


# visualize distribution of up vs down for some sets
s_a = metrics["mean_difference"].argmax()
s_rs = np.random.choice(metrics["mean_difference"].index, 23)
s_z = metrics["mean_difference"].argmin()

fig, axis = plt.subplots(5, 5, sharex=True, sharey=True)
axis = axis.flatten()

axis[0].scatter(set_matrix.ix[s_a, "down"], set_matrix.ix[s_a, "up"])
for i in range(23):
    axis[i + 1].scatter(set_matrix.ix[s_rs[i], "down"], set_matrix.ix[s_rs[i], "up"])
axis[-1].scatter(set_matrix.ix[s_z, "down"], set_matrix.ix[s_z, "up"])


# Volcano plot
plt.scatter(metrics['mean_difference'], -np.log10(metrics['p_values']))
