# read more https://en.wikipedia.org/wiki/Hierarchical_clustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

# creating dataframe
# has to use absolute path as pd has an issue with set workdir
seeds_df = pd.read_csv("/Users/azhmakin/Documents/projects/own/university/intelligent-data-analysis/lab-2/assets/hierarchy-seeds-less-rows.csv")

# take information ab. grain variety, saving it for using in future
varieties = list(seeds_df.pop("grain_variety"))

# take measurements as a NumPy array
samples = seeds_df.values

# Implementation of hierarchical clustering using the linkage function
mergings = linkage(samples, method="complete")

# construct a dendrogram, specifying the parameters that are convenient for display
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)

plt.show()
