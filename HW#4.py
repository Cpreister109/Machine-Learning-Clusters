import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

path = '/Users/cadepreister/Desktop/Intro to Python Data Analytics/'
sep = '-'*80

def one(ef):
    k_values = [2, 3, 4, 5, 6]
    distort_data = []

    x = ef.iloc[:, :4].values
    species = ef.iloc[:, -1]
    species_names = species.unique()

    col_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    print(sep)

    for k in k_values:
        data = collect_data(x, k)
        center_data = data[0]
        distort_data.append(data[1])

        for col_x, col_y in col_pairs:
            plot_scatter(ef, species_names, species, col_x, col_y, center_data)

    plot_elbow(k_values, distort_data)

def collect_data(x, k):
    kmeans = KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=k,
                    n_init=10, random_state=0, tol=0.0001, verbose=0).fit(x)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    distort = (sum(np.min(cdist(x, centers, 'euclidean'), axis=1)) / x.shape[0])
    print(f"KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, \n"
          f"n_clusters={k}, n_init=10, random_state=0, tol=0.0001, verbose=0).fit(x)\n"
          f"Data set length (rows): 150\nNumber of Clusters: {k}\nThe Cluster Centroids:\n"
          f"{centers}\n{labels}\nSilhouette Score: {silhouette_score(x, labels)}\n{sep}")

    return [centers, distort]

def plot_scatter(ef, species_names, species, col_x, col_y, centers):
    x = ef.iloc[:, :4].values
    color_map = {'Iris-setosa': 'b', 'Iris-versicolor': 'g', 'Iris-virginica': 'r'}

    plt.scatter(x[:, 0], x[:, 1])

    for sp in species_names:
        plt.scatter(x[species == sp, col_x], x[species == sp, col_y], label=sp, color=color_map[sp])

    plt.scatter(centers[:, col_x], centers[:, col_y],
                c='Magenta', marker='p', s=200, edgecolor='black', label='Centroids')

    plt.xlabel(ef.columns[col_x])
    plt.ylabel(ef.columns[col_y])
    plt.title("K-Means: Iris Data Points & Cluster Centroids")
    plt.legend(loc='lower right')
    plt.show()

def plot_elbow(k_vals, dists):
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, dists, marker="o")
    plt.xlabel("# of K Clusters")
    plt.ylabel("Distortion (Avg distance to closest cluster center)")
    plt.title("K Clusters v. Distortion")
    plt.xticks(k_vals)
    plt.grid(True)
    plt.show()


def get_file():
    ef = pd.read_excel(f'{path}iris.xlsx', sheet_name=f'Sheet1')
    return ef

def main():
    file = get_file()
    one(file)

if __name__ == "__main__":

    main()