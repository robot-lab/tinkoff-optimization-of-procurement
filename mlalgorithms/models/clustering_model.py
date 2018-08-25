from sklearn.cluster  import KMeans
import pandas as pd
import numpy as np

from . import model


class ClusteringModel(model.IModel):

    def __init__(self, **kwargs):
        self.kmeans = KMeans(**kwargs)
        self.orders = {}
        self.clustering_table = pd.DataFrame()
        self.largest_cluster_goods = []

    def train(self, train_samples, train_labels, **kwargs):
        persons_ids = [person_data[0] for person_data in train_samples]
        for persons_id, label in zip(persons_ids, train_labels):
            if self.orders.get(persons_id) is None:
                self.orders[persons_id] = np.array(label)
            else:
                self.orders[persons_id] += np.array(label)

        self.clustering_table = pd.DataFrame(
            self.kmeans.fit_predict(
                pd.DataFrame.from_dict(self.orders, orient='index')),
            columns=['cluster'])

        cluster_id = self.clustering_table['cluster'].value_counts().index[0]
        larg_clust_center = self.kmeans.cluster_centers_[cluster_id]
        self.largest_cluster_goods = (larg_clust_center >= 6).astype(np.int)

    def predict(self, samples, **kwargs):
        predictions = []
        for i in samples:
            if i[0] in self.clustering_table.index:
                cluster_id = self.clustering_table.at[i[0], 'cluster']
                clust_center = self.kmeans.cluster_centers_[cluster_id]
                prediction = (clust_center >= 6).astype(np.int)
            else:
                prediction = np.array(self.largest_cluster_goods)
            predictions.append(prediction)
        return predictions
