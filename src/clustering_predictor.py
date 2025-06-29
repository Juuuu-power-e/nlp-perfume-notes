from sklearn.cluster import KMeans
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import pandas as pd

def loocv_cluster_predict(df, sentence_embeddings, Y, all_notes, ks=[1, 2, 3], num_clusters=10):
    results = []
    for test_idx in range(len(df)):
        X_train = np.delete(sentence_embeddings, test_idx, axis=0)
        Y_train = np.delete(Y, test_idx, axis=0)
        X_test = sentence_embeddings[test_idx]
        Y_test = Y[test_idx]

        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X_train.astype(np.float64))
        train_labels = kmeans.labels_

        cluster_top_notes = {}
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(train_labels == cluster_id)[0]
            cluster_notes = np.sum(Y_train[cluster_indices], axis=0)
            top_indices = np.argsort(cluster_notes)[::-1]
            cluster_top_notes[cluster_id] = top_indices

        test_cluster_id = kmeans.predict([X_test.astype(np.float64)])[0]
        predicted_note_ranks = cluster_top_notes[test_cluster_id]

        for k in ks:
            y_pred_vec = np.zeros(len(all_notes))
            y_pred_vec[predicted_note_ranks[:k]] = 1
            ap_score = label_ranking_average_precision_score([Y_test], [y_pred_vec])
            results.append({
                "test_index": test_idx,
                "k": k,
                "label_ranking_average_precision": ap_score
            })

    return pd.DataFrame(results)
