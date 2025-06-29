import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import label_ranking_average_precision_score

def hit_at_k(y_true, y_pred, k):
    """Top-k 안에 정답 노트가 하나라도 있는 경우 1, 없으면 0"""
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    return int(np.any(y_true[top_k_indices] > 0))

def reciprocal_rank(y_true, y_pred):
    """가장 먼저 정답을 맞춘 노트의 순위 역수 (MRR 계산용)"""
    ranked_indices = np.argsort(y_pred)[::-1]
    for i, idx in enumerate(ranked_indices):
        if y_true[idx] > 0:
            return 1.0 / (i + 1)
    return 0.0

def loocv_cluster_predict(df, sentence_embeddings, Y, all_notes, ks=[1, 2, 3], num_clusters=10):
    results = []
    for test_idx in range(len(df)):
        # 훈련셋 구성
        X_train = np.delete(sentence_embeddings, test_idx, axis=0).astype(np.float64)
        Y_train = np.delete(Y, test_idx, axis=0)
        X_test = sentence_embeddings[test_idx].astype(np.float64)
        Y_test = Y[test_idx]

        # 클러스터링 수행
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X_train)
        train_labels = kmeans.labels_

        # 클러스터별 대표 노트 추출
        cluster_top_notes = {}
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(train_labels == cluster_id)[0]
            cluster_notes = np.sum(Y_train[cluster_indices], axis=0)
            top_indices = np.argsort(cluster_notes)[::-1]
            cluster_top_notes[cluster_id] = top_indices

        # 테스트 샘플 클러스터 할당 및 예측
        test_cluster_id = kmeans.predict([X_test])[0]
        predicted_note_ranks = cluster_top_notes[test_cluster_id]

        for k in ks:
            y_pred_vec = np.zeros(len(all_notes))
            y_pred_vec[predicted_note_ranks[:k]] = 1

            ap_score = label_ranking_average_precision_score([Y_test], [y_pred_vec])
            hit = hit_at_k(Y_test, y_pred_vec, k)
            mrr = reciprocal_rank(Y_test, y_pred_vec)

            results.append({
                "test_index": test_idx,
                "k": k,
                "label_ranking_average_precision": ap_score,
                "hit_at_k": hit,
                "mrr": mrr
            })

    return pd.DataFrame(results)
