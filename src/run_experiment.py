import os
import pandas as pd
from src.embed import get_embeddings
from src.clustering_predictor import loocv_cluster_predict

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    project_root = get_project_root()

    dataset_path = os.path.join(project_root, "data", "dataset.xlsx")
    embedding_path = os.path.join(project_root, "cache", "sentence_embeddings.npy")
    results_path = os.path.join(project_root, "results", "result_loocv_cluster.csv")

    # 데이터 로딩 및 문장 임베딩
    df, sentence_embeddings, Y, all_notes = get_embeddings(
        dataset_path=dataset_path,
        embedding_path=embedding_path
    )

    # ✅ ks 리스트에 5를 추가하여 실험
    results_df = loocv_cluster_predict(df, sentence_embeddings, Y, all_notes, ks=[1, 2, 3, 5, 10], num_clusters=4)

    # 결과 저장
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)

    # 결과 출력
    print(" 평균 성능 요약 (k=1,2,3,5,10)")
    print(results_df.groupby("k")[["label_ranking_average_precision", "hit_at_k", "mrr"]].mean())
