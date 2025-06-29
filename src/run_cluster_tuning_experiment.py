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
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)  # results 폴더 없으면 생성

    df, sentence_embeddings, Y, all_notes = get_embeddings(
        dataset_path=dataset_path,
        embedding_path=embedding_path
    )

    cluster_range = [2, 3, 4, 5, 10, 15, 20, 25, 30]

    for num_clusters in cluster_range:
        print(f"\n========== 📦 num_clusters = {num_clusters} ==========")
        results_df = loocv_cluster_predict(
            df,
            sentence_embeddings,
            Y,
            all_notes,
            ks=[1, 2, 3, 5, 10],
            num_clusters=num_clusters
        )

        # 콘솔 출력
        summary = results_df.groupby("k")[["label_ranking_average_precision", "hit_at_k", "mrr"]].mean()
        print("📊 평균 성능 요약 (k=1,2,3,5,10)")
        print(summary.round(6))

        # 결과 CSV 저장
        result_csv_path = os.path.join(results_dir, f"cluster_eval_k{num_clusters}.csv")
        results_df.to_csv(result_csv_path, index=False)
