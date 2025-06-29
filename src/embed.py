import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import os

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_embeddings(dataset_path=None, embedding_path=None, model_name="all-mpnet-base-v2"):
    project_root = get_project_root()

    if dataset_path is None:
        dataset_path = os.path.join(project_root, "data", "dataset.xlsx")
    if embedding_path is None:
        embedding_path = os.path.join(project_root, "cache", "sentence_embeddings.npy")

    # 데이터셋 로드
    df = pd.read_excel(dataset_path)
    df["parsed_notes"] = df["notes"].apply(ast.literal_eval)

    # 임베딩 로드 또는 생성
    if os.path.exists(embedding_path):
        sentence_embeddings = np.load(embedding_path)
    else:
        model = SentenceTransformer(model_name)
        sentence_embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

        # 디렉토리가 없을 경우 생성
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, sentence_embeddings)

    # 전체 노트 목록 및 인덱스 매핑
    all_notes = sorted(set(note for notes in df["parsed_notes"] for note in notes))
    note_to_idx = {note: idx for idx, note in enumerate(all_notes)}

    def notes_to_vector(notes):
        vec = np.zeros(len(all_notes))
        for n in notes:
            if n in note_to_idx:
                vec[note_to_idx[n]] = 1
        return vec

    Y = np.array([notes_to_vector(notes) for notes in df["parsed_notes"]])
    return df, sentence_embeddings, Y, all_notes
