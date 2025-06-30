import pandas as pd
import spacy
import numpy as np
import ast
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter
import os

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 모델 초기화

project_root = get_project_root()

dataset_path = os.path.join(project_root, "data", "dataset.xlsx")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-mpnet-base-v2")

# 데이터 불러오기
df = pd.read_excel(dataset_path)


# 1. 의미 단위(semantic chunks) 추출
def extract_chunks(text):
    doc = nlp(text)
    chunks = []

    # 명사구 + 형용사 단어 추출
    for chunk in doc.noun_chunks:
        chunks.append(chunk.text.lower())

    for token in doc:
        if token.pos_ == "ADJ":
            if token.text.lower() not in chunks:
                chunks.append(token.text.lower())

    return list(set(chunks))

df["chunks"] = df["description"].apply(extract_chunks)

# 2. 각 chunk에 대한 임베딩 생성
all_chunks = sorted({chunk for chunk_list in df["chunks"] for chunk in chunk_list})
chunk_embeddings = model.encode(all_chunks, convert_to_tensor=True)

# 3. 각 chunk에 대해 유사한 chunk 그룹 찾기 (semantic 그룹핑)
chunk_to_similar_chunks = defaultdict(list)
similarity_threshold = 0.75

for i, query_chunk in enumerate(all_chunks):
    query_vec = chunk_embeddings[i]
    hits = util.semantic_search(query_vec, chunk_embeddings, top_k=10)[0]

    for hit in hits:
        if hit["score"] >= similarity_threshold:
            similar_chunk = all_chunks[hit["corpus_id"]]
            chunk_to_similar_chunks[query_chunk].append(similar_chunk)

# 4. 각 chunk 그룹이 등장한 문장에서 노트 수집
chunk_group_to_notes = defaultdict(list)

for chunk, similar_chunks in chunk_to_similar_chunks.items():
    for i, row in df.iterrows():
        if any(sim_chunk in row["chunks"] for sim_chunk in similar_chunks):
            chunk_group_to_notes[chunk].extend(row["notes"])

# 5. 각 chunk 그룹에 대해 가장 자주 등장한 노트 top-k 저장
chunk_to_top_notes = {}

for chunk, notes in chunk_group_to_notes.items():
    counter = Counter(notes)
    top_notes = [note for note, _ in counter.most_common(3)]  # top-3
    chunk_to_top_notes[chunk] = top_notes

# 6. 예측 함수: 새로운 문장 → 의미 단위 추출 → 각 chunk → 예측된 노트 병합
def predict_notes(description):
    chunks = extract_chunks(description)
    predicted_notes = []

    for chunk in chunks:
        if chunk in chunk_to_top_notes:
            predicted_notes.extend(chunk_to_top_notes[chunk])
        else:
            # 유사한 chunk가 있는지 찾기
            chunk_vec = model.encode(chunk, convert_to_tensor=True)
            hits = util.semantic_search(chunk_vec, chunk_embeddings, top_k=3)[0]
            for hit in hits:
                sim_chunk = all_chunks[hit["corpus_id"]]
                if sim_chunk in chunk_to_top_notes and hit["score"] > similarity_threshold:
                    predicted_notes.extend(chunk_to_top_notes[sim_chunk])
                    break

    return list(set(predicted_notes))  # 중복 제거

# 7. 예시 실행
example = "Cozy and smoky scent with hints of vanilla"
print("💡 예측 결과:", predict_notes(example))
