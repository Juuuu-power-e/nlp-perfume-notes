import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. 사전 학습된 문장 임베딩 모델
model = SentenceTransformer("all-mpnet-base-v2")

# 2. spaCy 영어 모델 로딩
nlp = spacy.load("en_core_web_sm")

# 3. 예시: 테스트할 설명 문장
test_sentence = "Warm and woody with a hint of spice"

# 4. 의미 단위 추출 함수 (단어 or chunk 기반)
def extract_chunks(sentence):
    doc = nlp(sentence)
    chunks = []

    # 명사구, 형용사 등 추출
    for chunk in doc.noun_chunks:
        chunks.append(chunk.text)

    # 추가로 형용사도 단독 추출
    for token in doc:
        if token.pos_ == "ADJ":
            if token.text not in chunks:
                chunks.append(token.text)

    return list(set(chunks))  # 중복 제거

# 5. 의미 단위 추출
chunks = extract_chunks(test_sentence)
print("✅ 의미 단위:", chunks)

# 6. 유사 문장 후보군 (기존 전체 설명 문장 리스트)
# 예시용. 실제로는 62개 설명 전체가 여기 있어야 함
corpus = [
    "A cozy and warm scent with amber and vanilla",
    "Fruity citrus blend that feels fresh",
    "Earthy and woody tones with moss and vetiver",
    "Sweet floral notes with jasmine and rose",
    "Spicy cardamom with a smoky finish",
    "Fresh and aquatic with a clean musk base",
]

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 7. 각 chunk마다 유사 문장 찾기
for chunk in chunks:
    chunk_embedding = model.encode(chunk, convert_to_tensor=True)
    hits = util.semantic_search(chunk_embedding, corpus_embeddings, top_k=3)[0]
    print(f"\n🔎 '{chunk}' 와 유사한 설명 문장:")
    for hit in hits:
        score = hit['score']
        text = corpus[hit['corpus_id']]
        print(f" - ({score:.3f}) {text}")
