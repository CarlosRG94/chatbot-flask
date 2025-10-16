import json, re
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# -------- utilidades de preprocesado y mask
url_re = re.compile(r'https?://\S+')
email_re = re.compile(r'[\w\.-]+@[\w\.-]+')
phone_re = re.compile(r'\b\d{6,15}\b')  # tolerante

def mask_entities(text, placeholder_map=None):
    """Reemplaza urls/emails/phones por placeholders y guarda mapping."""
    if placeholder_map is None:
        placeholder_map = {}
    def sub_func(m, tag):
        val = m.group(0)
        key = f"<{tag}_{len(placeholder_map)+1}>"
        placeholder_map[key] = val
        return key

    text = url_re.sub(lambda m: sub_func(m, "URL"), text)
    text = email_re.sub(lambda m: sub_func(m, "EMAIL"), text)
    text = phone_re.sub(lambda m: sub_func(m, "PHONE"), text)
    return text, placeholder_map

def normalize(text):
    # lowercase, strip, remove punctuation except placeholders <...>
    text = text.lower().strip()
    # keep placeholders like <URL_1>, remove other punctuation
    # elimina puntuación salvo <, > y letras/dígitos/espacio/_
    text = re.sub(r"(?!<|>)[^\w\s\-_<>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -------- cargar y preparar dataset
with open("data/dataset.json", encoding="utf-8") as f:
    data = json.load(f)

questions_raw = [d["input_text"] for d in data]
answers_raw = [d["target_text"] for d in data]

# crear versiones enmascaradas y mapping global
global_placeholder_map = {}
questions_masked = []
answers_masked = []
for q,a in zip(questions_raw, answers_raw):
    q_masked, _ = mask_entities(q, global_placeholder_map)#Se añaden la plabras enmascaradas al diccionario
    a_masked, _ = mask_entities(a, global_placeholder_map)
    questions_masked.append(normalize(q_masked))
    answers_masked.append(a_masked)  # guardamos respuesta enmascarada tal cual (mantener placeholders)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Generando embeddings del dataset...")
question_embeddings = embed_model.encode(questions_masked, convert_to_tensor=True, show_progress_bar=True)

# Cross-encoder para re-rank 
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 

def improved_chat(query, top_k=5, threshold=0.7):
    query_masked, query_map = mask_entities(query, {})
    q_norm = normalize(query_masked)
    q_emb = embed_model.encode(q_norm, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, question_embeddings)[0]
    top = torch.topk(cos_scores, k=min(top_k, len(questions_masked)))
    top_scores = top.values.tolist()
    top_idxs = top.indices.tolist()

    if top_scores[0] < threshold:
        qlow = query.lower()
        if "linkedin" in qlow:
            return answers_raw[0]
        return "No estoy seguro de eso. Prueba con otra pregunta relacionada con mi perfil."

    pairs = [(query, questions_raw[i]) for i in top_idxs]
    rerank_scores = cross_encoder.predict(pairs)
    best_local = int(rerank_scores.argmax())
    best_idx = top_idxs[best_local]
    answer = answers_masked[best_idx]

    for ph, val in global_placeholder_map.items():
        if ph in answer:
            answer = answer.replace(ph, val)
    return answer
