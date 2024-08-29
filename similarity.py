from openai import OpenAI
import pickle
import json
import numpy as np 
import pandas as pd
import time
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
from cluster import cluster_labels, draw_clusters, sample_cluster_labels

class gpt3_embedding_model:
    def __init__(self):
        self.client = OpenAI(api_key='')
        self.model = 'text-embedding-3-small'

    def get_model_name(self):
        return self.model

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        while(True):
            try:
                response = self.client.embeddings.create(input = [text], model = self.model)
                break        
            except Exception as e:
                print(e)
                time.sleep(3)
                continue

        return response.data[0].embedding
    
    def encode(self, e_input):
        if isinstance(e_input, list):
            return np.array([self.get_embedding(e) for e in e_input])
        elif isinstance(e_input, str):
            return np.array(self.get_embedding(e_input))

def generate_answer_similarity(options, options_embeddings, embedding):
    
    similarity = []

    for e in options_embeddings:
        cos_sim = util.cos_sim(embedding, e)
        similarity.append(cos_sim)
    m = max(similarity)
    i = similarity.index(m)

    return options[i], i

def scoring(response_file, model_type, label_type):
    
    with open('./new_answer.pkl','rb') as f:
        new_labels = pickle.load(f)
    print(len(new_labels)) 
    with open('./answer.pkl','rb') as f:
        labels = pickle.load(f)
    print(len(labels))

    if model_type == 'sbert':
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif model_type == 'gpt3':
        model = gpt3_embedding_model()
    embeddings = model.encode(new_labels)
        
    with open('./ref_answer.pkl','rb') as f1:
        ref_answer = pickle.load(f1)
    print(len(ref_answer)) 
    
    if label_type == 'high':
        n_clusters = 8
        cluster_new_labels = cluster_labels(n_clusters, embeddings, ref_answer)
        draw_clusters(cluster_new_labels, embeddings, n_clusters)
        sample_cluster_labels(n_clusters, new_labels, cluster_new_labels)
        
    pred = []
    true = []
    scores = []
    errors = []
    eval_result = {}
    answer_df = pd.read_json(response_file, lines=True)
    assert len(answer_df) == 400
    for i, row in answer_df.iterrows():
        gen_answer = row['text']
        emb = model.encode(gen_answer)
        similar_gen_answer, pred_index = generate_answer_similarity(new_labels, embeddings, emb)
        ref = ref_answer[i]
        if label_type == 'high':
            pred_index = cluster_new_labels[pred_index]
            ref = cluster_new_labels[ref]
        if pred_index == ref:
            scores.append(1)
        else:
            scores.append(0)
            errors.append(gen_answer+"####"+similar_gen_answer+'####'+labels[ref_answer[i]])
                   
        pred.append(pred_index)
        true.append(ref)
    eval_result['accuracy'] = sum(scores) / len(scores)
    eval_result['f1_micro'] = f1_score(true, pred, average='micro')
    eval_result['f1_macro'] = f1_score(true, pred, average='macro')
    eval_result['error_gen_answer'] = errors

    return eval_result

