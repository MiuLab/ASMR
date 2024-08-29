from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import pickle
from sentence_transformers import SentenceTransformer, util
from random import sample


def cluster_labels(n_clusters, embeddings, ref_labels):
    
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    return labels


def draw_clusters(cluster_new_labels, embeddings, n_clusters):
    colors = ['purple','green','red','blue','yellow','pink','black','brown','orange','coral']
    cluster_new_labels = pd.Series(cluster_new_labels)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(embeddings)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]
    colors = colors[:n_clusters]
    for category, color in enumerate(colors):
        xs = np.array(x)[cluster_new_labels == category]
        ys = np.array(y)[cluster_new_labels == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
    plt.savefig("cluster.png")
    print('finish ploting')
    return None

def sample_cluster_labels(n_clusters, new_labels, cluster_new_labels):

    client = OpenAI(api_key='')
    
    sample_per_cluster = 3
    new_labels = pd.Series(new_labels)
    cluster_new_labels = pd.Series(cluster_new_labels)
    for i in range(n_clusters):
        print(f"Cluster {i} Theme:", end=" ")

        sample_actions = sample(list(new_labels[cluster_new_labels == i]), sample_per_cluster)
        actions = '\n'.join(sample_actions)
        
        messages = [
            {"role": "user", "content": f'What do the following reflected actions have in common?\n\nReflected actions:\n"""\n{actions}\n"""\n\nTheme:'}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        print(response.choices[0].message.content.replace("\n", ""))

        for j in range(sample_per_cluster):
            print(sample_actions[j])

        print("-" * 100)
    
    return None
