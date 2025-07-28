import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def evaluate_retrieval(embeddings, labels, k=5):
    """
    评估检索任务性能
    """
    # 准备数据
    X = np.array([item['embedding'] for item in embeddings])
    y = np.array([item['label'] for item in embeddings])
    
    # 使用kNN进行评估
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # 评估
    y_pred = knn.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def evaluate_text_matching(embeddings, pairs, threshold=0.8):
    """
    评估文本匹配任务性能
    """
    similarities = []
    true_labels = []
    
    for i, (idx1, idx2, label) in enumerate(pairs):
        # 计算余弦相似度
        vec1 = np.array(embeddings[idx1]['embedding'])
        vec2 = np.array(embeddings[idx2]['embedding'])
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        similarities.append(sim)
        true_labels.append(label)
    
    # 计算准确率
    predicted_labels = [1 if sim > threshold else 0 for sim in similarities]
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'threshold': threshold
    }

if __name__ == "__main__":
    # 示例用法
    with open('embeddings.jsonl', 'r') as f:
        embeddings = [json.loads(line) for line in f]
    
    # 评估检索任务
    retrieval_results = evaluate_retrieval(embeddings, k=5)
    print(f"Retrieval task results: {retrieval_results}")
    
    # 评估文本匹配任务
    # 假设我们有 pairs = [(idx1, idx2, is_similar), ...]
    # text_matching_results = evaluate_text_matching(embeddings, pairs)
    # print(f"Text matching results: {text_matching_results}")
