"""
多層索引結構模組 - 專為大型文本集設計的高效檢索系統
"""

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import pickle

class IndexNode:
    """
    索引節點，用於多層索引結構
    """
    def __init__(self, vector, text=None, index=None, children=None):
        self.vector = vector  # 節點向量
        self.text = text      # 文本內容（僅葉節點有）
        self.index = index    # 節點索引
        self.children = children or []  # 子節點列表
        self.sample_count = 1  # 樣本數量

class MultiLevelIndex:
    """
    多層索引結構，用於大型文本集的高效檢索
    完全獨立於現有檢索樹架構
    """
    def __init__(self, max_nodes_per_cluster=100, max_clusters_per_level=10):
        self.max_nodes_per_cluster = max_nodes_per_cluster  # 每個聚類的最大節點數
        self.max_clusters_per_level = max_clusters_per_level  # 每層的最大聚類數
        self.top_level = []  # 頂層節點
        self.sub_trees = []  # 子樹列表
        self.vectors = None  # 原始向量
        self.texts = None    # 原始文本
        
    def build(self, vectors, texts):
        """
        構建多層索引結構
        
        Args:
            vectors: 文本向量 (numpy array)
            texts: 對應的文本列表
        """
        self.vectors = vectors
        self.texts = texts
        n_vectors = len(vectors)
        
        # 計算需要的聚類數
        n_clusters = min(self.max_clusters_per_level, 
                         max(2, n_vectors // self.max_nodes_per_cluster))
        
        print(f"構建多層索引：總文本數 {n_vectors}，聚類數 {n_clusters}")
        
        # 執行KMeans聚類
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        centroids = kmeans.cluster_centers_
        
        # 創建頂層索引節點
        self.top_level = []
        for i in range(n_clusters):
            # 標準化向量
            centroid = centroids[i]
            normalized_centroid = centroid / np.linalg.norm(centroid)
            
            # 創建代表該聚類的節點
            node = IndexNode(normalized_centroid, index=i)
            node.sample_count = np.sum(cluster_labels == i)
            self.top_level.append(node)
            
        # 為每個聚類創建子樹
        self.sub_trees = []
        for i in range(n_clusters):
            # 獲取屬於該聚類的向量和文本
            mask = cluster_labels == i
            cluster_vectors = vectors[mask]
            cluster_texts = [text for j, text in enumerate(texts) if mask[j]]
            
            # 為該聚類創建子樹
            subtree = self._build_subtree(cluster_vectors, cluster_texts)
            self.sub_trees.append(subtree)
            
            # 將子樹連接到頂層節點
            self.top_level[i].children = [subtree]
            
        print(f"多層索引構建完成：頂層節點 {len(self.top_level)}，子樹 {len(self.sub_trees)}")
        
    def _build_subtree(self, vectors, texts):
        """
        為聚類構建子樹
        
        Args:
            vectors: 聚類內的向量
            texts: 聚類內的文本
            
        Returns:
            IndexNode: 子樹根節點
        """
        # 如果只有一個文本，直接創建葉節點
        if len(vectors) == 1:
            return IndexNode(vectors[0] / np.linalg.norm(vectors[0]), texts[0], 0)
            
        # 使用階層式聚類構建子樹
        from scipy.cluster.hierarchy import linkage
        
        # 計算鏈接矩陣
        linkage_matrix = linkage(vectors, method="single", metric="cosine")
        
        # 構建子樹
        nodes = [
            IndexNode(vector / np.linalg.norm(vector), text, i)
            for i, (vector, text) in enumerate(zip(vectors, texts))
        ]
        
        n = len(nodes)
        current_index = n
        
        # 根據鏈接矩陣構建樹
        for i, (c1, c2, dist, sample_count) in enumerate(linkage_matrix):
            c1, c2 = int(c1), int(c2)
            count_c1 = nodes[c1].sample_count
            count_c2 = nodes[c2].sample_count
            
            # 計算新節點的向量（加權平均）
            new_vector = (nodes[c1].vector * count_c1 + nodes[c2].vector * count_c2) / (
                count_c1 + count_c2
            )
            new_vector /= np.linalg.norm(new_vector)
            
            # 創建新節點
            new_node = IndexNode(new_vector, None, current_index, [nodes[c1], nodes[c2]])
            new_node.sample_count = count_c1 + count_c2
            nodes.append(new_node)
            
            current_index += 1
            
        # 返回根節點
        return nodes[-1]
        
    def search(self, query, model, top_k=3, max_results=50):
        """
        在多層索引中搜索
        
        Args:
            query: 查詢文本
            model: 詞嵌入模型
            top_k: 頂層選擇的聚類數量
            max_results: 最大結果數量
            
        Returns:
            list: 檢索到的文本列表
        """
        if not self.top_level:
            return []
            
        # 編碼查詢
        query_vector = model.encode(query)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 計算與頂層節點的相似度
        similarities = [1 - cosine(query_vector, node.vector) for node in self.top_level]
        
        # 選擇最相似的top_k個聚類
        top_k = min(top_k, len(self.top_level))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 在選定的聚類中搜索
        all_results = []
        for idx in top_indices:
            subtree = self.sub_trees[idx]
            results = self._search_subtree(subtree, query_vector)
            all_results.extend(results)
            
        # 如果結果太多，進行排序和篩選
        if len(all_results) > max_results:
            # 計算每個結果的相似度
            result_texts = [r[0] for r in all_results]
            result_vectors = [r[1] for r in all_results]
            
            # 計算相似度
            similarities = [1 - cosine(query_vector, v) for v in result_vectors]
            
            # 排序並選擇前max_results個
            sorted_indices = np.argsort(similarities)[::-1][:max_results]
            all_results = [result_texts[i] for i in sorted_indices]
        else:
            all_results = [r[0] for r in all_results]
            
        return all_results
        
    def _search_subtree(self, node, query_vector):
        """
        在子樹中搜索
        
        Args:
            node: 子樹根節點
            query_vector: 查詢向量
            
        Returns:
            list: (文本, 向量) 元組列表
        """
        # 如果是葉節點，直接返回文本
        if not node.children:
            return [(node.text, node.vector)]
            
        # 使用BFS找到最相似的節點
        most_similar_node = self._find_most_similar_node(node, query_vector)
        
        # 收集葉節點文本
        return self._collect_leaf_texts(most_similar_node)
        
    def _find_most_similar_node(self, root, query_vector):
        """
        使用BFS找到最相似的節點
        
        Args:
            root: 子樹根節點
            query_vector: 查詢向量
            
        Returns:
            IndexNode: 最相似的節點
        """
        min_distance = float("inf")
        most_similar_node = None
        
        # 使用BFS遍歷樹
        stack = [root]
        while stack:
            node = stack.pop()
            distance = cosine(query_vector, node.vector)
            
            if distance < min_distance:
                min_distance = distance
                most_similar_node = node
                
            stack.extend(node.children)
            
        return most_similar_node
        
    def _collect_leaf_texts(self, node):
        """
        收集節點下所有葉節點的文本
        
        Args:
            node: 起始節點
            
        Returns:
            list: (文本, 向量) 元組列表
        """
        # 如果是葉節點，直接返回
        if not node.children:
            return [(node.text, node.vector)]
            
        # 否則遞歸收集所有葉節點
        results = []
        for child in node.children:
            results.extend(self._collect_leaf_texts(child))
            
        return results
        
    def save(self, filename):
        """
        保存索引到文件
        
        Args:
            filename: 文件路徑
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"多層索引已保存到 {filename}")
        
    @staticmethod
    def load(filename):
        """
        從文件加載索引
        
        Args:
            filename: 文件路徑
            
        Returns:
            MultiLevelIndex: 加載的索引
        """
        with open(filename, "rb") as f:
            index = pickle.load(f)
        print(f"多層索引已從 {filename} 加載")
        return index


# 工具函數

def create_multi_level_index(vectors, texts, max_nodes_per_cluster=100):
    """
    創建多層索引的工廠函數
    
    Args:
        vectors: 文本向量
        texts: 文本列表
        max_nodes_per_cluster: 每個聚類的最大節點數
        
    Returns:
        MultiLevelIndex: 創建的索引
    """
    index = MultiLevelIndex(max_nodes_per_cluster=max_nodes_per_cluster)
    index.build(vectors, texts)
    return index

def query_multi_level_index(index, query, model, top_k=3, max_results=50):
    """
    查詢多層索引
    
    Args:
        index: 多層索引
        query: 查詢文本
        model: 詞嵌入模型
        top_k: 頂層選擇的聚類數量
        max_results: 最大結果數量
        
    Returns:
        list: 檢索到的文本列表
    """
    return index.search(query, model, top_k, max_results)