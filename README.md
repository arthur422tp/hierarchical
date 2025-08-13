# Legal Document Retrieval System Based on Hierarchical Clustering
## 階層式聚類法規文本檢索系統

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

**基於階層式聚類與 RAG 的法規文本智慧檢索引擎**

[📖 快速開始](#🚀-快速開始) • [📚 使用指南](#📚-詳細使用指南) • [🔧 API 參考](#🔬-api-參考) • [📄 arXiv 論文](https://arxiv.org/abs/2506.13607)

</div>

## 📖 系統介紹

本系統是一個結合 AI 法律助手與法規查詢功能的智慧檢索引擎，核心技術為階層式聚類（Hierarchical Clustering）與餘弦相似度（Cosine Similarity），並透過 OpenAI 的 Retrieval-Augmented Generation（RAG）技術進行答案生成。適用於智慧律所、法律聊天機器人、學術研究或多語言法條索引場景，能有效提供準確、可解釋的法規查詢回覆。

### 🎯 核心特色

- **🌳 階層式檢索樹**：自動建構語意層次索引結構
- **🔍 雙重檢索模式**：支援直接檢索與查詢提取兩種方式
- **🧠 RAG 技術整合**：結合 OpenAI GPT 進行智能答案生成
- **⚡ 高效能檢索**：無須手動設定 k 值，自動篩選相關文本
- **🎨 模組化設計**：易於整合到現有專案中
- **🌐 全端解決方案**：內建前端 UI + REST API
- **🐳 Docker 支援**：支援容器化部署，一鍵啟動

## 🛠️ 技術架構

| Component | Tech Used |
|----------|------------|
| Frontend | HTML / JavaScript / Tailwind CSS |
| Backend | FastAPI |
| Embedding Model | `intfloat/multilingual-e5-large` |
| Retrieval Tree | Hierarchical Clustering + Cosine Similarity |
| LLM API | OpenAI GPT (ChatGPT API) |
| Containerization | Docker & Docker Compose |

### 核心組件架構

```
hierarchical-rag-retrieval/
├── retrieval/          # 檢索核心模組
│   ├── RAGTree_function.py      # 階層式檢索樹
│   ├── multi_level_search.py    # 多層索引檢索
│   └── generated_function.py    # 查詢提取功能
├── utils/              # 工具模組
│   ├── word_embedding.py        # 詞嵌入處理
│   ├── word_chunking.py         # 文本分塊
│   └── query_retrieval.py       # FAISS 檢索
├── data_processing/    # 資料處理模組
│   └── data_dealer.py           # 資料格式處理
└── app/                # 演示應用
    ├── main.py                  # FastAPI 主程式
    └── static/index.html        # 前端介面
```

## 🚀 快速開始

### 📦 套件安裝

```bash
pip install hierarchical-rag-retrieval
```

### 🎯 基本使用範例

```python
from src.retrieval import create_ahc_tree, tree_search
from src.utils import WordEmbedding

# 1. 初始化詞嵌入模型
embedding_model = WordEmbedding()
model = embedding_model.load_model()

# 2. 準備您的文本資料
texts = [
    "民法總則規定自然人之權利能力始於出生終於死亡",
    "土地法規定土地所有權之移轉應辦理登記",
    "都市計畫法規定都市計畫區域內土地使用分區",
    # ... 更多文本
]

# 3. 生成文本向量
vectors = model.encode(texts)

# 4. 建立階層式檢索樹
tree_root = create_ahc_tree(vectors, texts)

# 5. 進行檢索
query = "土地所有權移轉需要什麼程序？"
results = tree_search(
    tree_root, 
    query, 
    model, 
    chunk_size=100, 
    chunk_overlap=20
)

# 6. 查看結果
for i, result in enumerate(results, 1):
    print(f"{i}. {result}")
```

### 🔧 環境設置（演示應用）

#### 前置條件

- Python 3.8+ 或 Docker 環境
- OpenAI API 金鑰

#### 方法一：傳統部署

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置環境變數
echo "OPENAI_API_KEY=your_openai_api_key" > .env

# 啟動應用
cd app && python main.py
```

#### 方法二：Docker 部署 (推薦)

```bash
# 設置環境變數
echo "OPENAI_API_KEY=your_openai_api_key" > .env

# 啟動服務
docker-compose up -d

# 查看日誌
docker-compose logs -f
```

應用啟動後，瀏覽器訪問 http://localhost:8000 使用系統。

## 📚 詳細使用指南

### 1. 階層式檢索樹 (RAGTree)

階層式檢索樹是本系統的核心功能，通過聚類算法自動組織文本向量。

```python
from src.retrieval import create_ahc_tree, tree_search, save_tree, load_tree

# 建立檢索樹
tree_root = create_ahc_tree(vectors, texts)

# 儲存檢索樹（可重複使用）
save_tree(tree_root, "my_retrieval_tree.pkl")

# 載入已儲存的檢索樹
tree_root = load_tree("my_retrieval_tree.pkl")

# 進行檢索
results = tree_search(
    root=tree_root,
    query="您的查詢問題",
    model=embedding_model.load_model(),
    chunk_size=100,
    chunk_overlap=20,
    max_chunks=10
)
```

**參數說明：**
- `chunk_size`: 文本分塊大小，較大的值保留更多上下文
- `chunk_overlap`: 分塊重疊大小，避免重要資訊被截斷
- `max_chunks`: 最大分塊數量，控制處理效率

### 2. 多層索引檢索(目前尚未完成，以下為預計情形｜Not done yet)

針對大型文本庫優化的多層索引系統。

```python
from src.retrieval import (
    build_multi_level_index_from_files, 
    multi_level_tree_search,
    multi_level_extraction_tree_search
)

# 從檔案建立多層索引
index = build_multi_level_index_from_files(
    embeddings_path="embeddings.pkl",
    texts_path="texts.pkl"
)

# 直接檢索
results = multi_level_tree_search(
    index=index,
    query="查詢問題",
    model=model,
    chunk_size=100,
    chunk_overlap=20
)

# 使用查詢提取的檢索（適合複雜問題）
results = multi_level_extraction_tree_search(
    index=index,
    query="複雜的法律問題描述...",
    model=model,
    chunk_size=100,
    chunk_overlap=20,
    llm=openai_llm  # OpenAI 語言模型
)
```

### 3. 查詢提取與優化

對於複雜或冗長的查詢，系統可以自動提取核心問題。

```python
from src.retrieval import extraction_tree_search
from langchain_openai import ChatOpenAI

# 設定 OpenAI 模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="your-openai-api-key"
)

# 使用查詢提取進行檢索
complex_query = """
我想了解關於土地買賣的法律規定，特別是在都市計畫區域內，
如果我要購買一塊土地用於商業用途，需要注意哪些法律條文？
另外，土地所有權的移轉程序是什麼？
"""

results = extraction_tree_search(
    root=tree_root,
    query=complex_query,
    model=model,
    chunk_size=100,
    chunk_overlap=20,
    llm=llm
)
```

### 4. 自定義文本處理

```python
from src.utils import WordEmbedding, RagChunking
from src.data_processing import DataDealer
import pickle

# 處理自定義文本資料
dealer = DataDealer()

# 準備文本資料
custom_texts = [
    "您的第一個文檔內容...",
    "您的第二個文檔內容...",
    # ... 更多文本
]

# 生成嵌入向量
embedding_model = WordEmbedding()
model = embedding_model.load_model()
vectors = model.encode(custom_texts)

# 儲存處理後的資料
with open('custom_texts.pkl', 'wb') as f:
    pickle.dump(custom_texts, f)
with open('custom_embeddings.pkl', 'wb') as f:
    pickle.dump(vectors, f)

# 文本分塊處理
chunker = RagChunking("長文本內容...")
chunks = chunker.text_chunking(chunk_size=200, chunk_overlap=50)
```

## 🎨 進階用法（非主要功能，有些已經廢棄）

### 1. 自定義重排序(需自行導入cross-encoder model)

```python
from src.retrieval import rerank_texts

# 對檢索結果進行重新排序
query = "查詢問題"
passages = ["文檔1", "文檔2", "文檔3"]
reranked_passages = rerank_texts(query, passages, model)
```

### 2. 批次處理

```python
def batch_search(queries, tree_root, model):
    """批次處理多個查詢"""
    all_results = {}
    for query in queries:
        results = tree_search(tree_root, query, model, 100, 20)
        all_results[query] = results
    return all_results

queries = [
    "土地法相關問題",
    "民法總則規定", 
    "都市計畫法條文"
]

batch_results = batch_search(queries, tree_root, model)
```

### 3. 結果後處理

```python
def process_results(results, max_results=5):
    """處理和過濾檢索結果"""
    # 去重
    unique_results = list(set(results))
    
    # 長度過濾
    filtered_results = [r for r in unique_results if len(r.strip()) > 20]
    
    # 限制數量
    return filtered_results[:max_results]

processed_results = process_results(results)
```

## 📝 更新附註（最近變更）

- 外部化參數：現在可透過環境變數調整，不需改碼。
  - LLM：`OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_TEMPERATURE`、`OPENAI_TOP_P`、`OPENAI_MAX_TOKENS`
  - Embedding：`EMBEDDING_MODEL_NAME`
  - 檢索：`CHUNK_SIZE`、`CHUNK_OVERLAP`、`MAX_CHUNKS`、`MAX_RESULTS`、`TOP_K`
  - Rerank：`RERANKER_ENABLE_IN_PIPELINE`（預設 false）、`RERANKER_USE_CROSS_ENCODER`（預設 false）、`RERANKER_MODEL_NAME`
  - API：`CORS_ORIGINS`、`API_TITLE`

- Rerank 管線開關：
  - 當 `RERANKER_ENABLE_IN_PIPELINE=true` 且檢索結果數量 > `MAX_RESULTS` 時，系統會自動對候選結果進行重排序。
  - 若同時設定 `RERANKER_USE_CROSS_ENCODER=true`，會改用 Cross-Encoder 進行配對打分重排（可透過 `RERANKER_MODEL_NAME` 指定模型）。

- .env 範例：

```bash
OPENAI_API_KEY=sk-xxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
OPENAI_TOP_P=0.9
OPENAI_MAX_TOKENS=4096
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
CHUNK_SIZE=150
CHUNK_OVERLAP=50
MAX_CHUNKS=12
MAX_RESULTS=100
TOP_K=15
RERANKER_ENABLE_IN_PIPELINE=true
RERANKER_USE_CROSS_ENCODER=true
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
CORS_ORIGINS=http://localhost:3000,https://your.domain
API_TITLE=Hierarchical RAG API
```

## 🔧 配置參數

### 詞嵌入模型配置

```python
# 預設使用 intfloat/multilingual-e5-large
# 您也可以使用其他 Sentence Transformers 模型

from sentence_transformers import SentenceTransformer

# 自定義模型
custom_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 在檢索中使用
results = tree_search(tree_root, query, custom_model, 100, 20)
```

### 系統參數調整

```python
# 針對不同場景的參數建議

# 精確檢索（較慢但更準確）
results = tree_search(
    tree_root, query, model,
    chunk_size=50,      # 較小的分塊
    chunk_overlap=10,   # 較小的重疊
    max_chunks=5        # 較少的分塊數
)

# 快速檢索（較快但可能遺漏細節）
results = tree_search(
    tree_root, query, model,
    chunk_size=200,     # 較大的分塊
    chunk_overlap=40,   # 較大的重疊
    max_chunks=15       # 較多的分塊數
)
```

## 📊 效能優化建議

### 1. 記憶體管理

```python
# 對於大型文本庫，建議分批處理
def process_large_corpus(texts, batch_size=1000):
    trees = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_vectors = model.encode(batch)
        tree = create_ahc_tree(batch_vectors, batch)
        trees.append(tree)
    return trees
```

### 2. 快取機制

```python
import os

# 檢查是否已有建立好的檢索樹
tree_file = "retrieval_tree.pkl"
if os.path.exists(tree_file):
    tree_root = load_tree(tree_file)
else:
    tree_root = create_ahc_tree(vectors, texts)
    save_tree(tree_root, tree_file)
```

## 🔍 實際應用案例

### 法律文件檢索

```python
# 法律條文檢索系統
legal_texts = [
    "民法第一條：民事，法律所未規定者，依習慣...",
    "刑法第十條：稱以上、以下、以內、以外者...",
    # ... 更多法條
]

# 建立法律檢索系統
legal_vectors = model.encode(legal_texts)
legal_tree = create_ahc_tree(legal_vectors, legal_texts)

# 查詢法律問題
legal_query = "關於契約的法律效力規定"
legal_results = tree_search(legal_tree, legal_query, model, 100, 20)
```

### 學術論文檢索

```python
# 學術文獻檢索
papers = [
    "本研究探討機器學習在自然語言處理中的應用...",
    "深度學習模型在圖像識別領域的最新進展...",
    # ... 更多論文摘要
]

academic_vectors = model.encode(papers)
academic_tree = create_ahc_tree(academic_vectors, papers)

research_query = "transformer模型在文本分類的效果"
academic_results = tree_search(academic_tree, research_query, model, 150, 30)
```

## 🔬 API 參考

### Web 應用 API

- `GET /`: 主頁面
- `GET /available-texts`: 獲取可用的文本列表
- `POST /query`: 提交查詢請求
  - 請求體：
    ```json
    {
        "query": "您的問題",
        "use_extraction": true/false,
        "text_name": "文本名稱",
        "prompt_type": "task_oriented" | "cot"
    }
    ```
  - 響應：
    ```json
    {
        "answer": "系統回答",
        "retrieved_docs": ["檢索到的文檔1", "檢索到的文檔2", ...]
    }
    ```

### Python API

#### 核心檢索函數

```python
# 主要檢索函數
from src.retrieval import create_ahc_tree, tree_search, save_tree, load_tree

# 多層檢索函數
from src.retrieval import (
    build_multi_level_index_from_files,
    multi_level_tree_search,
    multi_level_extraction_tree_search
)

# 查詢提取函數
from src.retrieval import extraction_tree_search

# 工具函數
from src.utils import WordEmbedding, RagChunking
from src.data_processing import DataDealer
```

## 🐛 常見問題與解決方案

### Q: 檢索結果不夠精確？
**A:** 嘗試調整參數：
- 減少 `chunk_size` 提高精度
- 增加 `max_chunks` 獲得更多候選結果
- 使用查詢提取功能處理複雜問題

### Q: 處理速度較慢？
**A:** 優化建議：
- 增加 `chunk_size` 減少分塊數量
- 減少 `max_chunks` 限制處理範圍
- 使用多層索引代替單一檢索樹

### Q: 記憶體使用過多？
**A:** 記憶體管理：
- 分批處理大型文本庫
- 定期清理不需要的變數
- 使用生成器而非列表存儲大量資料

### Q: 如何處理不同語言的文本？
**A:** 多語言支援：
- 使用多語言嵌入模型（如預設的 multilingual-e5-large）
- 確保查詢語言與文本語言一致
- 考慮使用語言特定的分詞策略

## 📄 系統功能說明

### 檢索流程

本系統提供兩種檢索模式：

1. **直接檢索** - 適合簡單明確的問題
   - 將用戶輸入直接向量化
   - 通過檢索樹尋找相似文本片段
   - 使用語言模型生成答案

2. **查詢提取檢索** - 適合複雜或冗長問題
   - 先使用語言模型提取核心法律問題和概念
   - 將提取後的關鍵要點向量化
   - 通過檢索樹查找相關片段
   - 使用語言模型針對提取要點生成精確答案

### 回答方式說明

#### 任務導向（Task-Oriented）
- 特點：簡潔直接，快速提供答案
- 適用：需要明確法條解釋或操作指引的問題

#### 思維鏈（Chain of Thought, CoT）
- 特點：詳細分析，提供推理過程
- 適用：複雜法律邏輯分析或需要多步推論的問題

## 📦 部署與發布

### PyPI 安裝

```bash
pip install hierarchical-rag-retrieval
```

### 從 GitHub 安裝開發版本

```bash
pip install git+https://github.com/arthur422tp/hierarchical.git
```

### 本機開發安裝

```bash
# 克隆專案
git clone https://github.com/arthur422tp/hierarchical.git
cd hierarchical

# 安裝開發依賴
pip install -e .[dev]
```

## 📚 更多資源

- **GitHub Repository**: https://github.com/arthur422tp/hierarchical
- **arXiv 論文**: https://arxiv.org/abs/2506.13607
- **PyPI 套件**: https://pypi.org/project/hierarchical-rag-retrieval/
- **Issue 回報**: https://github.com/arthur422tp/hierarchical/issues

## 🤝 貢獻與支援

歡迎提交 Issue 和 Pull Request！如果您在使用過程中遇到任何問題，或有改進建議，請隨時聯繫我們。

### 開發環境設置

```bash
# 克隆專案
git clone https://github.com/arthur422tp/hierarchical.git
cd hierarchical

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝開發依賴
pip install -e .[dev,app]
```

## 📜 License

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 📞 聯繫方式

- 作者：arthur422tp
- Email：arthur422tp@gmail.com
- GitHub：https://github.com/arthur422tp

---

**祝您使用愉快！如果這個系統對您的專案有幫助，請考慮給我們一個 ⭐ Star！**


