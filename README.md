# Legal Document Retrieval System Based on Hierarchical Clustering （階層式聚類法規文本檢索系統）

### 基於階層式聚類與 RAG 的法規文本智慧檢索引擎

本系統是一個結合 AI 法律助手與法規查詢功能的智慧檢索引擎，核心技術為階層式聚類（Hierarchical Clustering）與餘弦相似度（Cosine Similarity），並透過 OpenAI 的 Retrieval-Augmented Generation（RAG）技術進行答案生成。適用於智慧律所、法律聊天機器人、學術研究或多語言法條索引場景，能有效提供準確、可解釋的法規查詢回覆。

arXiv論文連結：https://arxiv.org/abs/2506.13607

## AI-Powered Legal Document Retrieval Engine | Hierarchical Clustering & RAG 

This repository offers a high-accuracy legal document retrieval engine based on hierarchical clustering and cosine similarity, enhanced with RAG using OpenAI GPT. Suitable for AI-based legal assistants, legal chatbot systems, academic research tools, and multilingual law text indexing.

Paper is now in arXiv: https://arxiv.org/abs/2506.13607

## 📌 Features | 系統特色

- 🔍 **Hierarchical Clustering-based Retrieval Tree**：構建語意層次索引結構
- 🔁 **Dual Retrieval Modes**：支援直接檢索與查詢提取兩種模式
- 🧠 **RAG with OpenAI API**：結合語言模型生成精確法律回答
- 🧩 **Modular and Scalable**：可快速切換資料、部署方便
- ✅ **No manual `k` setting**：自動篩選所有相關文本
- 🌐 **Full-stack ready**：內建前端 UI + REST API
- 🐳 **Docker ready**：支援 Docker 容器化部署，一鍵啟動

## 🧭 System Overview | 系統概述

本系統為法律文件查詢提供了創新解法，適用於：
- 法律 AI 助理、智慧律所
- NLP 法規問答研究
- 學術或政府法規搜尋平台

支援中文法律文件（如民法、土地法等），後端以 FastAPI 構建，前端使用 HTML + Tailwind CSS。

本系統主要特點：

- **階層式檢索樹**：使用聚類方法自動構建文本向量的樹狀索引結構，實現高效檢索
- **語言層次**：保留了文本由具體到抽象的層次概念
- **雙模式檢索**：支援直接檢索與查詢提取兩種檢索模式
- **靈活適配**：針對複雜查詢與簡單查詢分別最佳化處理流程
- **無須設置k值**：自動回傳與問題有關的所有文本
- **易於部署**：提供完整的前後端解決方案，支援傳統部署與 Docker 容器化部署

## 🛠️ Technology Stack | 技術架構

| Component | Tech Used |
|----------|------------|
| Frontend | HTML / JavaScript / Tailwind CSS |
| Backend | FastAPI |
| Embedding Model | `intfloat/multilingual-e5-large` |
| Retrieval Tree | Hierarchical Clustering + Cosine Similarity |
| LLM API | OpenAI GPT (ChatGPT API) |
| Containerization | Docker & Docker Compose |

### 核心組件

- **前端**：HTML/JavaScript/Tailwind CSS 實現的互動介面
- **後端**：FastAPI 提供 RESTful API 服務
- **檢索引擎**：基於階層式聚類的向量檢索樹
- **語言模型**：使用 OpenAI API 進行查詢提取與答案生成
- **容器化**：支援 Docker 快速部署與擴展

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

## 🚀 快速開始

### 前置條件

- Python 3.8+ 或 Docker 環境
- OpenAI API 金鑰

### 方法一：傳統部署

#### 安裝依賴

```bash
pip install -r requirements.txt
```

#### 設置環境變數

在專案根目錄下創建 `.env` 檔案：

```
# OpenAI API金鑰
OPENAI_API_KEY=your_openai_api_key
```

#### 啟動應用

```bash
# Linux/Mac
chmod +x app/run.sh
./app/run.sh

# Windows
app\run.bat
```

### 方法二：Docker 部署 (推薦)

#### 前置條件
- 安裝 [Docker](https://www.docker.com/get-started) 和 [Docker Compose](https://docs.docker.com/compose/install/)

#### 設置環境變數
在專案根目錄下創建 `.env` 檔案：

```
# OpenAI API金鑰
OPENAI_API_KEY=your_openai_api_key
```

#### 使用 Docker Compose 啟動

```bash
# 建構並啟動容器
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# 查看日誌
docker-compose logs -f
```

應用啟動後，瀏覽器訪問 http://localhost:8000 使用系統。

### 使用現有數據

系統預設載入：
- `data/data_processed/民法總則.pkl` 與 `民法總則_embeddings.pkl`
- `data/data_processed/土地法與都市計畫法.pkl` 與 `土地法與都市計畫法_embeddings.pkl`

## 📚 使用自定義文本

要使用自己的法規文本：

1. **準備文本**：將文本準備為分段格式，確保每段內容具有足夠上下文

2. **生成向量與文本檔案**：
```python
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 加載模型
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 將文本分段
texts = [...] # 您的文本分段列表

# 生成向量
vectors = model.encode(texts)

# 保存文本和向量
with open('data/data_processed/自定義文本.pkl', 'wb') as f:
    pickle.dump(texts, f)
with open('data/data_processed/自定義文本_embeddings.pkl', 'wb') as f:
    pickle.dump(vectors, f)
```

3. **重啟應用**：系統會自動檢測並載入新的文本檔案

## 🔍 系統功能使用

1. 從下拉選單選擇要檢索的法律文本
2. 輸入您的法律問題
3. 選擇是否使用查詢提取功能（複雜問題推薦使用）
4. 選擇回答方式：
   - 任務導向（簡潔直接）：適合需要快速得到明確答案的問題
   - 思維鏈（詳細分析）：適合需要詳細推理過程的複雜問題
5. 點擊「提交問題」按鈕
6. 系統將顯示檢索結果和基於檢索內容生成的答案

## 🔬 API 參考

- `GET /`: 主頁面
- `GET /available-texts`: 獲取可用的文本列表
- `POST /query`: 提交查詢
  - 參數：
    - `query`: 查詢文本
    - `use_extraction`: 是否使用查詢提取功能（布林值）
    - `text_name`: 要檢索的文本名稱
    - `prompt_type`: 回答方式（"task_oriented" 或 "cot"）

## 📝 回答方式說明

### 任務導向（Task-Oriented）
- 特點：簡潔直接，快速提供答案
- 適用場景：
  - 需要快速得到明確答案的問題
  - 簡單的法律概念查詢
  - 法條內容確認

### 思維鏈（Chain of Thought）
- 特點：詳細分析，提供完整的推理過程
- 適用場景：
  - 複雜的法律問題分析
  - 需要理解推理過程的問題
  - 多個法律概念交織的情況

## 🧠 階層式檢索樹原理

本系統的核心為階層式聚類檢索樹，其運作原理如下：

1. **向量化**：使用 Pre-trained Model 將文本轉換為高維向量
2. **階層聚類**：採用單鏈接（Single Linkage）方法構建聚類樹
3. **樹結構遍歷**：檢索時通過向量相似度定位最相似節點
4. **相關片段收集**：收集定位節點下所有文本片段

這種方法相比傳統的暴力檢索和 Faiss 索引，能更好地保留文本的語義結構和關聯關係。

## 📊 系統效能

- **處理能力**：支持同時處理多達數萬條文本片段
- **檢索精度**：通過階層式聚類提高語義相關性
- **響應時間**：典型查詢響應時間 2-5 秒（視查詢複雜度與文本規模而定）

## 🛠️ 進階配置

可在 `app/main.py` 中調整以下參數：

```python
# 檢索參數配置
chunk_size = 100      # 切分長查詢的大小
chunk_overlap = 40    # 切分重疊率
```

## 📝 注意事項

- 系統需要足夠的記憶體來處理大型文檔
- 確保 `.env` 文件已正確設置 API 金鑰
- 建議使用現代瀏覽器以獲得最佳體驗
- 查詢提取功能處理時間較長，但對複雜問題效果更佳
- 使用 Docker 部署時，請確保 Docker 和 Docker Compose 已正確安裝


