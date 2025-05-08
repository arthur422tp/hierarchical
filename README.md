# 階層式聚類法規文本檢索系統

本系統在RAG（Retrieval-Augmented Generation）技術下進行修改，捨棄傳統的內積搜索（Inner Product Search），改使用基於階層式聚類（Hierarchical Clustering）與餘弦相似度（Cosine Similarity）建構的檢索樹，實現法規文本的高效搜尋，生成精確的答案。

## 📖 系統概述

本系統主要特點：

- **階層式檢索樹**：使用聚類方法自動構建文本向量的樹狀索引結構，實現高效檢索
- **雙模式檢索**：支援直接檢索與查詢提取兩種檢索模式
- **靈活適配**：針對複雜查詢與簡單查詢分別最佳化處理流程
- **無須設置k值**：自動回傳與問題有關的所有文本
- **易於部署**：提供完整的前後端解決方案，快速建立文本檢索演示

## 💻 技術架構

### 核心組件

- **前端**：HTML/JavaScript/Tailwind CSS 實現的互動介面
- **後端**：FastAPI 提供 RESTful API 服務
- **檢索引擎**：基於階層式聚類的向量檢索樹
- **語言模型**：使用 OpenAI API 進行查詢提取與答案生成

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

- Python 3.8+
- pip 套件管理器
- OpenAI API 金鑰

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 設置環境變數

在 `app` 目錄下創建 `.env` 檔案：

```
# OpenAI API金鑰
OPENAI_API_KEY=your_openai_api_key
```

### 使用現有數據

系統預設載入：
- `data/data_processed/民法總則.pkl` 與 `民法總則_embeddings.pkl`
- `data/data_processed/土地法與都市計畫法.pkl` 與 `土地法與都市計畫法_embeddings.pkl`

### 啟動應用

```bash
# Linux/Mac
chmod +x app/run.sh
./app/run.sh

# Windows
app\run.bat
```

應用啟動後，瀏覽器訪問 http://localhost:8000 使用系統。

## 📚 使用自定義文本

要使用自己的法規文本：

1. **準備文本**：將文本準備為分段格式，確保每段內容具有足夠上下文

2. **生成向量與文本檔案**：
```python
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 加載模型
model = SentenceTransformer('all-MiniLM-L6-v2')

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
4. 點擊「提交問題」按鈕
5. 系統將顯示檢索結果和基於檢索內容生成的答案

## 🔬 API 參考

- `GET /`: 主頁面
- `GET /available-texts`: 獲取可用的文本列表
- `POST /query`: 處理查詢請求
  - 請求體：
    ```json
    {
        "query": "您的問題",
        "use_extraction": true/false,
        "text_name": "文本名稱"
    }
    ```
  - 響應：
    ```json
    {
        "answer": "系統回答",
        "retrieved_docs": ["檢索到的文檔1", "檢索到的文檔2", ...]
    }
    ```

## 🧠 階層式檢索樹原理

本系統的核心為階層式聚類檢索樹，其運作原理如下：

1. **向量化**：使用 Sentence Transformers 將文本轉換為高維向量
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
chunk_size = 200      # 切分長查詢的大小
chunk_overlap = 40    # 切分重疊率
```

## 📝 注意事項

- 系統需要足夠的記憶體來處理大型文檔
- 確保 `.env` 文件已正確設置 API 金鑰
- 建議使用現代瀏覽器以獲得最佳體驗
- 查詢提取功能處理時間較長，但對複雜問題效果更佳
