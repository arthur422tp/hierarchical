# RAG法律文本檢索演示系統

這是一個基於RAG（檢索增強生成）的法律文本檢索演示系統，使用者可以選擇要檢索的法律文本，並選擇是否使用查詢提取功能來優化檢索效果。

## 系統架構

- **前端**：HTML/JavaScript/Tailwind CSS
- **後端**：FastAPI
- **檢索引擎**：基於階層式聚類的向量檢索系統
- **語言模型**：OpenAI GPT

## 開始使用

### 前置條件

- Python 3.8+
- pip
- OpenAI API金鑰

### 安裝依賴

```bash
pip install -r ../requirements.txt
pip install python-dotenv  # 安裝dotenv套件
```

### 設置API金鑰

在app目錄下創建一個名為`.env`的文件，並添加您的OpenAI API金鑰：

```
# OpenAI API金鑰
OPENAI_API_KEY=your_openai_api_key
```

將`your_openai_api_key`替換為您的實際API金鑰。

### 啟動應用

#### Linux/Mac:
```bash
chmod +x run.sh
./run.sh
```

#### Windows:
```
run.bat
```

應用啟動後，在瀏覽器中打開 http://localhost:8000 即可使用系統。

## 使用說明

1. 從下拉選單中選擇要檢索的法律文本（民法總則或土地法與都市計畫法）
2. 輸入您的法律問題
3. 選擇是否使用查詢提取功能（可提升檢索質量但處理較慢）
4. 點擊「提交問題」按鈕
5. 系統將顯示回答結果和檢索到的文檔

## 系統功能

- **文本選擇**：使用者可選擇不同的法律文本進行檢索
- **查詢提取**：可選擇是否使用查詢提取功能來優化檢索效果
- **檢索結果**：展示檢索到的相關文本片段
- **AI回答**：基於檢索結果生成回答

## 技術細節

本系統使用了以下技術：

- 文本嵌入：使用Sentence Transformers將文本轉換為向量
- 檢索引擎：基於階層式聚類樹的檢索系統
- 語言模型：使用OpenAI API進行查詢提取和回答生成
- 環境變數管理：使用python-dotenv加載API金鑰

## API 端點

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

## 注意事項

- 確保所有依賴都已正確安裝
- 系統需要足夠的內存來處理大型文檔
- 建議使用現代瀏覽器以獲得最佳體驗
- 確保`.env`文件已正確設置且不會被上傳到公共存儲庫（已在.gitignore中添加） 