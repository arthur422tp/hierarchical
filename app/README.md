# RAG 法律問答系統

這是一個基於 FastAPI 的 RAG（Retrieval-Augmented Generation）法律問答系統的前端界面。

## 功能特點

- 簡潔現代的用戶界面
- 支持查詢提取功能
- 實時顯示檢索結果
- 響應式設計，支持各種設備

## 安裝步驟

1. 確保已安裝所有依賴：
```bash
pip install -r requirements.txt
```

2. 運行應用程序：
```bash
cd app
uvicorn main:app --reload
```

3. 在瀏覽器中訪問：
```
http://localhost:8000
```

## 使用說明

1. 在文本框中輸入您的法律問題
2. 可選：勾選"使用查詢提取功能"以啟用查詢提取
3. 點擊"提交問題"按鈕
4. 等待系統處理並顯示結果

## API 端點

- `GET /`: 主頁面
- `POST /query`: 處理查詢請求
  - 請求體：
    ```json
    {
        "query": "您的問題",
        "use_extraction": true/false
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