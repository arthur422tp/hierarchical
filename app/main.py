from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv
import pickle
import sys
import numpy as np

# 添加專案根目錄到路徑，以便引入其他模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入配置
from app.config import (
    APP_DIR, STATIC_DIR, DATA_DIR,
    MAX_TOKENS, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS,
    CORS_ORIGINS, API_TITLE
)

# 導入檢索和生成模組
from src.utils.word_embedding import WordEmbedding
import src.retrieval.RAGTree_function as rf
import src.retrieval.generated_function as gf
from langchain_openai import ChatOpenAI

# 載入環境變數
load_dotenv()

app = FastAPI(title=API_TITLE)
# 建立 ChatOpenAI 的工廠函式，支援從環境變數覆寫參數
def _create_chat_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 環境變數未設置，請確認 .env 檔案")

    model = os.getenv("OPENAI_MODEL")
    temperature_str = os.getenv("OPENAI_TEMPERATURE")
    top_p_str = os.getenv("OPENAI_TOP_P")
    max_tokens_env = os.getenv("OPENAI_MAX_TOKENS")

    kwargs = {"api_key": api_key}
    # 僅在提供時設定，以沿用套件預設
    if model:
        kwargs["model"] = model
    if temperature_str:
        try:
            kwargs["temperature"] = float(temperature_str)
        except ValueError:
            pass
    if top_p_str:
        try:
            kwargs["top_p"] = float(top_p_str)
        except ValueError:
            pass
    # max_tokens 以環境優先，否則使用 config 的 MAX_TOKENS
    try:
        kwargs["max_tokens"] = int(max_tokens_env) if max_tokens_env else MAX_TOKENS
    except ValueError:
        kwargs["max_tokens"] = MAX_TOKENS

    return ChatOpenAI(**kwargs)

# 應用狀態
class AppState:
    model = None
    trees: Dict[str, Any] = {}
    llm = None

app.state.app_state = AppState()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    query: str
    use_extraction: bool = False
    text_name: str  # 添加文本名稱欄位
    prompt_type: str = "task_oriented"  # 新增 prompt 類型欄位，預設為 task_oriented

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]

class TextListResponse(BaseModel):
    available_texts: List[str]

# 初始化模型和語言模型
@app.on_event("startup")
async def startup_event():
    try:
        # 初始化embedding模型
        word_embedding = WordEmbedding()
        app.state.app_state.model = word_embedding.load_model()
        print("詞嵌入模型已載入")
        
        # 初始化語言模型
        try:
            app.state.app_state.llm = _create_chat_llm()
            print("OpenAI語言模型已載入")
        except ValueError as _:
            print("警告：OPENAI_API_KEY 環境變數未設置")
    except Exception as e:
        print(f"初始化模型時發生錯誤: {str(e)}")

def get_tree(text_name):
    """
    根據分開儲存的 embeddings 與 texts 建構檢索樹：
    - {text_name}_embeddings.pkl：儲存 vectors
    - {text_name}_texts.pkl：儲存 texts
    """
    if text_name not in app.state.app_state.trees:
        try:
            embeddings_path = os.path.join(DATA_DIR, f"{text_name}_embeddings.pkl")
            texts_path = os.path.join(DATA_DIR, f"{text_name}.pkl")

            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"找不到向量檔案: {embeddings_path}")
            if not os.path.exists(texts_path):
                raise FileNotFoundError(f"找不到文本檔案: {texts_path}")

            # 分別載入
            with open(embeddings_path, 'rb') as f:
                vectors = pickle.load(f)
            with open(texts_path, 'rb') as f:
                texts = pickle.load(f)

            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors)

            # 維度一致性檢查：避免查詢模型與已存向量維度不一致
            try:
                embedding_dim = vectors.shape[1]
            except Exception:
                raise ValueError(f"讀取到的向量格式異常，請確認檔案內容：{embeddings_path}")

            try:
                expected_dim = app.state.app_state.model.get_sentence_embedding_dimension()
            except Exception:
                # 後備：若模型無該方法，嘗試編碼一個樣本以取得維度
                expected_dim = int(app.state.app_state.model.encode(["dim_check"]).shape[1])

            if embedding_dim != expected_dim:
                raise ValueError(
                    "Embedding 維度不一致：\n"
                    f"- 檔案 {embeddings_path} 維度 = {embedding_dim}\n"
                    f"- 目前模型維度 = {expected_dim}\n\n"
                    "請調整環境變數 EMBEDDING_MODEL_NAME 以匹配原先建立向量所用模型，"
                    "或重新產生 embeddings 使其與目前模型一致。"
                )

            tree = rf.create_ahc_tree(vectors, texts)
            app.state.app_state.trees[text_name] = tree
            print(f"✅ 已建構檢索樹：{text_name}")
        except Exception as e:
            raise ValueError(f"❌ 建構檢索樹失敗 '{text_name}': {str(e)}")

    return app.state.app_state.trees[text_name]

# 獲取可用的文本列表
def get_available_texts():
    try:
        if not os.path.exists(DATA_DIR):
            print(f"警告：目錄不存在: {DATA_DIR}")
            return []

        files = os.listdir(DATA_DIR)
        texts = set(f.replace('.pkl', '') for f in files if f.endswith('.pkl'))
        embeddings = set(f.replace('_embeddings.pkl', '') for f in files if f.endswith('_embeddings.pkl'))

        available = sorted(list(texts & embeddings))  # 交集，確保兩個都存在
        return available
    except Exception as e:
        print(f"無法獲取文本列表: {str(e)}")
        return []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Static files not found. Please ensure the app/static directory exists with index.html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading static files: {str(e)}")

@app.get("/available-texts", response_model=TextListResponse)
async def get_texts():
    try:
        texts = get_available_texts()
        print(f"可用文本: {texts}")
        return TextListResponse(available_texts=texts)
    except Exception as e:
        print(f"獲取文本列表時發生錯誤: {str(e)}")
        # 如果出現錯誤，返回預設值
        return TextListResponse(available_texts=["民法總則", "土地法與都市計畫法"])

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # ✅ 防呆處理：強制將輸入轉為純文字
        def normalize_query(query):
            if isinstance(query, list):
                return " ".join(map(str, query))
            if not isinstance(query, str):
                return str(query)
            return query

        normalized_query = normalize_query(request.query)
        
        # 檢查模型是否已初始化
        if app.state.app_state.model is None:
            word_embedding = WordEmbedding()
            app.state.app_state.model = word_embedding.load_model()
            print("詞嵌入模型已重新載入")
        
        # 檢查語言模型是否已初始化
        if app.state.app_state.llm is None:
            app.state.app_state.llm = _create_chat_llm()
            print("語言模型已重新載入")
        
        print(f"接收查詢: {normalized_query}, 文本: {request.text_name}, 使用提取: {request.use_extraction}")

        # 獲取檢索樹
        tree = get_tree(request.text_name)

        start_time = __import__('time').time()
        
        # 創建 GeneratedFunction 實例
        generator = gf.GeneratedFunction()

        # 執行檢索與生成
        if request.use_extraction:
            print("使用提取方法進行檢索...")
            retrieved_docs = rf.extraction_tree_search(tree, normalized_query, app.state.app_state.model,
                                                     CHUNK_SIZE, CHUNK_OVERLAP, app.state.app_state.llm, MAX_CHUNKS)
            if request.prompt_type == "cot":
                answer = generator.RAG_CoT(normalized_query, retrieved_docs, app.state.app_state.llm)
            else:
                answer = generator.LLM_Task_Oriented(normalized_query, app.state.app_state.llm, retrieved_docs)
        else:
            print("使用直接檢索方法...")
            retrieved_docs = rf.tree_search(tree, normalized_query, app.state.app_state.model,
                                           CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS)
            if request.prompt_type == "cot":
                answer = generator.RAG_CoT(normalized_query, retrieved_docs, app.state.app_state.llm)
            else:
                answer = generator.LLM_Task_Oriented(normalized_query, app.state.app_state.llm, retrieved_docs)

        elapsed_time = __import__('time').time() - start_time
        print(f"檢索和生成完成，耗時: {elapsed_time:.2f}秒")

        return QueryResponse(
            answer=answer,
            retrieved_docs=retrieved_docs
        )

    except ValueError as e:
        print(f"值錯誤: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"處理查詢時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Ensure the static directory exists
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
        
    # 啟動時打印一些診斷信息
    print(f"APP_DIR: {APP_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"DATA_DIR 存在: {os.path.exists(DATA_DIR)}")
    if os.path.exists(DATA_DIR):
        print(f"DATA_DIR 內容: {os.listdir(DATA_DIR)}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 