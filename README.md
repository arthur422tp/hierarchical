# Hierarchical Clustering Retrieval System - 以餘弦相似度為度量的階層式法規文本檢索系統。

本系統基於階層式聚類（Hierarchical Clustering），結合餘弦相似度（Cosine Similarity）進行文本向量的聚類，能有效處理法規文本的檢索需求。
使用者可以根據查詢直接進行樹檢索，或先透過語言模型進行關鍵字擷取後再檢索，靈活應對不同應用場景。
## 📦 使用方法

```bash
#安裝所需套件
pip install -r requirements.txt
```
```
import src.retrieval.generated_function as gf
import src.retrieval.RAGTree_function as rf
import src.utils.query_retrieval as qr
import src.utils.word_chunking as wc
import src.utils.word_embedding as we
import pickle

#準備好文本，初始化Chunking
chunk = wc.RagChunking(text)
text = chunk.text_chunking(chunk_size: int, chunk_overlap: int)

#初始化Embedding
word_embedding = we.WordEmbedding()
#讀入預設模型(可以改成自己喜歡的model)
model = word_embedding.load_model()

#建構檢索樹
tree = rf.create_ahc_tree(embeddings, text)
```
## 我們提供了兩種檢索方法
### 1.直接進行樹檢索
```
#方法一：直接使用檢索樹檢索，若query過長，會將其切chunk
tree_search_result = rf.tree_search(tree, query, model, chunk_size: int, chunk_overlap: int)
output = gf.GeneratedFunction.RAG_LLM_chain(query, llm, tree_search_result)
```
### 2.Query Extraction後再進行檢索
```
#方法二：先讓query進一層語言模型做extraction，再進入檢索樹檢索
QE_tree_search_result = gf.GeneratedFunction.extraction_tree_search(root, query, model, chunk_size: int, chunk_overlap: int, llm, max_chunks=10)
output = gf.GeneratedFunction.RAG_cluster_LLM(query, QE_tree_search_result, llm)
```
## 📦 架構
```
src/
├── data_processing/
│   ├── __init__.py
│   ├── data_dealer.py
├── retrieval/
│   ├── __init__.py
│   ├── generated_function.py
│   ├── RAGTree_function.py
├── utils/
│   ├── __init__.py
│   ├── query_retrieval.py
│   ├── word_chunking.py
│   ├── word_embedding.py

```

### Data_dealer提供將.md轉成txt、讀取.txt等功能

# 範例
## 使用者提問：
```
甲授與代理權予乙，欲乙代理甲向丙購買A物一件，若乙係以自己名義向丙購買A物一件，此買賣效力是否及於甲？又丁授與代理權予戊，欲戊代理丁向庚購買B物一件，並給予戊授權書。
在戊未購買前，丁發現戊不能信賴，遂向戊表示撤回前開授權，惟忘記將授權書索回，旋戊出示授權書，以丁之名義向庚購買B物一件，則該買賣效力是否及於丁？
```
## QE+樹檢索回答：
```
Final Result:
 【Step 1】明確界定核心法律問題  
本案的核心法律問題分為兩部分：  
1. 乙以自己名義向丙購買A物的買賣效力是否及於甲？  
2. 戊在丁撤回授權後仍以丁名義向庚購買B物的買賣效力是否及於丁？

【Step 2】概念與法律地位辨析  
- **代理權**：指授權一方（授權者）允許另一方（代理人）以其名義進行法律行為的權限。  
- **授詢書**：授權者發出的書面文件，證明代理人擁有代理權。  
- **代理行為的效力**：通常，代理人在其權限內所為的法律行為，會直接對授權者（本人）發生效力（依據《民法》第 103 條）。  
- **撤回授權**：代理權可以隨時撤回，但代理人須將授權書交還授權者（依據《民法》第 109 條）。  

【Step 3】提取關鍵法律事實與條文  
1. 乙在未獲甲的允許下，以自己名義向丙購買A物。  
2. 丁撤回對戊的授權，但未能收回授權書，戊仍以丁名義向庚購買B物。  
3. 《民法》第 103 條、第 106 條和第 109 條與本案相關。  

【Step 4】逐步推理與法律適用  
1. **甲與乙的情形**：  
   乙以自己名義向丙購買A物，這樣的交易如果不在其代理權限內，則無法使甲負擔法律效果，因為根據《民法》第 106 條，代理人不得自行為本人與自己之法律行為。因此，該買賣效力不及於甲。  

2. **丁與戊的情形**：  
   丁已經通知戊撤回授權，根據《民法》第 109 條的規定，代理權的撤回應立即生效。然而，由於丁並未取回授權書，戊仍可憑著該授權書進行交易。根據《民法》第 103 條，即使撤回授權，若代理人（戊）的行為未違反法律規定，且第三人（庚）不知情，這筆交易仍可能對丁生效。然而，丁出的撤回授權行為可被認定為對抗第三人的根據，因此該交易的效力是否及於丁仍然存在爭議。

【Step 5】得出法律結論  
1. 乙以自己名義向丙購買A物的買賣效力不及於甲。  
2. 戊在丁撤回授權後出示授權書仍向庚購買B物，此買賣效力是否及於丁的問題具有爭議，因不明確是否戊的行為符合第三人善意的條件，故需根據具體情況進一步確認。該交易的效力可能受限於丁撤回授權的有效性及第三方的善意信賴。
```

# Prompt設計

## QE prompt
```
prompt = f"""
你是一位專業的法律助理，專責協助律師從案件事實或法律問題中，提取出核心法律事實、主要法律爭點，以及涉及的法律條文或當事人主張，並依下列結構化格式整理：

【背景核心事實】
- 條列案件或題目中出現的客觀事實，包括當事人身分、行為、處分或其他法律上重要事實。

【主要法律爭點】
- 條列案件或題目中需要解決的核心法律問題，包括法律關係、權利義務歸屬、法律行為效力、構成要件等。

【概念與專有名詞辨析】（如無混淆疑慮可省略）
- 若案件中出現易混淆或法律效果不同之相似名詞（如「無行為能力人」與「限制行為能力人」），請完整列出並分別定義，明確說明彼此差異與法律效果。

⚠️ 以下兩個區塊僅在有內容時產出，若無內容請完全省略，不得填「無」：

【當事人或機關主張】
- 條列當事人或機關在案件中明確表達的主張、法律見解或立場。
- 特別注意：若案件有引用法律條文，請完整標示為：
**「XX法第X條第X項」**，不得簡化、不得省略法條名稱與條次。

【涉及的法律條文】

- 案件中直接提及的重要法律依據，完整標示為：
**「XX法第X條第X項」**。

⚠️ 特別注意：
- 嚴禁加入任何推論、補充、法條或自行延伸，僅可案件提取描述內容。
- 請保持結構清晰、條列明確，專業且客觀，符合法律書面標準。
- 所有法律條文及專有名詞須確保完整、正確，避免概念混淆。

以下為案件事實或法律問題內容：
{query}
"""
```
## CoT設計
```
prompt = f"""
你將獲得以下資訊：
- 原始問題：{query}
- 檢索內容：{context}

你是一位專業法律顧問機器人，請依照以下【Chain of Thought（CoT）推理步驟】完整邏輯分析並作答：

【Step 1】明確界定核心法律問題
- 依原始問題清楚界定本案需解決的核心法律問題。

【Step 2】概念與法律地位辨析（此步驟必須執行）
- 針對檢索內容中出現的**專有法律概念、主體身分或專業術語**進行清楚辨析與定義。
- 如發現有**名稱相近但法律效果或法律地位不同的概念**（例如：無行為能力人 vs 限制行為能力人），請完整區分並標示，避免混淆。
- 所有專業概念請逐一定義，並說明彼此區別及法律效果差異。

【Step 3】提取關鍵法律事實與條文
- 條列與核心法律問題相關的法律事實、法條或案例。

【Step 4】逐步推理與法律適用
- 依照抽取出的事實與條文，進行條理清晰的法律推理。
- 如有爭議或多種見解，請分別說明。

【Step 5】得出法律結論
- 明確回答本案的法律問題。
- 若檢索內容不足，請直接回答：「依目前檢索內容，無法完整回答。」

⚠️ 特別規則：
- 嚴格依照檢索內容推理，禁止引入外部知識或假設。
- 法律條文請完整標示「XX法第 X 條第 X 項」。
- 保持用語精確，嚴防法律概念混淆。
"""
```
