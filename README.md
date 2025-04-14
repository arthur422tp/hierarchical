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
- 本案需解決的核心法律問題為：甲與丁在授權代理的情況下，乙與戊以各自名義購入物品的交易有效性及其對於甲和丁的效力。

【Step 2】概念與法律地位辨析
- **代理權**：指一方（授權者）授予另一方（代理人）代為處理事務的權利。
- **無行為能力人**：依據法律，無法獨立從事法律行為的人，其一切法律行為均無效。
- **限制行為能力人**：依據法律，雖然可以從事法律行為，但在某些行為上需要他人的同意或許可（如法定代理人的允許）。
- **善意第三人**：對於法律行為抱持善意並無過失的人，法律特別保護其權益，不受他人不法行為影響。

【Step 3】提取關鍵法律事實與條文
- 甲授與代理權於乙，乙向丙購買A物，若乙以自己名義購買，根據《民法》第104條，代理人所為之法律行為，其效力應由授權者承擔，假如代理人沒有超越授權的情況。
- 丁授予戊代理權並賦予授權書，但若丁發現戊不可信賴並撤回授權，根據《民法》第107條，代理權之限制及撤回不得對抗善意第三人，且《民法》第109條規定代理人不應留置授權書，需交還授權者。

【Step 4】逐步推理與法律適用
- **針對甲和乙的情形**：
  - 乙若在其授權範圍內以甲的名義購買A物，該買賣效力及於甲。
  - 若乙以自己名義購買，則該買賣一般視為不符合代理的要求，如無法證明乙在行為時的權限，則不會對甲產生法律效力。

- **針對丁和戊的情形**：
  - 丁在知道戊不可靠的情況下撤回授權，但未索回授權書，根據《民法》第107條，戊依舊可以以丁的名義與庚交易，因為戊在第三方（庚）的善意情況下使交易有效。
  - 若庚是善意的，他無法知道戊的授權已被撤回，則此交易對丁依然具有約束力。

【Step 5】得出法律結論
- 乙以自己名義購買A物的交易不及於甲，因未在授權範圍內。而戊向庚購買B物的交易，因戊未撤回的授權書對於善意的第三人庚有效，故該交易效力及於丁。

```
