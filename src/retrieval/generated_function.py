"""
這裡蒐集生成與評測時所使用的函式
"""

from langchain.prompts import PromptTemplate
from langchain import LLMChain



class GeneratedFunction:

    def __init__(self):
        pass

    def query_extraction(self, query: str, llm):
        """
        Summary:
        這是一個提取query的函式

        query: str
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""

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

- 若案件事實或法律問題內容中直接提及的重要法律依據，完整標示為：
**「XX法第X條第X項」**。

⚠️ 特別注意：
- 嚴禁加入任何推論、補充、法條或自行延伸，僅可案件提取描述內容。
- 請保持結構清晰、條列明確，專業且客觀，符合法律書面標準。
- 所有法律條文及專有名詞須確保完整、正確，避免概念混淆。

以下為案件事實或法律問題內容：
{query}
""",
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        final_result = llm_chain.run(query=query)

        print("Final Result:\n", final_result)
        return final_result

    def LLM_Task_Oriented(self, query: str, llm, retrieved_docs: list) -> str:
        """
        使用任務導向方式生成回答。
        
        Args:
            query: 使用者查詢
            llm: 語言模型實例
            retrieved_docs: 檢索到的文檔列表
            
        Returns:
            str: 生成的回答
        """

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""
你將獲得以下兩個資訊：
- **法律問題:** {query}
- **檢索內容:** {context}

你是一個專業的法律顧問機器人，需根據法律問題和檢索內容提供準確且清晰的回答。

回答步驟：
1.確認關聯性
若問題與檢索內容相關，則分析並回答。
若無關，則忽略。
2.使用 Thought → Action → Observation 框架

Thought: 確認問題可否根據檢索內容回答。
Action: 從檢索內容中提取所有你認為解答所需之關鍵檢索內容或細化問題以確保準確性。
Observation: 提供清晰、準確的回答。
回應規則

僅使用檢索內容，不使用內部知識
如資訊不足，應明確指出缺少部分
無法回答時，直接回應：「我沒有足夠的相關資訊來回答這個問題」

  

    """,
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        final_result = llm_chain.run(context=retrieved_docs, query=query)

        print("Final Result:\n", final_result)
        return final_result

    def RAG_CoT(self, query: str, context: list, llm) -> str:
        """
        使用思維鏈方法生成答案，適合需要詳細分析的問題。
        
        Args:
            query: 原始使用者問題
            context: 檢索到的文本列表
            llm: 語言模型實例
            
        Returns:
            str: 根據思維鏈方式生成的詳細解答
        """
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""

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
""",
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        final_result = llm_chain.run(context=context, query=query)

        print("Final Result:\n", final_result)
        return final_result

    def LLM_benchmark(self, query, llm, retrieved_docs, answer1, answer2):
        """
        Summary:
        這是一個benchmark的語言模型
        
        query: str - 查詢問題
        llm: LLM - 語言模型
        retrieved_docs: list[str] - 檢索文檔
        answer1: str - 系統一的答案 
        answer2: str - 系統二的答案
        """
        prompt = PromptTemplate(
            input_variables=["context", "query", "answer_1", "answer_2"],
            template="""
        
            你是一名具有法律背景的專業評審，你的目標是根據提供的問題、檢索內容與答案，客觀、公正地評估兩個系統的效能。

### **輸入內容**
- **問題:** {query}
- **檢索內容:** {context}
- **系統一的答案:** {answer_1}
- **系統二的答案:** {answer_2}

### **評估標準**
請根據以下五個標準對兩個系統進行嚴格評分（1-5 分），並給出具體的評價理由：
1. **準確性**（是否正確涵蓋問題？答案是否與檢索內容一致？）
2. **全面性**（答案是否充分涵蓋問題的所有面向？細節是否足夠？）
3. **可信賴度**（答案與檢索內容的相關性如何？是否有不符合檢索內容的部分？）
4. **賦能性**（答案是否幫助使用者理解並做出合理判斷？）
5. **直接性**（答案是否清晰、直截了當地回應問題？是否避免冗長或模糊？）

### **評估方式**
1. **分步驟思考**：
   - 先分析檢索內容與問題的關聯性。
   - 再比較系統一與系統二的答案，確保評估基於客觀標準。
   - 對每個標準進行獨立評分，並提供具體例子支持你的評價。

2. **請依照以下格式輸出最終評估結果**
準確性 全面性 可信賴度 賦能性 直接性 系統一 X X X X X 系統二 X X X X X

### **評估原則**
- **嚴格遵循數據進行評估，不受系統先後順序影響。**
- **不因檢索內容的長度或答案的字數影響評價。**
- **確保評分有充分的理據支持，而非主觀判斷。**
- **對兩個系統一視同仁，確保客觀公平。**
- **若兩個系統的表現相當，則應明確指出「平手」，而非強行選擇勝者。**

請依據以上指引，公正客觀地評估系統效能，並產生評估結果。


    """,
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        who_win = llm_chain.run(
            context=retrieved_docs, query=query, answer_1=answer1, answer_2=answer2
        )

        print("final answer:\n", who_win)
        return who_win
