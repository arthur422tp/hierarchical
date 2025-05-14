# 使用官方 Python 映像
FROM python:3.11.7-slim

# 安裝系統依賴（如需可擴充）
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製需求檔並安裝 Python 套件
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 複製全部程式碼（包含 src/, app/, data/）
COPY . .

# 預設啟動指令：執行 FastAPI API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
