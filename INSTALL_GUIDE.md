# Hierarchical RAG Retrieval 安裝與發布指南

## 📦 套件安裝指南

### 從PyPI安裝（發布後）

```bash
pip install hierarchical-rag-retrieval
```

### 從GitHub安裝（開發版本）

```bash
pip install git+https://github.com/yourusername/hierarchical.git
```

### 本機開發安裝

```bash
# 克隆專案
git clone https://github.com/yourusername/hierarchical.git
cd hierarchical

# 安裝開發依賴
pip install -e .[dev]
```

## 🚀 基本使用方法

安裝完成後，您可以這樣使用：

```python
# 導入主要模組
from src.retrieval import create_ahc_tree, tree_search
from src.utils import WordEmbedding

# 初始化模型
embedding_model = WordEmbedding()
model = embedding_model.load_model()

# 準備文本資料
texts = ["您的法律條文1", "您的法律條文2", ...]
vectors = model.encode(texts)

# 建立檢索樹
tree_root = create_ahc_tree(vectors, texts)

# 進行檢索
query = "您的查詢問題"
results = tree_search(tree_root, query, model, chunk_size=100, chunk_overlap=20)
```

## 📤 發布到PyPI指南

### 1. 準備工作

確保您已經完成以下設定：

1. **更新個人資訊**：編輯 `setup.py` 和 `pyproject.toml` 中的：
   - 作者姓名和郵箱
   - GitHub repository URL
   - 專案描述

2. **版本號管理**：更新 `src/__init__.py` 中的版本號

3. **測試套件**：確保所有功能正常運作

```bash
# 測試套件安裝
pip install -e .
python examples/basic_usage.py
```

### 2. 建置套件

```bash
# 安裝建置工具
pip install build twine

# 清理之前的建置檔案
rm -rf dist/ build/ *.egg-info/

# 建置套件
python -m build
```

### 3. 上傳到TestPyPI（測試）

```bash
# 註冊 TestPyPI 帳戶：https://test.pypi.org/
# 建立 API token

# 上傳到測試環境
python -m twine upload --repository testpypi dist/*

# 測試安裝
pip install --index-url https://test.pypi.org/simple/ hierarchical-rag-retrieval
```

### 4. 發布到PyPI（正式）

```bash
# 註冊 PyPI 帳戶：https://pypi.org/
# 建立 API token

# 上傳到正式環境
python -m twine upload dist/*
```

### 5. 驗證發布

```bash
# 安裝發布的套件
pip install hierarchical-rag-retrieval

# 測試導入
python -c "from src.retrieval import create_ahc_tree; print('安裝成功！')"
```

## 🔧 開發工作流程

### 本機開發設定

```bash
# 克隆專案
git clone https://github.com/yourusername/hierarchical.git
cd hierarchical

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝開發依賴
pip install -e .[dev,app]
```

### 版本更新流程

1. **更新版本號**：
   ```python
   # 編輯 src/__init__.py
   __version__ = "0.1.1"  # 新版本號
   ```

2. **測試變更**：
   ```bash
   python examples/basic_usage.py
   ```

3. **提交變更**：
   ```bash
   git add .
   git commit -m "Release v0.1.1"
   git tag v0.1.1
   git push origin main --tags
   ```

4. **重新建置和發布**：
   ```bash
   rm -rf dist/
   python -m build
   python -m twine upload dist/*
   ```

## 📋 檢查清單

發布前請確認：

- [ ] 所有測試通過
- [ ] 文檔更新完整
- [ ] 版本號已更新
- [ ] GitHub repository URL 正確
- [ ] 作者資訊正確
- [ ] 相依套件版本正確
- [ ] 範例程式碼可執行
- [ ] README.md 內容完整

## 🐛 常見問題

### Q: 上傳失敗，提示檔案已存在
A: PyPI 不允許覆蓋已存在的版本，請更新版本號後重新建置

### Q: 導入模組失敗
A: 檢查 `__init__.py` 檔案的導入路徑是否正確

### Q: 相依套件衝突
A: 檢查 `requirements.txt` 中的版本限制，使用更寬鬆的版本範圍

## 📞 支援

如果遇到問題，請在 GitHub Issues 中提出：
https://github.com/yourusername/hierarchical/issues 