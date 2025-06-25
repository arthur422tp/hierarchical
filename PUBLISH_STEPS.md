# PyPI 發布步驟指南

## 📋 發布前檢查清單

在發布之前，請確認以下項目：

- [x] 所有代碼已提交到GitHub
- [x] 版本號已更新
- [x] 依賴套件已正確配置
- [x] 文檔已完成
- [x] 測試通過

## 🔧 Step 1: 準備發布環境

```bash
# 安裝發布工具
pip install build twine

# 清理舊的建置檔案
rm -rf dist/ build/ *.egg-info/
```

## 📦 Step 2: 建置套件

```bash
# 建置套件
python -m build

# 檢查建置結果
ls dist/
# 應該看到：
# hierarchical_rag_retrieval-0.1.0-py3-none-any.whl
# hierarchical_rag_retrieval-0.1.0.tar.gz
```

## 🧪 Step 3: 先在TestPyPI測試（推薦）

### 3.1 註冊TestPyPI帳戶
1. 前往 https://test.pypi.org/
2. 註冊帳戶
3. 在 Account Settings 中生成 API Token
4. 複製Token（格式：`pypi-...`）

### 3.2 上傳到TestPyPI
```bash
# 上傳到測試環境
python -m twine upload --repository testpypi dist/*

# 輸入帳戶資訊：
# Username: __token__
# Password: 您的TestPyPI API Token
```

### 3.3 測試安裝
```bash
# 從TestPyPI安裝測試
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hierarchical-rag-retrieval

# 測試導入
python -c "from src.retrieval import create_ahc_tree; print('✅ 測試成功！')"
```

## 🎯 Step 4: 發布到正式PyPI

### 4.1 註冊PyPI帳戶
1. 前往 https://pypi.org/
2. 註冊帳戶
3. 在 Account Settings 中生成 API Token
4. 複製Token

### 4.2 正式發布
```bash
# 上傳到正式PyPI
python -m twine upload dist/*

# 輸入帳戶資訊：
# Username: __token__
# Password: 您的PyPI API Token
```

## ✅ Step 5: 驗證發布

```bash
# 安裝已發布的套件
pip install hierarchical-rag-retrieval

# 驗證安裝
python -c "
from src.retrieval import create_ahc_tree, tree_search
from src.utils import WordEmbedding
print('✅ 套件安裝成功！所有主要模組都可正常導入')
"
```

## 🏷️ Step 6: 創建GitHub Release

```bash
# 創建並推送標籤
git tag v0.1.0
git push origin v0.1.0
```

然後在GitHub上：
1. 前往 Releases 頁面
2. 點擊 "Create a new release"
3. 選擇標籤 v0.1.0
4. 填寫發布說明
5. 發布

## 🔄 後續版本更新流程

```bash
# 1. 更新版本號（在 src/__init__.py 和 setup.py）
# 2. 提交更改
git add .
git commit -m "Release v0.1.1"

# 3. 清理並重新建置
rm -rf dist/ build/ *.egg-info/
python -m build

# 4. 發布新版本
python -m twine upload dist/*

# 5. 創建新標籤
git tag v0.1.1
git push origin v0.1.1
```

## 🐛 常見問題解決

### Q: 上傳失敗 "File already exists"
**A:** PyPI不允許重複上傳相同版本，請：
- 更新版本號
- 重新建置
- 重新上傳

### Q: 導入錯誤
**A:** 檢查：
- 套件結構是否正確
- `__init__.py` 檔案是否存在
- 依賴套件是否正確安裝

### Q: 建置失敗
**A:** 檢查：
- `setup.py` 語法是否正確
- `pyproject.toml` 格式是否正確
- 所有必要檔案是否存在

## 📞 需要幫助？

如果遇到問題：
1. 檢查 PyPI 官方文檔：https://packaging.python.org/
2. 查看錯誤訊息詳細內容
3. 在GitHub Issues中提問

**恭喜！您的套件現在已經可以供全世界的開發者使用了！** 🎉 