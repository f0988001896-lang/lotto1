import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="今彩 539 AI 智慧決策系統（Excel 穩定版）",
    layout="wide"
)

st.title("🏆 今彩 539 AI 智慧決策系統（Excel 穩定版）")
st.write("---")

# ===============================
# 上傳檔案
# ===============================
uploaded_file = st.file_uploader(
    "📂 請上傳 今彩539 歷史資料（Excel 或 CSV）",
    type=["xlsx", "csv"]
)

if uploaded_file is None:
    st.info("⬆️ 請先上傳你剛剛整理好的 2025+2026 檔案")
    st.stop()

# ===============================
# 讀取資料
# ===============================
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

except Exception as e:
    st.error(f"❌ 讀取資料失敗：{e}")
    st.stop()

# ===============================
# 基本清理
# ===============================
required_cols = ["日期", "號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]
if not all(c in df.columns for c in required_cols):
    st.error("❌ 檔案欄位錯誤，必須包含：日期、號碼1～號碼5")
    st.stop()

df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
df = df.dropna(subset=["日期"])
df = df.sort_values("日期").reset_index(drop=True)

history = df[["號碼1","號碼2","號碼3","號碼4","號碼5"]].astype(int).values

st.success(f"✅ 資料讀取成功，共 {len(history)} 期")

# ===============================
# AI 分析
# ===============================
nums = np.arange(1, 40)
freq = {n: 0 for n in nums}
last_seen = {n: None for n in nums}

for i, draw in enumerate(history):
    for n in draw:
        freq[n] += 1
        last_seen[n] = i

total = len(history)
scores = {}

for n in nums:
    gap = total - last_seen[n] if last_seen[n] is not None else total
    scores[n] = freq[n] * 10 + gap * 3

score_df = pd.DataFrame({
    "號碼": list(scores.keys()),
    "信心分": list(scores.values())
}).sort_values("信心分", ascending=False)

top10 = score_df.head(10)

# ===============================
# 顯示結果
# ===============================
st.write("---")
st.subheader("🌟 AI 核心推薦號碼（10 碼）")

cols = st.columns(10)
for i, row in top10.iterrows():
    cols[list(top10.index).index(i)].metric(
        label=f"{int(row['號碼']):02d}",
        value=int(row["信心分"])
    )

st.caption("📌 本版本完全依賴你上傳的 Excel / CSV，不使用爬蟲、不使用隨機資料")
