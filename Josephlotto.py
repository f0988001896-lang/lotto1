import os
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
# 自動讀取 repo 內的 Excel
# ===============================
DATA_FILE = "data.xlsx"  # 👈 你 GitHub repo 裡的檔名

st.sidebar.header("資料來源")
st.sidebar.caption("✅ 若 repo 內有 data.xlsx 會自動讀；沒有才需要上傳")

if os.path.exists(DATA_FILE):
    st.sidebar.success(f"自動讀取：{DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
else:
    st.sidebar.warning("repo 找不到 data.xlsx，請上傳一次")
    uploaded_file = st.file_uploader("上傳 Excel / CSV", type=["xlsx", "csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

# ===============================
# 接下來才是你的 AI 分析
# ===============================


# ===============================
# 2) 讀取資料
# ===============================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        source_type = "CSV"
    else:
        df = pd.read_excel(uploaded_file)
        source_type = "Excel"
except Exception as e:
    st.error(f"❌ 讀取失敗：{e}")
    st.stop()

required_cols = ["日期", "號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]
if not all(c in df.columns for c in required_cols):
    st.error("❌ 欄位不對，必須包含：日期、號碼1～號碼5")
    st.stop()

# 清理
df = df[required_cols].copy()
df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
for c in ["號碼1","號碼2","號碼3","號碼4","號碼5"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().copy()
df[["號碼1","號碼2","號碼3","號碼4","號碼5"]] = df[["號碼1","號碼2","號碼3","號碼4","號碼5"]].astype(int)

# 過濾不合理列
def valid_row(r):
    nums = [r["號碼1"], r["號碼2"], r["號碼3"], r["號碼4"], r["號碼5"]]
    return (len(set(nums)) == 5) and all(1 <= x <= 39 for x in nums)

df = df[df.apply(valid_row, axis=1)].copy()
df = df.sort_values("日期").reset_index(drop=True)

history_old2new = df[["號碼1","號碼2","號碼3","號碼4","號碼5"]].to_numpy(dtype=int)
history_new2old = history_old2new[::-1]

st.success(f"✅ 資料讀取成功：{source_type}｜總期數：{len(df)}｜最新日期：{df['日期'].max().date()}")

# ===============================
# 3) AI 評分（穩定版）
#    - 出現次數 * 10
#    - 遺漏期數（最近沒出現越久加權）
# ===============================
def build_score_table(hist_new_to_old: np.ndarray) -> pd.DataFrame:
    n_draws = len(hist_new_to_old)
    nums = np.arange(1, 40)

    # 出現次數
    freq = {n: int(np.count_nonzero(hist_new_to_old == n)) for n in nums}

    # 遺漏期數：從最新開始往回找第一次出現
    miss = {}
    for n in nums:
        idx = np.where((hist_new_to_old == n).any(axis=1))[0]
        miss[n] = int(idx[0]) if idx.size > 0 else n_draws

    # 你可調參數
    score = {}
    for n in nums:
        score[n] = freq[n] * 10 + miss[n] * 3  # 這裡是核心權重：miss*3
    out = pd.DataFrame({
        "號碼": list(score.keys()),
        "出現次數": [freq[n] for n in nums],
        "遺漏期數": [miss[n] for n in nums],
        "信心分": [score[n] for n in nums],
    }).sort_values("信心分", ascending=False).reset_index(drop=True)
    return out

score_df = build_score_table(history_new2old)

TOP_K = 10
top10 = score_df.head(TOP_K).copy()
top10_list = top10["號碼"].astype(int).tolist()

# ===============================
# 顯示 10 碼
# ===============================
st.write("---")
st.subheader("🌟 AI 核心推薦號碼（10 碼）")
cols = st.columns(10)
for i, r in top10.iterrows():
    n = int(r["號碼"])
    cols[i].metric(label=f"{n:02d}", value=int(r["信心分"]))

st.caption("📌 本系統完全依賴你上傳的資料檔；資料一樣→結果一定一樣。")

# ===============================
# 4) 視覺化：散點圖
# ===============================
st.write("---")
st.subheader("📊 號碼動能分佈圖")
fig_scatter = px.scatter(
    score_df,
    x="遺漏期數",
    y="信心分",
    size="出現次數",
    color="信心分",
    hover_name="號碼",
    title="橫軸：多久沒出現｜圓越大：總出現越多｜顏色越深：信心分越高",
    color_continuous_scale="YlOrBr"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ===============================
# (1) 回測：中2 / 中3 隨時間曲線
# ===============================
st.write("---")
st.subheader("🧪 回測（中2 / 中3 隨時間變化）")

left, mid, right = st.columns([1,1,2])
with left:
    window = st.slider("回測視窗 window（用前幾期推下一期）", 10, 200, 30, step=5)
with mid:
    roll = st.slider("滾動窗口（多少期平均）", 10, 120, 30, step=5)
with right:
    st.caption("做法：每一期用前 window 期計分→選出10碼→對比下一期真實5碼，計算命中數。")

def pick_top10_from_past(past_old_to_new: np.ndarray) -> list[int]:
    past_new_to_old = past_old_to_new[::-1]
    s = build_score_table(past_new_to_old)
    return s.head(10)["號碼"].astype(int).tolist()

def run_backtest(hist_old_to_new: np.ndarray, dates_old_to_new: np.ndarray, window: int) -> pd.DataFrame:
    n = len(hist_old_to_new)
    if n <= window:
        return pd.DataFrame()

    rows = []
    for i in range(window, n):
        past = hist_old_to_new[i-window:i]
        pred10 = set(pick_top10_from_past(past))
        real = set(hist_old_to_new[i])
        hit = len(pred10 & real)

        rows.append({
            "日期": pd.to_datetime(dates_old_to_new[i]),
            "命中數": hit,
            "中2": 1 if hit >= 2 else 0,
            "中3": 1 if hit >= 3 else 0,
            "預測10碼": ",".join(f"{x:02d}" for x in sorted(pred10)),
            "真實5碼": ",".join(f"{x:02d}" for x in sorted(real)),
        })

    out = pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)
    return out

bt = run_backtest(history_old2new, df["日期"].to_numpy(), window)

if bt.empty:
    st.warning("⚠️ 期數不足，請把 window 調小或確認資料期數。")
else:
    # 指標
    k1, k2, k3 = st.columns(3)
    k1.metric("平均命中數", f"{bt['命中數'].mean():.2f}")
    k2.metric("≥2 命中率", f"{(bt['命中數']>=2).mean()*100:.1f}%")
    k3.metric("≥3 命中率", f"{(bt['命中數']>=3).mean()*100:.1f}%")

    # 滾動命中率
    bt["中2_rolling"] = bt["中2"].rolling(roll, min_periods=max(5, roll//3)).mean()
    bt["中3_rolling"] = bt["中3"].rolling(roll, min_periods=max(5, roll//3)).mean()

    plot_df = bt[["日期","中2_rolling","中3_rolling"]].dropna().copy()
    plot_long = plot_df.melt(id_vars="日期", var_name="指標", value_name="滾動命中率")
    plot_long["指標"] = plot_long["指標"].replace({
        "中2_rolling":"中2（≥2）滾動命中率",
        "中3_rolling":"中3（≥3）滾動命中率"
    })

    fig_line = px.line(
        plot_long,
        x="日期",
        y="滾動命中率",
        color="指標",
        title=f"中2 / 中3 滾動命中率（roll={roll}）"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📋 回測明細（最近 50 期）")
    st.dataframe(bt.tail(50), use_container_width=True)

# ===============================
# (3) 匯出投注單 / 回測結果 Excel
# ===============================
st.write("---")
st.subheader("⬇️ 匯出 Excel（投注單 & 回測明細）")

# 投注單 DataFrame
bet_df = pd.DataFrame({
    "推薦10碼(排序)": [",".join(f"{x:02d}" for x in sorted(top10_list))],
    "產生時間": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
    "資料最新日期": [str(df["日期"].max().date())],
    "使用期數": [len(df)],
})

colA, colB = st.columns(2)

with colA:
    st.download_button(
        label="📥 下載：投注單（Excel）",
        data=to_excel_bytes(bet_df, sheet_name="投注單"),
        file_name="539_投注單.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with colB:
    if not bt.empty:
        st.download_button(
            label="📥 下載：回測明細（Excel）",
            data=to_excel_bytes(bt, sheet_name="回測"),
            file_name="539_回測明細.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("回測資料不足（bt 空），先調小 window 或上傳更多期數。")

st.write("---")
st.subheader("📄 原始資料預覽（最新 20 期）")
st.dataframe(df.sort_values("日期", ascending=False).head(20), use_container_width=True)
