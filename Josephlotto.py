import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# 0) Streamlit 基本設定
# =========================
st.set_page_config(page_title="今彩 539 AI 智慧決策系統", layout="wide")
st.markdown("""
<style>
.big-font { font-size:30px !important; color: #D4AF37; font-weight: bold; text-align:center; }
.small-muted { color:#666; font-size:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 今彩 539 AI 智慧決策系統（Excel 穩定版）")
st.write("---")

# =========================
# 1) 讀取合併後 Excel（你的檔案）
# =========================
DATA_FILE = "今彩539_2025_2026合併_修正版.xlsx"

@st.cache_data
def load_history_from_excel(path: str):
    df = pd.read_excel(path)

    required_cols = ["日期", "號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少欄位：{missing}（需要：{required_cols}）")

    # 日期
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

    # 號碼（你檔案裡是兩位數字串，例如 '05'，先轉 int）
    num_cols = ["號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["日期"] + num_cols).copy()
    df[num_cols] = df[num_cols].astype(int)

    # 去除不合理列（539 一期 5 號碼都在 1~39 且不重複）
    df = df[
        df[num_cols].apply(lambda r: all(1 <= x <= 39 for x in r) and len(set(r)) == 5, axis=1)
    ].copy()

    # 排序：舊→新（方便回測），另外準備「最新在前」給你算遺漏期數
    df = df.sort_values("日期").reset_index(drop=True)
    history_old_to_new = df[num_cols].to_numpy(dtype=int)
    dates_old_to_new = df["日期"].to_numpy()

    history_new_to_old = history_old_to_new[::-1]  # 最新在前
    dates_new_to_old = dates_old_to_new[::-1]

    return df, history_old_to_new, dates_old_to_new, history_new_to_old, dates_new_to_old


# 頁面上給你手動選檔名（避免部署時檔名不一致）
st.sidebar.header("資料來源")
data_file = st.sidebar.text_input("Excel 檔名", value=DATA_FILE)

try:
    df_full, hist_old2new, dates_old2new, hist_new2old, dates_new2old = load_history_from_excel(data_file)
except Exception as e:
    st.error(f"❌ 讀取資料失敗：{e}")
    st.stop()

st.info(f"📌 資料來源：{data_file}｜總期數：{len(df_full)}｜最新日期：{df_full['日期'].max().date()}")

# =========================
# 2) AI 分析（你的規則：出現次數 + 遺漏期數加權）
# =========================
def build_features(history_new_to_old: np.ndarray) -> pd.DataFrame:
    """
    history_new_to_old: shape (n,5) 且 [0] 是最新一期
    """
    n_draws = len(history_new_to_old)
    df = pd.DataFrame({"號碼": range(1, 40)})

    # 出現次數
    df["出現次數"] = df["號碼"].apply(lambda x: int(np.count_nonzero(history_new_to_old == x)))

    # 遺漏期數：從最新開始往回找，第一次出現的位置
    def omission(x: int) -> int:
        rows = np.where((history_new_to_old == x).any(axis=1))[0]
        return int(rows[0]) if rows.size > 0 else n_draws  # 沒出現就給 n_draws

    df["遺漏期數"] = df["號碼"].apply(omission)

    # 你的 AI 信心分規則（可自行調參）
    df["AI 信心分"] = (df["出現次數"] * 10) + df["遺漏期數"].apply(
        lambda t: 50 if 7 <= t <= 12 else 10
    )

    return df


df_score = build_features(hist_new2old)

TOP_K = 10
top10 = df_score.sort_values("AI 信心分", ascending=False).head(TOP_K)

# =========================
# 3) 顯示 10 碼推薦（不拆組）
# =========================
st.subheader("🌟 AI 核心推薦號碼（10 碼）")

cols = st.columns(TOP_K)
for i, (_, row) in enumerate(top10.iterrows()):
    num = int(row["號碼"])
    score = int(row["AI 信心分"])
    miss = int(row["遺漏期數"])
    freq = int(row["出現次數"])

    cols[i].markdown(f"<div class='big-font'>{num:02d}</div>", unsafe_allow_html=True)
    cols[i].metric("信心分", score)
    cols[i].markdown(f"<div class='small-muted'>遺漏 {miss}｜出現 {freq}</div>", unsafe_allow_html=True)

st.write("---")

# =========================
# 4) 視覺化
# =========================
st.subheader("📊 號碼動能分佈圖")
fig = px.scatter(
    df_score,
    x="遺漏期數",
    y="AI 信心分",
    size="出現次數",
    color="AI 信心分",
    hover_name="號碼",
    title="圓圈越大代表整體出現越多｜橫軸越大代表越久沒出現",
    color_continuous_scale="YlOrBr",
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# =========================
# 5) 回測（每期用前 window 期 → 預測下一期 10 碼）
# =========================
def pick_numbers(past_new_to_old: np.ndarray, top_k: int = 10) -> list[int]:
    df = build_features(past_new_to_old)
    top = df.sort_values("AI 信心分", ascending=False).head(top_k)
    return [int(x) for x in top["號碼"].tolist()]


def run_backtest(hist_old_to_new: np.ndarray, window: int = 30, top_k: int = 10) -> pd.DataFrame:
    """
    hist_old_to_new: shape (n,5) 且最後一列是最新一期
    """
    hist = np.array(hist_old_to_new)
    n = len(hist)
    if n <= window:
        return pd.DataFrame()

    records = []
    # i 指向「真實要驗證的那一期」
    for i in range(window, n):
        past_old_to_new = hist[i - window:i]  # old -> new
        real = set(hist[i])

        # 我們 build_features 的輸入要 new -> old，所以 past 反轉
        past_new_to_old = past_old_to_new[::-1]
        pred = set(pick_numbers(past_new_to_old, top_k=top_k))

        hit = len(pred & real)

        records.append({
            "期序(由舊到新索引)": i,
            "命中數": hit,
            "中2": 1 if hit >= 2 else 0,
            "中3": 1 if hit >= 3 else 0,
            "預測10碼": ",".join(f"{n:02d}" for n in sorted(pred)),
            "真實5碼": ",".join(f"{n:02d}" for n in sorted(real)),
        })

    return pd.DataFrame(records)


st.subheader("🧪 回測（每期預測 10 碼）")
a, b, c = st.columns([1, 1, 2])

with a:
    window = st.slider("回測視窗 window（用前幾期預測下一期）", 10, 200, 30, step=5)
with b:
    roll = st.slider("趨勢圖滾動窗口（期）", 10, 120, 30, step=5)
with c:
    st.caption("回測使用完整資料（舊→新），每次用前 window 期推下一期的 10 碼，計算中2/中3命中率走勢。")

bt = run_backtest(hist_old2new, window=window, top_k=TOP_K)

if bt.empty:
    st.warning("⚠️ 期數不足，無法回測（請降低 window 或確認資料期數）。")
else:
    avg_hit = bt["命中數"].mean()
    hit1 = (bt["命中數"] >= 1).mean()
    hit2 = (bt["命中數"] >= 2).mean()
    hit3 = (bt["命中數"] >= 3).mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("平均命中數", f"{avg_hit:.2f}")
    k2.metric("≥1 命中率", f"{hit1*100:.1f}%")
    k3.metric("≥2 命中率", f"{hit2*100:.1f}%")
    k4.metric("≥3 命中率", f"{hit3*100:.1f}%")

    # 滾動命中率
    bt["中2_rolling"] = bt["中2"].rolling(roll, min_periods=max(5, roll//3)).mean()
    bt["中3_rolling"] = bt["中3"].rolling(roll, min_periods=max(5, roll//3)).mean()

    trend = bt[["期序(由舊到新索引)", "中2_rolling", "中3_rolling"]].dropna().copy()
    trend = trend.rename(columns={
        "中2_rolling": "中2滾動命中率",
        "中3_rolling": "中3滾動命中率"
    })
    trend_long = trend.melt(
        id_vars="期序(由舊到新索引)",
        var_name="指標",
        value_name="滾動命中率"
    )

    st.subheader("📈 中2 / 中3 滾動命中率趨勢圖")
    fig2 = px.line(
        trend_long,
        x="期序(由舊到新索引)",
        y="滾動命中率",
        color="指標",
        title=f"滾動窗口：{roll} 期（越高代表那段時間策略越穩）"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 回測明細（最近 50 期）")
    st.dataframe(bt.tail(50), use_container_width=True)

st.write("---")
st.subheader("📄 原始資料預覽（最新 20 期）")
st.dataframe(df_full.sort_values("日期", ascending=False).head(20), use_container_width=True)
