import os
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===============================
# 基本設定
# ===============================
st.set_page_config(page_title="今彩 539 AI 智慧決策系統（Excel 穩定版）", layout="wide")

st.markdown("""
<style>
.big-title { font-size:48px; font-weight:800; }
.subtle { color:#666; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🏆 今彩 539 AI 智慧決策系統（Excel 穩定版）</div>", unsafe_allow_html=True)
st.write("---")

# ===============================
# 固定讀取 repo 內檔案
# ===============================
DATA_FILE = "data.xlsx"   # ✅ 你把合併檔放在 repo，同層命名 data.xlsx

st.sidebar.header("📌 資料來源")
st.sidebar.caption("本版本：只讀 repo 內 data.xlsx（不需上傳、不爬蟲、不隨機）")

if not os.path.exists(DATA_FILE):
    st.error(
        f"❌ 找不到檔案：{DATA_FILE}\n\n"
        "請把你的 Excel 檔放到 GitHub repo（與 Josephlotto.py 同一層），並命名為：data.xlsx\n\n"
        "部署到 Streamlit Cloud 後就會自動讀取。"
    )
    st.stop()

# ===============================
# 讀取 Excel（需要 openpyxl）
# ===============================
try:
    df = pd.read_excel(DATA_FILE)
except Exception as e:
    st.error(f"❌ Excel 讀取失敗：{e}\n\n"
             "若在 Streamlit Cloud：請確認 requirements.txt 有 openpyxl")
    st.stop()

# ===============================
# 工具：下載 Excel
# ===============================
def to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# ===============================
# 清理 & 檢查欄位
# ===============================
required_cols = ["日期", "號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ 欄位不對，缺少：{missing}\n\n必須包含：日期、號碼1～號碼5")
    st.stop()

work = df[required_cols].copy()
work["日期"] = pd.to_datetime(work["日期"], errors="coerce")

for c in ["號碼1", "號碼2", "號碼3", "號碼4", "號碼5"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work = work.dropna().copy()
work[["號碼1","號碼2","號碼3","號碼4","號碼5"]] = work[["號碼1","號碼2","號碼3","號碼4","號碼5"]].astype(int)

def valid_row(r):
    nums = [r["號碼1"], r["號碼2"], r["號碼3"], r["號碼4"], r["號碼5"]]
    return (len(set(nums)) == 5) and all(1 <= x <= 39 for x in nums)

work = work[work.apply(valid_row, axis=1)].copy()
work = work.sort_values("日期").reset_index(drop=True)

if len(work) < 60:
    st.warning(f"⚠️ 期數只有 {len(work)}，建議至少 100+ 期回測比較穩。")

# history：舊→新、以及新→舊
history_old2new = work[["號碼1","號碼2","號碼3","號碼4","號碼5"]].to_numpy(dtype=int)
history_new2old = history_old2new[::-1]

st.sidebar.success("✅ data.xlsx 讀取成功")
st.sidebar.write(f"期數：**{len(work)}**")
st.sidebar.write(f"最新日期：**{work['日期'].max().date()}**")
st.sidebar.write(f"最早日期：**{work['日期'].min().date()}**")

# ===============================
# AI 核心：評分表（可調權重）
# ===============================
def build_score_table(hist_new_to_old: np.ndarray, w_freq=10, w_miss=3) -> pd.DataFrame:
    n_draws = len(hist_new_to_old)
    nums = np.arange(1, 40)

    freq = {n: int(np.count_nonzero(hist_new_to_old == n)) for n in nums}

    miss = {}
    for n in nums:
        idx = np.where((hist_new_to_old == n).any(axis=1))[0]
        miss[n] = int(idx[0]) if idx.size > 0 else n_draws

    score = {n: freq[n] * w_freq + miss[n] * w_miss for n in nums}

    out = pd.DataFrame({
        "號碼": list(score.keys()),
        "出現次數": [freq[n] for n in nums],
        "遺漏期數": [miss[n] for n in nums],
        "信心分": [score[n] for n in nums],
    }).sort_values("信心分", ascending=False).reset_index(drop=True)

    return out

st.write("### ⚙️ 權重設定（你可微調）")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    w_freq = st.slider("出現次數權重", 1, 30, 10)
with c2:
    w_miss = st.slider("遺漏期數權重", 1, 30, 3)
with c3:
    st.caption("信心分 = 出現次數*w_freq + 遺漏期數*w_miss（遺漏越久越加分）")

score_df = build_score_table(history_new2old, w_freq=w_freq, w_miss=w_miss)

# ===============================
# 顯示：10 碼推薦
# ===============================
TOPK = 10
top10 = score_df.head(TOPK).copy()
top10_list = top10["號碼"].astype(int).tolist()

st.write("---")
st.subheader("🌟 AI 核心推薦號碼（10 碼）")
cols = st.columns(10)
for i, r in top10.iterrows():
    cols[i].metric(label=f"{int(r['號碼']):02d}", value=int(r["信心分"]))

st.caption("📌 檔案不變 → 推薦結果固定；本版本不爬蟲、不 random。")

# ===============================
# 視覺化：動能分佈圖
# ===============================
st.write("---")
st.subheader("📊 號碼動能分佈圖（遺漏期數 vs 信心分）")
fig_scatter = px.scatter(
    score_df,
    x="遺漏期數",
    y="信心分",
    size="出現次數",
    color="信心分",
    hover_name="號碼",
    title="圓越大=出現越多｜顏色越深=信心越高",
    color_continuous_scale="YlOrBr"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ===============================
# 回測：每期用前 window 期預測 10 碼，對比下一期真實
# ===============================
st.write("---")
st.subheader("🧪 回測（中2 / 中3 隨時間變化）")

a, b, c = st.columns([1,1,2])
with a:
    window = st.slider("回測視窗 window（用前 N 期推下一期）", 10, 250, 30, step=5)
with b:
    roll = st.slider("滾動平均窗口（命中率平滑）", 10, 150, 30, step=5)
with c:
    st.caption("做法：用前 window 期計分→選10碼→對比下一期開獎5碼→計算命中數（≥2、≥3）。")

def pick_top10_from_past(past_old_to_new: np.ndarray) -> list[int]:
    past_new_to_old = past_old_to_new[::-1]
    s = build_score_table(past_new_to_old, w_freq=w_freq, w_miss=w_miss)
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

bt = run_backtest(history_old2new, work["日期"].to_numpy(), window)

if bt.empty:
    st.warning("⚠️ 期數不足，請把 window 調小或增加資料期數。")
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("平均命中數", f"{bt['命中數'].mean():.2f}")
    k2.metric("≥2 命中率", f"{(bt['命中數']>=2).mean()*100:.1f}%")
    k3.metric("≥3 命中率", f"{(bt['命中數']>=3).mean()*100:.1f}%")

    bt["中2_rolling"] = bt["中2"].rolling(roll, min_periods=max(5, roll//3)).mean()
    bt["中3_rolling"] = bt["中3"].rolling(roll, min_periods=max(5, roll//3)).mean()

    plot_df = bt[["日期","中2_rolling","中3_rolling"]].dropna().copy()
    plot_long = plot_df.melt(id_vars="日期", var_name="指標", value_name="滾動命中率")
    plot_long["指標"] = plot_long["指標"].replace({
        "中2_rolling": "中2（≥2）滾動命中率",
        "中3_rolling": "中3（≥3）滾動命中率",
    })

    fig_line = px.line(
        plot_long,
        x="日期",
        y="滾動命中率",
        color="指標",
        title=f"中2 / 中3 滾動命中率（roll={roll}）"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("📋 回測明細（最近 80 期）", expanded=False):
        st.dataframe(bt.tail(80), use_container_width=True)

# ===============================
# 匯出 Excel：投注單 + 回測明細
# ===============================
st.write("---")
st.subheader("⬇️ 匯出 Excel（投注單 / 回測明細）")

bet_df = pd.DataFrame({
    "推薦10碼(排序)": [",".join(f"{x:02d}" for x in sorted(top10_list))],
    "產生時間": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
    "資料最新日期": [str(work["日期"].max().date())],
    "使用期數": [len(work)],
    "權重_w_freq": [w_freq],
    "權重_w_miss": [w_miss],
})

cA, cB = st.columns(2)
with cA:
    st.download_button(
        label="📥 下載：投注單（Excel）",
        data=to_excel_bytes(bet_df, sheet_name="投注單"),
        file_name="539_投注單.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with cB:
    if not bt.empty:
        st.download_button(
            label="📥 下載：回測明細（Excel）",
            data=to_excel_bytes(bt, sheet_name="回測"),
            file_name="539_回測明細.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("回測資料不足（bt 空），先調小 window 或增加資料期數。")

st.write("---")
st.subheader("📄 原始資料預覽（最新 20 期）")
st.dataframe(work.sort_values("日期", ascending=False).head(20), use_container_width=True)
