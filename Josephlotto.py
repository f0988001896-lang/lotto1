import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px

# =========================
# 0) Streamlit 基本設定
# =========================
st.set_page_config(page_title="今彩 539 AI 智慧決策系統", layout="wide")
st.markdown("""
<style>
.big-font { font-size:30px !important; color: #D4AF37; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 今彩 539 AI 智慧決策系統")
st.write("---")

if st.button("🔄 重新抓資料（清快取）"):
    st.cache_data.clear()
    st.rerun()


# =========================
# 1) 抓 pilio 539：改用 <table> 解析（重點修正）
# =========================
@st.cache_data(ttl=3600)
def get_data_pilio_table(min_periods: int = 60, fallback_periods: int = 200):
    """
    回傳：
      data: np.ndarray shape=(n,5)  (預期 n >= 100)
      source: str
    規則：
      - 解析所有 <tr>，從每列抽出整數
      - 過濾出 1~39 的號碼
      - 若該列有 >=5 個合法號碼：取最後 5 個當開獎號碼
      - 去重保序
    """
    url = "https://www.pilio.idv.tw/lto539/list.asp"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=20)
        res.raise_for_status()
        res.encoding = "big5"
        soup = BeautifulSoup(res.text, "html.parser")

        draws = []
        tables = soup.find_all("table")
        if not tables:
            # 沒表格就退回（理論上不會）
            data = np.random.randint(1, 40, size=(fallback_periods, 5), dtype=int)
            return data, "random(fallback; no table found)"

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if not tds:
                    continue

                nums = []
                for td in tds:
                    txt = td.get_text(strip=True)
                    # 找出 td 內所有整數（避免混雜文字）
                    for token in txt.replace("\xa0", " ").split():
                        if token.isdigit():
                            v = int(token)
                            if 1 <= v <= 39:
                                nums.append(v)
                        else:
                            # 有些格子可能像 "01" 或 "1," 或 "1/2" 之類，改用更寬鬆抓法
                            cleaned = "".join(ch for ch in token if ch.isdigit())
                            if cleaned.isdigit():
                                v = int(cleaned)
                                if 1 <= v <= 39:
                                    nums.append(v)

                # 如果一列有很多數字（常見：期別/日期 + 5個獎號）
                if len(nums) >= 5:
                    pick = tuple(nums[-5:])  # 取最後 5 個最穩
                    # 基本檢查：五個號碼要互不相同（539 一期不會重複）
                    if len(set(pick)) == 5:
                        draws.append(pick)

        # 去重保序（避免同一期被抓到兩次）
        if draws:
            draws = list(dict.fromkeys(draws))

        # 如果抓到太少，顯示明確訊息並 fallback（或你也可以改成 stop）
        if len(draws) < min_periods:
            data = np.random.randint(1, 40, size=(fallback_periods, 5), dtype=int)
            return data, f"random(fallback; pilio table got {len(draws)} < {min_periods})"

        data = np.array(draws, dtype=int)
        if data.ndim != 2 or data.shape[1] != 5:
            data = np.random.randint(1, 40, size=(fallback_periods, 5), dtype=int)
            return data, "random(fallback; table parse shape error)"

        return data, f"pilio_table(real; periods={len(data)})"

    except Exception as e:
        data = np.random.randint(1, 40, size=(fallback_periods, 5), dtype=int)
        return data, f"random(fallback; pilio error: {e})"


# =========================
# 2) 特徵 / 打分（你的規則保留）
# =========================
def build_features(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"號碼": range(1, 40)})
    df["出現次數"] = df["號碼"].apply(lambda x: int(np.count_nonzero(data == x)))

    def omission(x: int) -> int:
        rows = np.where((data == x).any(axis=1))[0]
        return int(rows[0]) if rows.size > 0 else 30

    df["遺漏期數"] = df["號碼"].apply(omission)

    df["AI 信心分"] = (df["出現次數"] * 10) + df["遺漏期數"].apply(
        lambda t: 50 if 7 <= t <= 12 else 10
    )
    return df


def pick_numbers(history_slice: np.ndarray, top_k: int = 10) -> list[int]:
    df = build_features(history_slice)
    top = df.sort_values("AI 信心分", ascending=False).head(top_k)
    return [int(x) for x in top["號碼"].tolist()]


# =========================
# 3) 回測：每期預測 10 碼 + 中2/中3
# =========================
def run_backtest(full_history: np.ndarray, window: int = 30, top_k: int = 10) -> pd.DataFrame:
    full_history = np.array(full_history)

    if full_history.size == 0:
        return pd.DataFrame()
    if full_history.ndim != 2 or full_history.shape[1] != 5:
        return pd.DataFrame()
    if len(full_history) <= window:
        return pd.DataFrame()

    # full_history[0] 當作最新 → 倒過來 old->new
    hist = full_history[::-1]

    records = []
    for i in range(window, len(hist)):
        past = hist[i - window:i]  # old->new
        real = set(hist[i])

        # build_features 期待「最新在前」，所以 past 反轉成 new->old
        pred = set(pick_numbers(past[::-1], top_k=top_k))
        hit = len(pred & real)

        records.append({
            "t": i,
            "命中數": hit,
            "中2": 1 if hit >= 2 else 0,
            "中3": 1 if hit >= 3 else 0,
            "預測10碼": ",".join(f"{n:02d}" for n in sorted(pred)),
            "真實5碼": ",".join(f"{n:02d}" for n in sorted(real)),
        })

    return pd.DataFrame(records)


# =========================
# 4) 主流程
# =========================
history, source = get_data_pilio_table(min_periods=60, fallback_periods=200)

st.info(f"📌 資料來源：{source}")
st.write("📌 history shape:", np.array(history).shape, "｜期數 len:", len(history))

TOP_K = 10

# 10碼推薦（用目前抓到的歷史算）
df_all = build_features(history)
top10 = df_all.sort_values("AI 信心分", ascending=False).head(TOP_K)

st.subheader("🌟 AI 核心推薦號碼（10碼）")
cols = st.columns(TOP_K)
for i, (_, row) in enumerate(top10.iterrows()):
    cols[i].markdown(f"<p class='big-font'>{int(row['號碼']):02d}</p>", unsafe_allow_html=True)
    cols[i].metric("信心分", int(row["AI 信心分"]))

st.write("---")
st.subheader("📊 號碼動能分佈圖")
fig = px.scatter(
    df_all,
    x="遺漏期數",
    y="AI 信心分",
    size="出現次數",
    color="AI 信心分",
    hover_name="號碼",
    title="圓圈越大代表近期出現越頻繁",
    color_continuous_scale="YlOrBr"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# 5) 回測 + 趨勢圖
# =========================
st.write("---")
st.subheader("🧪 回測（每期預測 10 碼）")

a, b, c = st.columns([1, 1, 2])
with a:
    window = st.slider("回測視窗 window", 10, 120, 30, step=5)
with b:
    roll = st.slider("趨勢圖滾動窗口（期）", 10, 100, 30, step=5)
with c:
    st.caption("回測：用前 window 期推 10 碼，對比下一期真實開獎。趨勢圖顯示中2/中3的滾動命中率。")

bt = run_backtest(history, window=window, top_k=TOP_K)

if bt.empty:
    st.warning("⚠️ 回測結果為空（期數不足或資料格式異常）。")
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

    bt["中2_rolling"] = bt["中2"].rolling(roll, min_periods=max(5, roll // 3)).mean()
    bt["中3_rolling"] = bt["中3"].rolling(roll, min_periods=max(5, roll // 3)).mean()

    trend = bt[["t", "中2_rolling", "中3_rolling"]].dropna().copy()
    trend = trend.rename(columns={"中2_rolling": "中2滾動命中率", "中3_rolling": "中3滾動命中率"})
    trend_long = trend.melt(id_vars="t", var_name="指標", value_name="滾動命中率")

    st.subheader("📈 中2 / 中3 滾動命中率趨勢圖")
    fig2 = px.line(trend_long, x="t", y="滾動命中率", color="指標",
                   title=f"滾動窗口：{roll} 期")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 回測明細（最近 50 期）")
    st.dataframe(bt.tail(50), use_container_width=True)




