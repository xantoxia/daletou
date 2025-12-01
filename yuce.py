import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import random
import datetime

# ------------------------------
# GitHub 历史数据加载
# ------------------------------
url = "https://raw.githubusercontent.com/xantoxia/daletou/main/data/history.csv"

@st.cache_data
def load_github_history():
    try:
        df = pd.read_csv(url)
        df = df.dropna()

        # 强制类型转换
        df.iloc[:, :7] = df.iloc[:, :7].astype(int)

        # 日期列可选
        if df.shape[1] >= 8:
            df["date"] = pd.to_datetime(df.iloc[:, 7], errors="coerce")
        else:
            df["date"] = None

        # 转为内部格式
        result = []
        for _, row in df.iterrows():
            front = row[:5].tolist()
            back = row[5:7].tolist()
            date = row["date"]
            result.append((front, back, date))

        return result
    except:
        st.error("⚠ 无法从 GitHub 加载数据：请检查 CSV 格式。")
        return []

# ------------------------------
# 初始化历史数据
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_github_history()

# ------------------------------
# 保存开奖号码
# ------------------------------
def add_new_result(front, back, date):
    st.session_state.history.append((front, back, date))

# ------------------------------
# 冷热号预测
# ------------------------------
def hot_cold_predict():
    history = st.session_state.history
    if len(history) == 0:
        return random_numbers()

    front_all, back_all = [], []

    for f, b, _ in history:
        front_all += f
        back_all += b

    front_count = Counter(front_all)
    back_count = Counter(back_all)

    def make_probs(counter, total):
        arr = np.array([counter.get(i, 0) + 1 for i in range(1, total + 1)], float)
        return arr / arr.sum()

    front_probs = make_probs(front_count, 35)
    back_probs = make_probs(back_count, 12)

    front_pred = np.random.choice(range(1, 36), 5, replace=False, p=front_probs)
    back_pred = np.random.choice(range(1, 13), 2, replace=False, p=back_probs)

    return sorted(front_pred.tolist()), sorted(back_pred.tolist())

# ------------------------------
# 随机号码
# ------------------------------
def random_numbers():
    front = sorted(np.random.choice(range(1, 36), 5, replace=False))
    back = sorted(np.random.choice(range(1, 13), 2, replace=False))
    return front, back

# ------------------------------
# ML 数据集
# ------------------------------
def build_ml_dataset():
    data = []
    for f, b, _ in st.session_state.history:
        data.append(f + b)
    return np.array(data)

# ------------------------------
# LSTM 预测
# ------------------------------
def lstm_predict():
    data = build_ml_dataset()
    if data.shape[0] < 10:
        return None

    X, y = data[:-1], data[1:]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential([
        LSTM(32, activation="tanh"),
        Dense(7)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=4, verbose=0)

    pred = model.predict(X[-1].reshape(1, 1, 7))[0]

    front = sorted([min(max(int(x), 1), 35) for x in pred[:5]])
    back = sorted([min(max(int(x), 1), 12) for x in pred[5:]])
    return front, back

# ------------------------------
# XGBoost 预测
# ------------------------------
def xgb_predict():
    data = build_ml_dataset()
    if data.shape[0] < 10:
        return None

    X, y = data[:-1], data[1:]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    model.fit(X, y)

    pred = model.predict(X[-1].reshape(1, -1))[0]

    front = sorted([min(max(int(x), 1), 35) for x in pred[:5]])
    back = sorted([min(max(int(x), 1), 12) for x in pred[5:]])
    return front, back

# ------------------------------
# 可视化
# ------------------------------
def render_visualizations():
    history = st.session_state.history
    if len(history) == 0:
        st.info("没有历史数据")
        return

    # 冷热号
    front_all, back_all = [], []
    for f, b, _ in history:
        front_all += f
        back_all += b

    df_front = pd.DataFrame({"number": front_all})
    heat = px.histogram(df_front, x="number", nbins=35, title="前区冷热号")
    st.plotly_chart(heat, use_container_width=True)

    # 走势（带日期）
    df_trend = pd.DataFrame(
        [{"date": d, **{f"n{i+1}": v for i, v in enumerate(f + b)}} for f, b, d in history]
    )

    df_trend = df_trend.sort_values("date")
    st.line_chart(df_trend.set_index("date"))

# ------------------------------
# UI 页面
# ------------------------------
st.title("🎯 大乐透 AI 智能预测系统（日期版）")

# 输入开奖号码
st.header("➕ 输入最新开奖号码")
nums = st.text_input("格式：1 5 9 22 33 3 11")
date_input = st.date_input("开奖日期", value=datetime.date.today())
btn = st.button("保存到历史记录")

if btn:
    try:
        parts = list(map(int, nums.split()))
        if len(parts) != 7:
            st.error("必须输入 7 个数字（前 5 + 后 2）")
        else:
            add_new_result(parts[:5], parts[5:], date_input)
            st.success("已添加！")
    except:
        st.error("数字格式错误")

# 历史数据表格（可下载）
st.header("📄 历史记录")
df_show = pd.DataFrame(
    [{
        "date": d,
        "f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3], "f5": f[4],
        "b1": b[0], "b2": b[1],
    } for f, b, d in st.session_state.history]
)
st.dataframe(df_show)

csv = df_show.to_csv(index=False).encode("utf-8")
st.download_button("下载历史 CSV", csv, "history.csv", "text/csv")

# 可视化
st.header("📊 数据分析")
render_visualizations()

# 预测
st.header("🔮 预测结果")

if st.button("冷热号预测"):
    f, b = hot_cold_predict()
    st.success(f"前区 {f}   后区 {b}")

if st.button("LSTM 神经网络预测"):
    res = lstm_predict()
    if res:
        st.success(f"LSTM 预测：前区 {res[0]}  后区 {res[1]}")
    else:
        st.error("历史数据不足（≥10期）")

if st.button("XGBoost 预测"):
    res = xgb_predict()
    if res:
        st.success(f"XGBoost 预测：前区 {res[0]}  后区 {res[1]}")
    else:
        st.error("历史数据不足（≥10期）")
