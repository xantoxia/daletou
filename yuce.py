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

# ------------------------------
# GitHub 历史数据加载
# ------------------------------
url = "https://raw.githubusercontent.com/xantoxia/daletou/main/data/history.csv"

@st.cache_data
def load_github_history():
    df = pd.read_csv(url)
    # 转换成你内部使用的格式：(前区, 后区)
    return [(row[:5].tolist(), row[5:].tolist()) for _, row in df.iterrows()]

# ------------------------------
# 初始化 Session State（放 GitHub 数据）
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = load_github_history()

# ------------------------------
# 保存号码到内存
# ------------------------------
def add_new_result(front, back):
    st.session_state.history.append((front, back))

# ------------------------------
# 冷热号权重预测
# ------------------------------
def hot_cold_predict():
    history = st.session_state.history
    if len(history) == 0:
        return random_numbers()

    front_all, back_all = [], []

    for f, b in history:
        front_all += f
        back_all += b

    front_count = Counter(front_all)
    back_count = Counter(back_all)

    def make_probs(counter, total):
        arr = np.array([counter.get(i, 0) + 1 for i in range(1, total + 1)], dtype=float)
        return arr / arr.sum()

    front_probs = make_probs(front_count, 35)
    back_probs = make_probs(back_count, 12)

    front_pred = np.random.choice(range(1, 36), size=5, replace=False, p=front_probs)
    back_pred = np.random.choice(range(1, 13), size=2, replace=False, p=back_probs)

    return sorted(front_pred.tolist()), sorted(back_pred.tolist())

# ------------------------------
# 纯随机
# ------------------------------
def random_numbers():
    front = sorted(np.random.choice(range(1, 36), size=5, replace=False))
    back = sorted(np.random.choice(range(1, 13), size=2, replace=False))
    return front, back

# ------------------------------
# 构造 ML 数据集（简单示例）
# ------------------------------
def build_ml_dataset():
    data = []
    for f, b in st.session_state.history:
        row = f + b
        data.append(row)
    return np.array(data)

# ------------------------------
# LSTM 预测（预测均值作为参考）
# ------------------------------
def lstm_predict():
    data = build_ml_dataset()
    if data.shape[0] < 10:
        return None  

    X, y = data[:-1], data[1:]

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential([
        LSTM(32, activation="tanh", return_sequences=False),
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
# 可视化：冷热号 & 走势
# ------------------------------
def render_visualizations():
    history = st.session_state.history
    if len(history) == 0:
        st.info("没有历史数据，无法绘图。")
        return

    front_all, back_all = [], []
    for f, b in history:
        front_all += f
        back_all += b

    df_front = pd.DataFrame({"number": front_all})
    heat_fig = px.histogram(df_front, x="number", nbins=35, title="前区冷热号分布")
    st.plotly_chart(heat_fig, use_container_width=True)

    df_trend = pd.DataFrame([f + b for f, b in history])
    st.line_chart(df_trend)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎯 大乐透 AI 智能预测系统（Streamlit 云版）")
st.write("历史数据全部保存在 Streamlit Session State，可在云端持续运行。")

st.header("➕ 输入最新开奖号码")
nums = st.text_input("格式：1 5 9 22 33 3 11 (前5后2)")
btn = st.button("保存并更新模型")

if btn:
    try:
        parts = list(map(int, nums.split()))
        if len(parts) != 7:
            st.error("格式错误，需要 7 个数字！")
        else:
            add_new_result(parts[:5], parts[5:])
            st.success("已添加最新开奖号码！")
    except:
        st.error("请输入正确的数字格式")

st.header("📊 数据可视化")
render_visualizations()

st.header("🔮 预测结果")

hc = st.button("冷热号模型预测")
lstm_btn = st.button("LSTM 神经网络预测")
xgb_btn = st.button("XGBoost 预测")

if hc:
    f, b = hot_cold_predict()
    st.success(f"冷热号预测：前区 {f}  后区 {b}")

if lstm_btn:
    res = lstm_predict()
    if res:
        f, b = res
        st.success(f"LSTM 预测：前区 {f}  后区 {b}")
    else:
        st.error("历史数据不足（需要≥10期）")

if xgb_btn:
    res = xgb_predict()
    if res:
        f, b = res
        st.success(f"XGBoost 预测：前区 {f}  后区 {b}")
    else:
        st.error("历史数据不足（需要≥10期）")
