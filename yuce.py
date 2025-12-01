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
from io import StringIO
import requests

# ------------------------------
# GitHub CSV 配置（读取）
# ------------------------------
GITHUB_CSV_URL = "https://raw.githubusercontent.com/xantoxia/daletou/main/data/history.csv"

# ------------------------------
# 历史数据加载
# ------------------------------
@st.cache_data
def load_github_history():
    try:
        response = requests.get(GITHUB_CSV_URL)
        response.encoding = 'utf-8-sig'
        df = pd.read_csv(StringIO(response.text))
        df = df.dropna()
        df.iloc[:, :7] = df.iloc[:, :7].astype(int)
        if df.shape[1] >= 8:
            df["date"] = pd.to_datetime(df.iloc[:, 7], errors="coerce")
        else:
            df["date"] = None

        result = []
        for _, row in df.iterrows():
            front = row[:5].tolist()
            back = row[5:7].tolist()
            date = row["date"]
            result.append((front, back, date))
        return result
    except Exception as e:
        st.error(f"⚠ 无法从 GitHub 加载数据：{e}")
        return []

# 初始化历史数据
if "history" not in st.session_state:
    st.session_state.history = load_github_history()

# 初始化预测历史
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# 初始化冷热号固定预测缓存
if "hotcold_fixed" not in st.session_state:
    st.session_state.hotcold_fixed = None

# ------------------------------
# 保存开奖号码到 session_state
# ------------------------------
def add_new_result(front, back, date):
    st.session_state.history.append((front, back, date))
    # 新数据加入后清除固定预测
    st.session_state.hotcold_fixed = None

# ------------------------------
# 保存预测记录
# ------------------------------
def save_prediction(front, back):
    today = datetime.date.today()
    st.session_state.pred_history.append({
        "date": today,
        "front": front,
        "back": back
    })

# ------------------------------
# 对比上次预测准确度
# ------------------------------
def compare_last_prediction(new_front, new_back):
    if len(st.session_state.pred_history) == 0:
        return None
    last_pred = st.session_state.pred_history[-1]
    front_hit = len(set(last_pred["front"]) & set(new_front))
    back_hit = len(set(last_pred["back"]) & set(new_back))
    return front_hit, back_hit

# ------------------------------
# 随机号码生成
# ------------------------------
def random_numbers():
    front = sorted(np.random.choice(range(1, 36), 5, replace=False))
    back = sorted(np.random.choice(range(1, 13), 2, replace=False))
    return front, back

# ------------------------------
# 冷热号预测（固定概率最大组合 + 自适应权重）
# ------------------------------
def hot_cold_predict():
    # 如果已有固定预测，直接返回
    if st.session_state.hotcold_fixed is not None:
        return st.session_state.hotcold_fixed

    history = st.session_state.history
    if len(history) == 0:
        front, back = random_numbers()
        st.session_state.hotcold_fixed = (front, back)
        return front, back

    front_all, back_all = [], []
    for f, b, _ in history:
        front_all += f
        back_all += b

    front_count = Counter(front_all)
    back_count = Counter(back_all)

    # 上次预测命中权重增加
    if len(st.session_state.pred_history) > 0:
        last_pred = st.session_state.pred_history[-1]
        for num in last_pred["front"]:
            front_count[num] += 1
        for num in last_pred["back"]:
            back_count[num] += 1

    # 生成概率
    front_probs = np.array([front_count.get(i,0)+1 for i in range(1,36)], dtype=float)
    front_probs /= front_probs.sum()
    back_probs = np.array([back_count.get(i,0)+1 for i in range(1,13)], dtype=float)
    back_probs /= back_probs.sum()

    # 取概率最高的号码
    front_dict = {i+1:p for i,p in enumerate(front_probs)}
    back_dict = {i+1:p for i,p in enumerate(back_probs)}

    front_pred = sorted(front_dict, key=lambda x: front_dict[x], reverse=True)[:5]
    back_pred = sorted(back_dict, key=lambda x: back_dict[x], reverse=True)[:2]

    st.session_state.hotcold_fixed = (front_pred, back_pred)
    return front_pred, back_pred

# ------------------------------
# 构建 ML 数据集
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
    X = X.reshape((X.shape[0],1,X.shape[1]))
    model = Sequential([LSTM(32, activation="tanh"), Dense(7)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=4, verbose=0)
    pred = model.predict(X[-1].reshape(1,1,7))[0]
    front = sorted([min(max(int(x),1),35) for x in pred[:5]])
    back = sorted([min(max(int(x),1),12) for x in pred[5:]])
    return front, back

# ------------------------------
# XGBoost 预测（CPU/GPU 自动选择）
# ------------------------------
def xgb_predict():
    data = build_ml_dataset()
    if data.shape[0] < 10:
        return None
    X, y = data[:-1], data[1:]
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        gpu_available = False

    if gpu_available:
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, tree_method='gpu_hist')
    else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, n_jobs=-1)

    model.fit(X, y)
    pred = model.predict(X[-1].reshape(1,-1))[0]
    front = sorted([min(max(int(x),1),35) for x in pred[:5]])
    back = sorted([min(max(int(x),1),12) for x in pred[5:]])
    return front, back

# ------------------------------
# 可视化
# ------------------------------
def render_visualizations():
    history = st.session_state.history
    if len(history) == 0:
        st.info("没有历史数据")
        return
    front_all, back_all = [], []
    for f,b,_ in history:
        front_all += f
        back_all += b
    df_front = pd.DataFrame({"number": front_all})
    heat = px.histogram(df_front, x="number", nbins=35, title="前区冷热号")
    st.plotly_chart(heat, use_container_width=True)

    df_trend = pd.DataFrame([{"date":d, **{f"n{i+1}":v for i,v in enumerate(f+b)}} for f,b,d in history])
    df_trend = df_trend.sort_values("date")
    st.line_chart(df_trend.set_index("date"))

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎯 大乐透 AI 智能预测系统（云端版，固定冷热号预测）")

# 输入开奖号码
nums = st.text_input("格式：1 5 9 22 33 3 11")
date_input = st.date_input("开奖日期", value=datetime.date.today())
btn = st.button("保存到历史记录（仅云端 session）")

if btn:
    try:
        parts = list(map(int, nums.split()))
        if len(parts) != 7:
            st.error("必须输入 7 个数字（前5+后2）")
        else:
            hits = compare_last_prediction(parts[:5], parts[5:])
            if hits:
                st.info(f"上次预测命中：前区 {hits[0]} 个号码，后区 {hits[1]} 个号码")
            add_new_result(parts[:5], parts[5:], date_input)
            st.success("已添加到云端 session！请下载 CSV 更新 GitHub")
    except:
        st.error("数字格式错误")

# 历史数据表格
df_show = pd.DataFrame([{"date":d,"f1":f[0],"f2":f[1],"f3":f[2],"f4":f[3],"f5":f[4],
                        "b1":b[0],"b2":b[1]} for f,b,d in st.session_state.history])
st.dataframe(df_show)
csv = df_show.to_csv(index=False).encode("utf-8-sig")
st.download_button("下载历史 CSV", csv, "history.csv", "text/csv")

# 可视化
st.header("📊 数据分析")
render_visualizations()

# 预测
st.header("🔮 预测结果")
if st.button("冷热号预测"):
    f,b = hot_cold_predict()
    save_prediction(f,b)
    st.success(f"前区 {f}   后区 {b}")

if st.button("LSTM 神经网络预测"):
    res = lstm_predict()
    if res:
        save_prediction(res[0], res[1])
        st.success(f"LSTM预测：前区 {res[0]}  后区 {res[1]}")
    else:
        st.error("历史数据不足（≥10期）")

if st.button("XGBoost 预测"):
    res = xgb_predict()
    if res:
        save_prediction(res[0], res[1])
        st.success(f"XGBoost预测：前区 {res[0]}  后区 {res[1]}")
    else:
        st.error("历史数据不足（≥10期）")
