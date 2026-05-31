import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import pytz

# Настройка страницы
st.set_page_config(page_title="ML Trading Predictor", layout="wide")
st.title("📈 AI Predictor: Движение цены AAPL (Moscow Time)")
st.markdown("Предсказание направления цены через 5 дней (дневной ТФ)")

# --- Загрузка и подготовка данных ---

@st.cache_data
def load_historical_data():
    try:
        # 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: имя файла
        df = pd.read_csv("data/aapl_5y_daily.csv", index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Ошибка: Файл data/aapl_5y_daily.csv не найден. Сначала выполните ноутбук NIRS_TMO.ipynb!")
        return None

def compute_features(df):
    df_feat = df.copy()
    df_feat['returns'] = df_feat['close'].pct_change()
    
    lags = [1, 2, 3, 5, 10, 20]
    for lag in lags:
        df_feat[f'close_lag_{lag}'] = df_feat['close'].shift(lag)
        df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)
        
    windows = [5, 10, 20]
    for window in windows:
        df_feat[f'ma_{window}'] = df_feat['close'].rolling(window).mean()
        df_feat[f'ma_ratio_{window}'] = df_feat['close'] / df_feat[f'ma_{window}'] - 1
        df_feat[f'volatility_{window}'] = df_feat['returns'].rolling(window).std()
        
    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    df_feat['rsi_14'] = compute_rsi(df_feat['close'], 14)
    df_feat['volume_ma_20'] = df_feat['volume'].rolling(20).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / df_feat['volume_ma_20']
    
    return df_feat.dropna()

# --- Интерфейс ---

st.sidebar.header("⚙️ Настройки модели (Gradient Boosting)")

n_estimators = st.sidebar.slider(
    "Количество деревьев (n_estimators)", 
    10, 200, 50, step=10,
    help=" **Больше деревьев** = выше точность, но медленнее работа. Риск переобучения при очень больших значениях."
)

learning_rate = st.sidebar.slider(
    "Скорость обучения (learning_rate)", 
    0.01, 0.2, 0.1, step=0.01,
    help="🏎️ **Ниже скорость** = модель учится аккуратнее, требует больше деревьев. **Выше** = быстрее, но может пропустить детали."
)

max_depth = st.sidebar.slider(
    "Глубина дерева (max_depth)", 
    2, 10, 5, step=1,
    help="🌳 **Глубже дерево** = улавливает сложные закономерности. **Мельче** = проще модель, меньше риск переобучения."
)

st.info("ℹ️ Изменение параметров выше приведет к переобучению модели на исторических данных.")

# Загрузка данных
df_hist = load_historical_data()

if df_hist is not None:
    # Подготовка признаков
    df_processed = compute_features(df_hist)
    
    feature_cols = [col for col in df_processed.columns if col not in ['returns']]
    X_full = df_processed[feature_cols]
    
    # 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: целевая переменная (без порога 0.2%)
    y_full = (df_processed['close'].shift(-5) > df_processed['close']).astype(int).dropna()
    
    min_len = min(len(X_full), len(y_full))
    X_full = X_full.iloc[:min_len]
    y_full = y_full.iloc[:min_len]
    
    scaler_app = StandardScaler()
    X_scaled = scaler_app.fit_transform(X_full)
    
    split_idx = int(len(X_scaled) * 0.9)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_full[:split_idx], y_full[split_idx:]
    
    # --- Обучение модели ---
    with st.spinner('Обучение модели...'):
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
    
    # --- Визуализация ---
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: "свечей" → "дней"
        st.subheader("📊 График цены (Последние 1000 дней)")
        last_n = 1000
        last_data = df_hist.tail(last_n).copy()
        
        # Конвертация времени в Московское
        moscow_tz = pytz.timezone('Europe/Moscow')
        if last_data.index.tz is None:
            last_data.index = last_data.index.tz_localize('UTC')
        last_data.index = last_data.index.tz_convert(moscow_tz)
        
        # Расчет процента изменения для каждой свечи для отображения в hover
        last_data['pct_change'] = ((last_data['close'] - last_data['open']) / last_data['open']) * 100
        
        # Формируем текст для hover
        hover_text = []
        for i, row in last_data.iterrows():
            pct = row['pct_change']
            color = "green" if pct >= 0 else "red"
            sign = "+" if pct >= 0 else ""
            # 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: формат даты без минут
            text = f"<b>Date:</b> {i.strftime('%Y-%m-%d')}<br>" \
                   f"<b>Open:</b> ${row['open']:.2f}<br>" \
                   f"<b>Close:</b> ${row['close']:.2f}<br>" \
                   f"<b>High:</b> ${row['high']:.2f}<br>" \
                   f"<b>Low:</b> ${row['low']:.2f}<br>" \
                   f"<b>Change:</b> <span style='color:{color}'>{sign}{pct:.2f}%</span>"
            hover_text.append(text)

        fig = go.Figure(data=[go.Candlestick(
            x=last_data.index,
            open=last_data['open'],
            high=last_data['high'],
            low=last_data['low'],
            close=last_data['close'],
            name="Price",
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000',
            hoverinfo='text',
            text=hover_text
        )])
        
        # 🔧 НИЧЕГО НЕ МЕНЯЕМ: твой рабочий layout
        fig.update_layout(
            height=500, 
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            dragmode='zoom',
            hovermode='x unified',
            
            yaxis=dict(
                title='Цена ($)',
                fixedrange=False,
                autorange=True,
                showspikes=True,
                spikecolor="grey",
                spikesnap="cursor",
                spikemode="across+toaxis",
                spikedash="dot"
            ),
            
            xaxis=dict(
                title='Время (МСК)',
                fixedrange=False
            )
        )
        
        # 🔧 НИЧЕГО НЕ МЕНЯЕМ: твой рабочий config
        config = {
            'scrollZoom': True,
            'doubleClick': 'reset',
            'modeBarButtonsToAdd': ['autoscale'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'displayModeBar': True
        }

        # 🔧 config передаётся СЮДА (как в твоём рабочем коде)
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        st.caption("💡 **Как менять масштаб:** Наведи на график и используй **колесико мыши** или выдели область.")
        
    with col2:
        st.subheader(" Метрики качества")
        
        # Блок с пояснением метрик
        with st.expander("ℹ️ Что означают эти цифры?", expanded=True):
            st.markdown("""
            ### 🎯 F1-Score
            **За что отвечает:** Баланс между точностью и полнотой предсказаний.  
            **К чему стремиться:** Чем ближе к **1.0**, тем лучше.  
            *В нашем случае:* Значение ~0.2–0.4 считается хорошим из-за сложности финансовых данных.
            
            ### 📈 ROC-AUC
            **За что отвечает:** Способность модели отличать рост от падения.  
            **К чему стремиться:** 
            - **0.5** = случайное угадывание (монетка).  
            - **> 0.7** = хорошая модель.  
            - **1.0** = идеальная модель.
            """)
        
        st.divider()
        
        st.metric("F1-Score", f"{f1:.4f}", delta=None)
        st.metric("ROC-AUC", f"{auc:.4f}", delta=None)
        
        st.divider()
        
        st.subheader("🔮 Предсказание на +5 дней")
        last_row_X = X_scaled[-1].reshape(1, -1)
        prediction = model.predict(last_row_X)[0]
        proba = model.predict_proba(last_row_X)[0][1]
        
        current_price = df_hist['close'].iloc[-1]
        
        st.write(f"Текущая цена: **${current_price:.2f}**")
        
        # 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: текст прогноза
        if prediction == 1:
            st.success(f"📈 ПРОГНОЗ: РОСТ (через 5 дней)")
            st.write(f"Вероятность: {proba:.1%}")
        else:
            st.warning(f"📉 ПРОГНОЗ: БЕЗ РОСТА")
            st.write(f"Вероятность роста: {proba:.1%}")

else:
    st.stop()

st.markdown("---")
# 🔧 ТОЛЬКО ЭТА СТРОКА ИЗМЕНЕНА: источник данных в футере
st.caption("НИРС ТМО | Александр ИУ5-61Б | Данные: Yahoo Finance (5y Daily) | Время: MSK (UTC+3)")