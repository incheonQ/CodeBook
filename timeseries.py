from collections.abc import Sequence
import numpy as np
import pandas as pd
import math

# 이동 평균법은 시계열 데이터의 단기 변동을 smoothing하여 장기적인 추세를 파악하는 데 사용됩니다.
# 장점: 간단하고 이해하기 쉬우며, 노이즈를 효과적으로 제거할 수 있습니다.
# 단점: 급격한 변화에 대한 반응이 느리고, 최근 데이터와 과거 데이터에 동일한 가중치를 부여합니다(SMA의 경우).
def moving_average_methods(
    data: Sequence[float], window_size: int
)->list[float | None]:
    
    if window_size < 1:
        raise ValueError
        
    if window_size > len(data):
        raise ValueError

    levels: list[float | None] = []
    
    for i in range(len(data)):
        if i < window_size - 1:
            levels.append(None)
        else:
            window = data[i-window_size+1 : i+1]
            level_value = sum(window)/window_size
            levels.append(level_value)
    return levels

# 지수 평활법은 최근 데이터에 더 높은 가중치를 부여하여 시계열을 평활화하는 방법입니다.
# 장점: 최근 데이터에 더 많은 중요성을 부여하며, 적은 데이터로도 예측이 가능합니다.
# 단점: 적절한 평활 상수를 선택하는 것이 중요하며, 장기 예측에는 부적합할 수 있습니다.
def exponential_smoothing_methods(
    data: Sequence[float], alpha: float = None
)-> list[float]:

    if alpha == None:
        alpha = 2 / ( 1 + len(data))
        print(f"Set alpha: {alpha}")

    levels = []
    
    first_level = sum(data)/len(data)
    levels.append(first_level)
    
    for i in range(len(data)):
        level_value = alpha * data[i] + ( 1 - alpha ) * levels[i]
        levels.append(level_value)

    return levels

# 단순선형회귀 회귀계수 b0, b1 
def betas(data):
    Xbar = sum(range(len(data)+1))/len(data)
    Ybar = sum(data)/len(data)

    X_minus_Xbar = [x - Xbar for x in range(1, len(data)+1)]
    Y_minus_Ybar = [y - Ybar for y in data]

    X_minus_Xbar_square = [x**2 for x in X_minus_Xbar]
    X_minus_Xbar_times_y_minus_Ybar = [x*y for x, y in zip(X_minus_Xbar, Y_minus_Ybar)]

    b1 = sum(X_minus_Xbar_times_y_minus_Ybar)/ sum(X_minus_Xbar_square)

    b0 = Ybar - b1 * Xbar
    return b0, b1

# 단순선형회귀 회귀계수 b0, b1 
def b1b0(
    x: np.ndarray,
    y: np.ndarray,
)->list[float, float]:

    x_bar = x.mean()
    y_bar = y.mean()

    upper_base = np.sum((x - x_bar)*(y - y_bar))

    lower_base = np.sum((x-x_bar)*(x-x_bar))

    b1 = upper_base / lower_base

    b0 = y_bar - b1*x_bar

    return [b1, b0]

# 홀트 모델은 단순 지수평활법을 확장한 것으로, 
# 추세(trend)를 가진 시계열 데이터를 예측하는 데 사용됩니다. 이 모델은 두 가지 구성요소를 고려합니다:
# 수준(level): 시계열의 현재 값
# 추세(trend): 시계열의 장기적인 증가 또는 감소 경향
def holt_model(
    data: Sequence[float], 
    alpha: float, 
    beta: float, 
)->list[float]:

    level0, trend0 = betas(data)
    forecast0 = level0 + trend0
    
    levels = []
    trends = []
    forecasts = []

    levels.append(level0)
    trends.append(trend0)
    forecasts.append(forecast0)

    for i in range(len(data)):
        level_value = alpha*data[i] + (1-alpha)*(levels[i]+trends[i])
        levels.append(level_value)
        trend_value = beta*(levels[i+1]-levels[i])+(1-beta)*trends[i]
        trends.append(trend_value)
        forecast_value = level_value + trend_value
        forecasts.append(forecast_value)

    return forecasts

# 윈터스 모델 (Winters Model):
# 윈터스 모델은 홀트 모델을 더욱 확장한 것으로, 
# 추세와 계절성(seasonality)을 모두 가진 시계열 데이터를 예측하는 데 사용됩니다. 이 모델은 세 가지 구성요소를 고려합니다:
# 수준(level)
# 추세(trend)
# 계절성(seasonality): 시계열의 주기적인 변동 패턴
def winters_model(
    data: Sequence[float],
    P: int,
    alpha: float, 
    beta: float, 
    gamma: float
)->list[float]:
    
    level0, trend0 = betas(data)

    # raise ValueError needs here
    r = len(data)//P

    # Preparation to calculate initial seasonal indexs
    seasonal_indexs = []
    for i in range(r):
        blocks = needs[i*P:i*P+P]
        for j in blocks:
            seasonal_index = j / (sum(blocks)/len(blocks))
            seasonal_indexs.append(seasonal_index)

    # Calculate initial seasonal indexs 1 ~ P
    for i in range(P):
        initial_seasonal_index = sum(seasonal_indexs[i::4])/3
        seasonal_indexs[i] = initial_seasonal_index
    
    # Calculate levels and Trends
    levels = []
    levels.append(level0)
    trends = []
    trends.append(trend0)

    for i in range(len(data)):
        level_value = alpha*(data[i]/seasonal_indexs[i])+(1-alpha)*(levels[i]+trends[i])
        levels.append(level_value)
        
        trend_value = beta*(levels[i+1]-levels[i])+(1-beta)*trends[i]
        trends.append(trend_value)

    # Calculate Seasonal Indexs
    seasonals = seasonal_indexs[:P]

    for i in range(len(data)-P):
        seasonal_value = gamma*(data[i]/levels[i+1])+(1-gamma)*seasonals[i]
        seasonals.append(seasonal_value)

    print(seasonals)

    # Predict
    forecasts = []

    for i in range(len(data)):
        forecast_value = (levels[i]+trends[i])*seasonals[i]
        forecasts.append(forecast_value)

    return forecasts

# ARIMA
# SARIMA
# ARIMAX
# ARCH
# GARCH
# State Space Models
# VAR (Vector AutoRegression)
# Prophet
# TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend and Seasonal components)
# Random Forest
# Support Vector Machine
# LSTM
# GRU
# TCN
# TFT (Temporal Fusion Transformers)
# Informer
# LogTrans
# N-BEATS
# DeepAR
# WaveNet
# LightGBM, XGBoost
# Random Forest
