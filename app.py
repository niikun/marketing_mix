import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import shap
import matplotlib.pyplot as plt

# SHAPの初期化
shap.initjs()

# タイトル
st.title("売上予測・媒体の重要度を可視化")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # データ前処理
    data["events"] = np.where(data["events"] == "na", 0, 1)

    target = "revenue"
    media_channels = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
    organic_channels = ["newsletter"]
    features = ["competitor_sales_B", "events"] + media_channels + organic_channels

    # モデルパラメータ設定
    params = {
        "n_estimators": 5,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_depth": 4,
        "ccp_alpha": 0.3,
        "bootstrap": True,
    }

    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=3)

    all_predictions = []

    for train_index, test_index in tscv.split(data):
        x_train = data.iloc[train_index][features]
        y_train = data[target].values[train_index]

        x_test = data.iloc[test_index][features]
        y_test = data[target].values[test_index]

        rf = RandomForestRegressor(random_state=0, **params)
        rf.fit(x_train, y_train)
        prediction = rf.predict(x_test)

        # 各分割セットの予測値を保存
        all_predictions.extend(prediction)

    # 予測結果を Pandas Series に変換してインデックスを揃える
    all_predictions = pd.Series(all_predictions, index=data.index[-len(all_predictions):])

    # 評価指標の計算
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    rmse_metric = root_mean_squared_error(data[target][-len(all_predictions):], all_predictions)
    mape_metric = mean_absolute_percentage_error(data[target][-len(all_predictions):], all_predictions)

    # NRMSE 関数の定義と計算
    def nrmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))

    nrmse_metric = nrmse(data[target][-len(all_predictions):], all_predictions)
    r2_metric = r2_score(data[target][-len(all_predictions):], all_predictions)

    # 結果の表示
    st.subheader("予測と評価指標")
    st.write(f"RMSE: {rmse_metric}")
    st.write(f"MAPE: {mape_metric}")
    st.write(f"NRMSE: {nrmse_metric}")
    st.write(f"R2 Score: {r2_metric}")

    # 予測結果のグラフ化
    fig = plt.figure(figsize=(25, 8))
    plt.plot(all_predictions, color="blue", label="Prediction")
    plt.plot(data[target][-len(all_predictions):], "ro", label="Actual")
    plt.legend(fontsize=18) 
    plt.title(
        "Prediction and Actual",fontsize=28
    )
    st.pyplot(fig)

    # SHAPによる特徴量重要度の計算と可視化
    explainer = shap.TreeExplainer(rf)
    shap_values_train = explainer.shap_values(data[features])

    df_shap_values = pd.DataFrame(shap_values_train, columns=features)
    feature_importance = np.abs(df_shap_values).mean()

    st.subheader("特徴量の重要度")
    fig = plt.figure(figsize=(10, 8))
    feature_importance.sort_values().plot.barh()
    plt.title("Feature Importance")
    st.pyplot(fig)

    responses = pd.DataFrame(df_shap_values[media_channels].abs().sum(axis = 0), columns = ["effect_share"])
    responses_percentages = responses / responses.sum()
    spends_percentages = pd.DataFrame(data[media_channels].sum(axis = 0) / data[media_channels].sum(axis = 0).sum(), columns = ["spend_share"])
    spend_effect_share = pd.merge(responses_percentages, spends_percentages, left_index = True, right_index = True)
    spend_effect_share = spend_effect_share.reset_index().rename(columns = {"index": "media"})
    df_spend_effect_share = spend_effect_share.copy()
    spend_effect_share.melt(id_vars=["media"])
    spend_effect_share.melt(id_vars=["media"],value_vars = ["spend_share", "effect_share"])
    df_share = spend_effect_share.melt(id_vars=["media"],value_vars = ["spend_share", "effect_share"])


    x = np.array(spend_effect_share["media"].unique())
    x_position = np.arange(len(x))


    y_control = np.array(spend_effect_share["spend_share"])
    y_stress = np.array(spend_effect_share["effect_share"])

    st.subheader("効果と出費のシェア")
    st.dataframe(df_spend_effect_share)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y_control,width=0.4,  label='spend share',color='red')
    ax.bar(x_position + 0.4,  y_stress, width=0.4, label='effect share',color='blue')
    ax.legend()
    ax.set_xticks(x_position + 0.2)
    ax.set_xticklabels(x)
    st.pyplot(fig)