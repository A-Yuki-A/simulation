import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ========== ヘッダ ==========
st.set_page_config(page_title="レジ待ち行列シミュレーション", layout="wide")
st.title("レジ待ち行列シミュレーション（M/M/1 風）")
st.caption("到着間隔と対応時間をランダムにして、待ち行列の混雑（最大待ち人数・最大待ち時間）を体験できます。")

# ========== 入力 ==========
with st.sidebar:
    st.header("パラメータ")
    n_kyaku = st.number_input("客数（人）", min_value=1, max_value=200, value=30, step=1)
    heikin_tochaku = st.number_input("平均到着間隔（分）", min_value=0.1, value=1.0, step=0.1, format="%.1f")
    heikin_taio = st.number_input("平均対応時間（分）", min_value=0.1, value=1.5, step=0.1, format="%.1f")
    time_res = st.selectbox("表示の時間刻み（分）", [0.25, 0.5, 1.0], index=1)
    seed = st.number_input("乱数シード（同じ数で再現）", min_value=0, max_value=10_000, value=42, step=1)
    example = st.toggle("例題モード（固定の到着時刻を使用・6人）", value=False)
    st.markdown("---")
    run = st.button("シミュレーション実行", use_container_width=True)

# ========== シミュレーション関数 ==========
def simulate_queue(N, mean_arrival, mean_service, seed=0, example=False):
    rng = np.random.default_rng(seed)

    if example:
        # 授業用の固定例（画像の課題と同型）
        # Nは6に固定・到着：0, 0.5, 1.0, 2.5, 3.0, 3.5
        arr = np.array([0.0, 0.5, 1.0, 2.5, 3.0, 3.5])
        N = len(arr)
        # 対応時間は平均1.5分の指数乱数（固定シードで再現性）
        service = rng.exponential(scale=1.5, size=N)
    else:
        # 到着間隔(指数分布)を累積和
        inter = rng.exponential(scale=mean_arrival, size=N)
        arr = np.cumsum(inter)
        service = rng.exponential(scale=mean_service, size=N)

    start = np.zeros(N)
    end = np.zeros(N)
    wait = np.zeros(N)
    queue_len_at_arrival = np.zeros(N, dtype=int)

    last_end = 0.0
    for i in range(N):
        # その到着時刻にまだ終わっていない人数 = 待ち人数 +（サービス中の1人）
        # 待ち人数（＝行列に並ぶ人数）は「システム内人数 - 1（サービス中）」を0未満なら0に丸め
        in_system = np.sum(end[:i] > arr[i])  # まだ終わっていない先客
        queue_len_at_arrival[i] = max(in_system - 1, 0)

        start[i] = max(arr[i], last_end)
        wait[i] = start[i] - arr[i]
        end[i] = start[i] + service[i]
        last_end = end[i]

    df = pd.DataFrame({
        "客番号": np.arange(1, N+1),
        "並び始め(分)": np.round(arr, 3),
        "対応開始(分)": np.round(start, 3),
        "対応終了(分)": np.round(end, 3),
        "待ち時間(分)": np.round(wait, 3),
        "対応時間(分)": np.round(service, 3),
        "到着時点の待ち人数": queue_len_at_arrival
    })

    # 指標
    max_wait_time = wait.max()
    max_queue = queue_len_at_arrival.max()

    return df, float(max_wait_time), int(max_queue)

# ========== 実行 ==========
if run:
    df, max_wait, max_queue = simulate_queue(
        int(n_kyaku),
        float(heikin_tochaku),
        float(heikin_taio),
        seed=int(seed),
        example=example
    )

    st.subheader("結果サマリ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("最大待ち時間", f"{max_wait:.2f} 分")
    with col2:
        st.metric("最大待ち人数", f"{max_queue} 人")
    with col3:
        avg_wait = df["待ち時間(分)"].mean()
        st.metric("平均待ち時間", f"{avg_wait:.2f} 分")

    st.subheader("タイムテーブル（到着・開始・終了）")
    st.dataframe(df, use_container_width=True)

    # ========== 0/1 グリッド（待ち=1, サービス=0） ==========
    st.subheader("0/1 グリッド（1=待ち, 0=サービス中, 空白=未到着/終了後）")
    t_min = 0.0
    t_max = max(df["対応終了(分)"].max(), df["並び始め(分)"].min() + 10)
    # 表示時間をきれいに
    t_max = np.ceil(t_max / time_res) * time_res
    times = np.arange(t_min, t_max + 1e-9, time_res)

    grid = []
    for _, row in df.iterrows():
        arr, start, end = row["並び始め(分)"], row["対応開始(分)"], row["対応終了(分)"]
        vals = []
        for t in times:
            # 区間 [t, t+Δt) の代表点として t を用いる簡易判定
            if arr <= t < start:
                vals.append(1)   # 待ち
            elif start <= t < end:
                vals.append(0)   # サービス中
            else:
                vals.append("")  # その時間帯は関係なし
        grid.append(vals)

    grid_df = pd.DataFrame(grid, columns=[f"{t:.2f}" for t in times])
    grid_df.insert(0, "客番号", df["客番号"].astype(int))
    st.dataframe(grid_df, use_container_width=True, height=min(600, 100 + 28*len(grid)))

    # ========== ガントチャート ==========
    st.subheader("ガントチャート（横棒：対応時間、薄い線：待ち）")
    fig, ax = plt.subplots(figsize=(10, 0.35*len(df)+2))

    y = np.arange(len(df))
    # 対応（塗りつぶし）
    ax.barh(y, df["対応終了(分)"] - df["対応開始(分)"],
            left=df["対応開始(分)"], height=0.6, align='center')
    # 待ち（枠線だけで前段に）
    ax.barh(y, df["対応開始(分)"] - df["並び始め(分)"],
            left=df["並び始め(分)"], height=0.6, align='center', fill=False)

    ax.set_xlabel("時間（分）")
    ax.set_ylabel("客（上から1,2,3,…）")
    ax.set_yticks(y, [int(i) for i in df["客番号"]])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    st.pyplot(fig, clear_figure=True)

    # ========== ダウンロード ==========
    st.download_button(
        "結果CSVをダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="queue_simulation_result.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("左のパラメータを設定して「シミュレーション実行」を押してください。『例題モード』をONにすると、授業プリントの6人の固定到着時刻で試せます。")
