import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Patch

# ========== 画面設定 ==========
st.set_page_config(page_title="レジ待ち行列シミュレーション（1台/2台）", layout="wide")
st.title("レジ待ち行列シミュレーション（レジ1台 ↔ 2台 切替）")
st.caption(
    "薄い枠＝待ち時間、太い棒＝サービス中。乱数シードを同じにすると結果は再現できます。"
)

# ========== サイドバー（入力） ==========
with st.sidebar:
    st.header("パラメータ")
    n_kyaku = st.number_input("客数（人）※1〜10", min_value=1, max_value=10, value=6, step=1)
    regis = st.selectbox("レジの台数", options=[1, 2], index=0, help="レジを1台か2台で切り替えて比較できます。")
    heikin_tochaku = st.number_input("平均到着間隔（分）", min_value=0.1, value=1.0, step=0.1, format="%.1f")
    heikin_taio = st.number_input("平均対応時間（分）", min_value=0.1, value=1.5, step=0.1, format="%.1f")
    time_res = st.selectbox("表示の時間刻み（分）", [0.25, 0.5, 1.0], index=1)
    seed = st.number_input(
        "乱数シード（同じ値なら同じ結果）",
        min_value=0, max_value=10_000, value=42, step=1,
        help="乱数の“種”。同じシードで実行すると毎回まったく同じ結果（再現性）。数値を変えると別のケースになります。"
    )
    example = st.toggle(
        "例題モード（到着：0,0.5,1.0,2.5,3.0,3.5）",
        value=True,
        help="授業プリントの6人シナリオ。OFFで到着・対応ともに乱数（指数分布）。"
    )
    st.markdown("---")
    run = st.button("シミュレーション実行", use_container_width=True)

# ========== シミュレーション（M/M/c風：c=1 or 2） ==========
def simulate_queue(N, mean_arrival, mean_service, servers=1, seed=0, example=False):
    """
    先着順（FCFS）、レジが空いたらすぐ次を開始。
    到着間隔 ~ Exp(mean_arrival), 対応時間 ~ Exp(mean_service)
    servers = レジ台数（1 or 2）
    """
    rng = np.random.default_rng(seed)

    # 到着時刻
    if example:
        base_arr = np.array([0.0, 0.5, 1.0, 2.5, 3.0, 3.5])
        if N <= len(base_arr):
            arr = base_arr[:N]
        else:
            extra = rng.exponential(scale=mean_arrival, size=N - len(base_arr))
            arr = np.concatenate([base_arr, base_arr[-1] + np.cumsum(extra)])
    else:
        inter = rng.exponential(scale=mean_arrival, size=N)
        arr = np.cumsum(inter)

    # 対応時間
    service = rng.exponential(scale=mean_service, size=N)

    # レジの「次に空く時刻」を管理（長さ=servers）
    server_free = np.zeros(servers)

    start = np.zeros(N)
    end = np.zeros(N)
    wait = np.zeros(N)
    queue_len_at_arrival = np.zeros(N, dtype=int)

    for i in range(N):
        # 到着時点でまだ終わっていない人数
        in_system = np.sum(end[:i] > arr[i])
        # 並んでいる人数 = max(（システム内人数） - レジ台数, 0)
        queue_len_at_arrival[i] = max(int(in_system) - servers, 0)

        # 直近で一番早く空くレジを探す
        s_idx = np.argmin(server_free)
        # そのレジが空く時刻と到着時刻の遅い方から開始
        start[i] = max(arr[i], server_free[s_idx])
        wait[i] = start[i] - arr[i]
        end[i] = start[i] + service[i]
        # そのレジの次に空く時刻を更新
        server_free[s_idx] = end[i]

    df = pd.DataFrame({
        "客番号": np.arange(1, N+1),
        "並び始め(分)": np.round(arr, 3),
        "対応開始(分)": np.round(start, 3),
        "対応終了(分)": np.round(end, 3),
        "待ち時間(分)": np.round(wait, 3),
        "対応時間(分)": np.round(service, 3),
        "到着時点の待ち人数": queue_len_at_arrival
    })

    max_wait_time = float(wait.max()) if N > 0 else 0.0
    avg_wait_time = float(wait.mean()) if N > 0 else 0.0
    max_queue = int(queue_len_at_arrival.max()) if N > 0 else 0

    return df, max_wait_time, avg_wait_time, max_queue

# ========== 実行 ==========
if run:
    N = int(min(max(n_kyaku, 1), 10))
    df, max_wait, avg_wait, max_queue = simulate_queue(
        N,
        float(heikin_tochaku),
        float(heikin_taio),
        servers=int(regis),
        seed=int(seed),
        example=example
    )

    # 利用率の目安（ρ = λ / (c μ)）
    lam = 1.0 / float(heikin_tochaku)   # 到着率
    mu = 1.0 / float(heikin_taio)       # サービス率（1台あたり）
    rho = lam / (int(regis) * mu)

    st.subheader("結果サマリ")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("レジ台数", f"{int(regis)} 台")
    with c2:
        st.metric("最大待ち時間", f"{max_wait:.2f} 分")
    with c3:
        st.metric("平均待ち時間", f"{avg_wait:.2f} 分")
    with c4:
        st.metric("最大待ち人数", f"{max_queue} 人")

    if rho >= 1.0:
        st.warning(
            f"参考：理論上の利用率 ρ = λ/(cμ) ≈ {rho:.2f} ≥ 1 なので、長期的には混みやすい条件です（列が伸びやすい）。"
        )
    else:
        st.info(
            f"参考：理論上の利用率 ρ = λ/(cμ) ≈ {rho:.2f} < 1 なので、長期的には安定しやすい条件です。"
        )

    st.subheader("タイムテーブル（到着・開始・終了）")
    st.dataframe(df, use_container_width=True)

    # ========== 0/1 グリッド（待ち=1, サービス=0） ==========
    st.subheader("0/1 グリッド（1=待ち, 0=サービス中, 空白=未到着/終了後）")
    t_min = 0.0
    t_max = max(df["対応終了(分)"].max(), df["並び始め(分)"].min() + 6)
    t_max = np.ceil(t_max / time_res) * time_res
    times = np.arange(t_min, t_max + 1e-9, time_res)

    grid = []
    for _, row in df.iterrows():
        arr, start, end = row["並び始め(分)"], row["対応開始(分)"], row["対応終了(分)"]
        vals = []
        for t in times:
            if arr <= t < start:
                vals.append(1)   # 待ち
            elif start <= t < end:
                vals.append(0)   # サービス中
            else:
                vals.append("")  # 範囲外
        grid.append(vals)

    grid_df = pd.DataFrame(grid, columns=[f"{t:.2f}" for t in times])
    grid_df.insert(0, "客番号", df["客番号"].astype(int))
    st.dataframe(grid_df, use_container_width=True, height=min(600, 100 + 28*len(grid)))

    # ========== ガントチャート ==========
    st.subheader("ガントチャート（太い棒＝サービス中／薄い枠＝待ち時間）")
    fig, ax = plt.subplots(figsize=(10, 0.35*len(df)+2))
    y = np.arange(len(df))

    # サービス中（塗りつぶしの太い棒）
    ax.barh(y, df["対応終了(分)"] - df["対応開始(分)"],
            left=df["対応開始(分)"], height=0.6, align='center')
    # 待ち時間（枠線のみ＝薄い線に見える）
    ax.barh(y, df["対応開始(分)"] - df["並び始め(分)"],
            left=df["並び始め(分)"], height=0.6, align='center', fill=False)

    ax.set_xlabel("時間（分）")
    ax.set_ylabel("客（上から1,2,3,…）")
    ax.set_yticks(y, [int(i) for i in df["客番号"]])
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    legend_items = [
        Patch(facecolor="gray", label="サービス中（太い棒）"),
        Patch(facecolor="none", edgecolor="black", label="待ち時間（薄い枠）")
    ]
    ax.legend(handles=legend_items, loc="upper right")
    st.pyplot(fig, clear_figure=True)

    # ========== ダウンロード ==========
    st.download_button(
        "結果CSVをダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"queue_sim_{regis}registers.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info(
        "左のパラメータを設定して「シミュレーション実行」。\n"
        "・レジ台数を 1 ↔ 2 で切り替えて混雑の違いを比較できます。\n"
        "・乱数シード：同じ値で同じ結果（再現性）。値を変えると別ケース。\n"
        "・ガントチャート：太い棒＝サービス中、薄い枠＝待ち時間。\n"
        "・客数は最大10人です。"
    )
