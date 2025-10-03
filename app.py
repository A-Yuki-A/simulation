import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
from matplotlib.patches import Patch
from pathlib import Path

# === フォント設定 ===
fp = Path("fonts/SourceHanCodeJP-Regular.otf")  # プロジェクトに fonts フォルダを作ってフォントを置く
if fp.exists():
    fm.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = "Source Han Code JP"
else:
    # 環境にある日本語フォントを優先して使う
    for name in ["Noto Sans JP", "IPAexGothic", "Yu Gothic", "Hiragino Sans", "Meiryo"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# ========== 画面設定 ==========
st.set_page_config(page_title="レジ待ち行列シミュレーション（1台/2台）", layout="wide")
st.title("レジ待ち行列シミュレーション（レジ1台 ↔ 2台 切替）")
st.caption("薄い枠＝待ち時間、太い棒＝サービス中。同じ乱数シードで実行すると結果は再現できます。")

# ========== サイドバー（入力） ==========
with st.sidebar:
    st.header("設定")
    n_kyaku = st.number_input("客数（人）", min_value=1, max_value=10, value=6, step=1, help="最大10人までです。")
    regis = st.selectbox("レジ台数", options=[1, 2], index=0, help="1台と2台で混雑の違いを比較できます。")
    heikin_tochaku = st.number_input("平均到着間隔（分）", min_value=0.1, value=1.0, step=0.1, format="%.1f")
    heikin_taio = st.number_input("平均対応時間（分）", min_value=0.1, value=1.5, step=0.1, format="%.1f")
    time_res = st.selectbox("表示の時間刻み（分）", [0.25, 0.5, 1.0], index=1)
    seed = st.number_input(
        "乱数シード",
        min_value=0, max_value=10_000, value=42, step=1,
        help="乱数の“種”です。同じ値で実行すると毎回まったく同じ結果（再現性）になります。値を変えると別のケースになります。"
    )
    example = st.toggle(
        "例題モード（到着：0, 0.5, 1.0, 2.5, 3.0, 3.5）",
        value=True,
        help="授業プリントの6人シナリオ。OFFで到着・対応ともに指数分布の乱数になります。"
    )
    st.markdown("---")
    run = st.button("シミュレーション実行", use_container_width=True)

# ========== シミュレーション関数 ==========
def simulate_queue(N, mean_arrival, mean_service, servers=1, seed=0, example=False):
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

    # 各レジが「次に空く時刻」
    server_free = np.zeros(servers)

    start = np.zeros(N)
    end = np.zeros(N)
    wait = np.zeros(N)
    queue_len_at_arrival = np.zeros(N, dtype=int)

    for i in range(N):
        in_system = np.sum(end[:i] > arr[i])
        queue_len_at_arrival[i] = max(int(in_system) - servers, 0)

        s_idx = np.argmin(server_free)
        start[i] = max(arr[i], server_free[s_idx])
        wait[i] = start[i] - arr[i]
        end[i] = start[i] + service[i]
        server_free[s_idx] = end[i]

    df = pd.DataFrame({
        "客番号": np.arange(1, N+1),
        "並び始め（分）": np.round(arr, 3),
        "対応開始（分）": np.round(start, 3),
        "対応終了（分）": np.round(end, 3),
        "待ち時間（分）": np.round(wait, 3),
        "対応時間（分）": np.round(service, 3),
        "到着時点の待ち人数（人）": queue_len_at_arrival
    })

    最大待ち時間 = float(wait.max()) if N > 0 else 0.0
    平均待ち時間 = float(wait.mean()) if N > 0 else 0.0
    最大待ち人数 = int(queue_len_at_arrival.max()) if N > 0 else 0

    return df, 最大待ち時間, 平均待ち時間, 最大待ち人数

# ========== 実行 ==========
if run:
    N = int(min(max(n_kyaku, 1), 10))
    df, 最大待ち時間, 平均待ち時間, 最大待ち人数 = simulate_queue(
        N,
        float(heikin_tochaku),
        float(heikin_taio),
        servers=int(regis),
        seed=int(seed),
        example=example
    )

    lam = 1.0 / float(heikin_tochaku)
    mu = 1.0 / float(heikin_taio)
    rho = lam / (int(regis) * mu)

    st.subheader("結果まとめ")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("レジ台数", f"{int(regis)} 台")
    with c2:
        st.metric("最大待ち時間", f"{最大待ち時間:.2f} 分")
    with c3:
        st.metric("平均待ち時間", f"{平均待ち時間:.2f} 分")
    with c4:
        st.metric("最大待ち人数", f"{最大待ち人数} 人")

    if rho >= 1.0:
        st.warning(f"参考：利用率 ρ = λ/(cμ) ≈ {rho:.2f}（1以上）→ 長期的に列が伸びやすい条件です。")
    else:
        st.info(f"参考：利用率 ρ = λ/(cμ) ≈ {rho:.2f}（1未満）→ 長期的に安定しやすい条件です。")

    st.subheader("到着・開始・終了の一覧（日本語表記）")
    st.dataframe(df, use_container_width=True)

    # ========== 0/1 グリッド ==========
    st.subheader("0/1 グリッド（1＝待ち，0＝サービス中，空白＝未到着/終了後）")
    t_min = 0.0
    t_max = max(df["対応終了（分）"].max(), df["並び始め（分）"].min() + 6)
    t_max = np.ceil(t_max / time_res) * time_res
    times = np.arange(t_min, t_max + 1e-9, time_res)

    grid = []
    for _, row in df.iterrows():
        arr, start, end = row["並び始め（分）"], row["対応開始（分）"], row["対応終了（分）"]
        vals = []
        for t in times:
            if arr <= t < start:
                vals.append(1)
            elif start <= t < end:
                vals.append(0)
            else:
                vals.append("")
        grid.append(vals)

    time_cols = [f"時刻（分）：{t:.2f}" for t in times]
    grid_df = pd.DataFrame(grid, columns=time_cols)
    grid_df.insert(0, "客番号", df["客番号"].astype(int))
    st.dataframe(grid_df, use_container_width=True, height=min(600, 100 + 28*len(grid)))

    # ========== ガントチャート ==========
    st.subheader("ガントチャート（太い棒＝サービス中／薄い枠＝待ち時間）")
    fig, ax = plt.subplots(figsize=(10, 0.35*len(df)+2))
    y = np.arange(len(df))

    ax.barh(y, df["対応終了（分）"] - df["対応開始（分）"],
            left=df["対応開始（分）"], height=0.6, align='center')
    ax.barh(y, df["対応開始（分）"] - df["並び始め（分）"],
            left=df["並び始め（分）"], height=0.6, align='center', fill=False)

    ax.set_xlabel("時間（分）")
    ax.set_ylabel("客番号（上から1,2,3,…）")
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
        file_name=f"待ち行列結果_{regis}台.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info(
        "左の設定を入力して「シミュレーション実行」を押してください。\n"
        "・レジ台数（1台/2台）を切り替えると、混雑の違いを比較できます。\n"
        "・乱数シード：同じ値で同じ結果（再現性）。値を変えると別ケースになります。\n"
        "・ガントチャート：太い棒＝サービス中、薄い枠＝待ち時間。\n"
        "・客数は最大10人です。"
    )
