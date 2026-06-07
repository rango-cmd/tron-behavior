# 錢包行為量化特徵提取規格說明書 (1-Hop USDT + TRX)

本模組 (`src/features/wallet_profiler.py`) 負責將採集到的 1,100 個原始交易流水 CSV，壓縮轉化為包含 12 個高鑑識價值指標的靜態特徵矩陣 (`data/features_matrix.csv`)，用以引導後續 Temporal GNN 模型的高效訓練。

## 📊 特徵資產對照與定義矩陣

| 特徵代號 | 特徵名稱 | 涉及資產 | 鑑識邏輯與定義說明 |
| :--- | :--- | :---: | :--- |
| **特徵 01** | `lifespan_days` | 二者都有 | 錢包最後一筆交易時間與第一筆交易時間的天數差（評估是否為短命涉詐錢包）。 |
| **特徵 02** | `usdt_active_day_intensity` | **USDT** | `USDT總交易筆數` / `有發生USDT轉帳的去重天數`（剔除靜默期，捕捉水房高頻操作爆發力）。 |
| **特徵 03** | `dwell_time_avg_sec` | **USDT** | USDT 從流入（in）到下一次流出（out）的平均間隔秒數（資金留存沉澱時間）。 |
| **特徵 04** | `in_out_amount_ratio` | **USDT** | 歷史總流入 USDT / 總流出 USDT 金額比值（越接近 1 代表過水、不留財特徵越明顯）。 |
| **特徵 05** | `peeling_chain_count` | **USDT** | 單筆大額進、且隨後 **2 小時內**拆成 $\ge 3$ 筆小額出的總次數（剝洋蔥分流）。 |
| **特徵 06** | `aggregation_count` | **USDT** | $\ge 3$ 筆小額進、且隨後 **2 小時內**融合成單筆大額出的總次數（資金歸集水房行為）。 |
| **特徵 07** | `daily_net_retention_avg` | **USDT** | 每日 `(進U - 出U) / (進U + 出U)` 的平均值（量化日內資金吞吐的對稱平衡度）。 |
| **特徵 08** | `balance_zero_days_ratio` | **USDT** | 每日 24:00 結算時，USDT 模擬動態餘額為 0 的天數佔總存續天數的比例（防凍結清零）。 |
| **特徵 09** | `balance_sawtooth_score` | **USDT** | 每日餘額變動曲線的斜率絕對值平均數（洗錢機器人會呈現高頻、尖銳的鋸齒波形）。 |
| **特徵 10** | `otc_ping_pong_count` | **USDT** | 同一對手在 10 分鐘內連續轉帳，且第二筆金額大於第一筆 $\ge 50$ 倍的次數（場外測活交易）。 |
| **特徵 11** | `trx_in_max_amount` | **TRX** | 錢包收到的單筆最大 TRX 金額（定位黑產的手續費生命線注資強度）。 |
| **特徵 12** | `trx_fuel_density` | **TRX** | `歷史TRX總流入金額` / `TRX發送源去重計數`（數值越高代表手續費越依賴集團中央燃料庫）。 |

## ⚙️ 落地資料集規格
* **輸出檔案**：`data/features_matrix.csv`
* **維度尺寸**：$1100 \times 14$ （1100 個錢包節點，包含 `address` 與 `seed_label` 標籤，外加 12 個特徵欄位）。