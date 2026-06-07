import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_DIR = "data/stage1_raw_logs/"
OUTPUT_PATH = "data/features_matrix.csv"

def profile_single_wallet(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        return None
    
    required_cols = ["Txn Hash", "Block", "Time", "From", "To", "Amount", "Asset", "seed_label"]
    for col in required_cols:
        if col not in df.columns:
            return None
            
    seed_label = df["seed_label"].iloc[0]
    filename = os.path.basename(file_path)
    
    df["From"] = df["From"].astype(str).str.strip()
    df["To"] = df["To"].astype(str).str.strip()
    df["Asset"] = df["Asset"].astype(str).str.strip()
    
    all_addresses = pd.concat([df["From"], df["To"]])
    wallet_address = all_addresses.value_counts().index[0]
    
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    df_usdt = df[df["Asset"] == "USDT"].copy()
    df_trx = df[df["Asset"] == "TRX"].copy()
    
    # 1. lifespan_days
    first_tx = df["Time"].min()
    last_tx = df["Time"].max()
    lifespan_days = max((last_tx - first_tx).days, 1)
    
    # 2. usdt_active_day_intensity
    if not df_usdt.empty:
        df_usdt["date_str"] = df_usdt["Time"].dt.strftime('%Y-%m-%d')
        usdt_active_dates = df_usdt["date_str"].nunique()
        usdt_total_txs = len(df_usdt)
        usdt_active_day_intensity = usdt_total_txs / usdt_active_dates if usdt_active_dates > 0 else 0.0
    else:
        df_usdt["date_str"] = ""
        usdt_active_day_intensity = 0.0
    
    # 3. dwell_time_avg_sec
    dwell_times = []
    last_in_time = None
    for _, row in df_usdt.iterrows():
        if row["To"] == wallet_address:
            last_in_time = row["Time"]
        elif row["From"] == wallet_address and last_in_time is not None:
            diff_sec = (row["Time"] - last_in_time).total_seconds()
            if diff_sec >= 0:
                dwell_times.append(diff_sec)
    dwell_time_avg_sec = np.mean(dwell_times) if dwell_times else 0.0
    
    # 4. in_out_amount_ratio
    usdt_in_sum = df_usdt[df_usdt["To"] == wallet_address]["Amount"].sum()
    usdt_out_sum = df_usdt[df_usdt["From"] == wallet_address]["Amount"].sum()
    in_out_amount_ratio = usdt_in_sum / (usdt_out_sum + 1e-6)
    
    # 5, 6, 13. peeling, aggregation, pass_through
    peeling_chain_count = 0
    aggregation_count = 0
    transit_pass_through_count = 0
    
    for idx, row in df_usdt.iterrows():
        if row["To"] == wallet_address:
            t_window_end = row["Time"] + pd.Timedelta(hours=2)
            subseq_out = df_usdt[(df_usdt["Time"] > row["Time"]) & (df_usdt["Time"] <= t_window_end) & (df_usdt["From"] == wallet_address)]
            
            if len(subseq_out) >= 3 and row["Amount"] > subseq_out["Amount"].max():
                peeling_chain_count += 1
                
            matching_out = subseq_out[np.abs(subseq_out["Amount"] - row["Amount"]) < 1e-2]
            if not matching_out.empty:
                transit_pass_through_count += 1
                
        if row["From"] == wallet_address:
            t_window_start = row["Time"] - pd.Timedelta(hours=2)
            prior_in = df_usdt[(df_usdt["Time"] >= t_window_start) & (df_usdt["Time"] < row["Time"]) & (df_usdt["To"] == wallet_address)]
            
            if len(prior_in) >= 3 and row["Amount"] > prior_in["Amount"].max():
                aggregation_count += 1

    # 7, 8, 9. balance metrics
    daily_net_retention_avg = 0.0
    balance_zero_days_ratio = 0.0
    balance_sawtooth_score = 0.0
    
    df_my_usdt = df_usdt[(df_usdt["To"] == wallet_address) | (df_usdt["From"] == wallet_address)].copy()
    
    if not df_my_usdt.empty:
        unique_dates = sorted(list(df_my_usdt["date_str"].unique()))
        daily_retentions = []
        current_balance = 0.0
        zero_balance_days = 0
        balance_history = []
        
        for d_str in unique_dates:
            day_records = df_my_usdt[df_my_usdt["date_str"] == d_str]
            day_in = day_records[day_records["To"] == wallet_address]["Amount"].sum()
            day_out = day_records[day_records["From"] == wallet_address]["Amount"].sum()
            
            retention = (day_in - day_out) / (day_in + day_out + 1e-6)
            daily_retentions.append(retention)
            
            current_balance += (day_in - day_out)
            if current_balance < 0.1:
                current_balance = 0.0
                zero_balance_days += 1
            balance_history.append(current_balance)
            
        daily_net_retention_avg = np.mean(daily_retentions) if daily_retentions else 0.0
        balance_zero_days_ratio = zero_balance_days / len(unique_dates) if unique_dates else 0.0
        if len(balance_history) > 1:
            balance_sawtooth_score = np.mean(np.abs(np.diff(balance_history)))

    # 10. otc_ping_pong_count
    otc_ping_pong_count = 0
    if len(df_usdt) > 1:
        for i in range(len(df_usdt) - 1):
            tx1 = df_usdt.iloc[i]
            tx2 = df_usdt.iloc[i+1]
            partner1 = tx1["From"] if tx1["To"] == wallet_address else tx1["To"]
            partner2 = tx2["From"] if tx2["To"] == wallet_address else tx2["To"]
            time_diff_min = (tx2["Time"] - tx1["Time"]).total_seconds() / 60.0
            
            if partner1 == partner2 and time_diff_min <= 10.0:
                if tx1["Amount"] > 0 and (tx2["Amount"] / tx1["Amount"]) >= 50.0:
                    otc_ping_pong_count += 1

    # 11, 12. trx metrics
    trx_in = df_trx[df_trx["To"] == wallet_address]
    trx_in_max_amount = trx_in["Amount"].max() if not trx_in.empty else 0.0
    trx_in_sum = trx_in["Amount"].sum()
    trx_sources_distinct = trx_in["From"].nunique()
    trx_fuel_density = (trx_in_sum / trx_sources_distinct) if trx_sources_distinct > 0 else 0.0

    filename_address = filename.replace("_1hop.csv", "").strip()
    return {
        "address": filename_address,
        "seed_label": seed_label,
        "lifespan_days": lifespan_days,
        "usdt_active_day_intensity": usdt_active_day_intensity,
        "dwell_time_avg_sec": dwell_time_avg_sec,
        "in_out_amount_ratio": in_out_amount_ratio,
        "peeling_chain_count": peeling_chain_count,
        "aggregation_count": aggregation_count,
        "transit_pass_through_count": transit_pass_through_count,
        "daily_net_retention_avg": daily_net_retention_avg,
        "balance_zero_days_ratio": balance_zero_days_ratio,
        "balance_sawtooth_score": balance_sawtooth_score,
        "otc_ping_pong_count": otc_ping_pong_count,
        "trx_in_max_amount": trx_in_max_amount,
        "trx_fuel_density": trx_fuel_density
    }

def main():
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*_1hop.csv"))
    if not csv_files:
        print("Error: No data found.")
        return
        
    features_list = []
    # 💡 使用 tqdm 封裝迭代器，自動生成優雅的終端機進度條
    for file_path in tqdm(csv_files, desc="Extracting features", unit="file"):
        try:
            res = profile_single_wallet(file_path)
            if res is not None:
                features_list.append(res)
        except Exception:
            continue
            
    if features_list:
        df_matrix = pd.DataFrame(features_list)
        cols = ["address", "seed_label"] + [c for c in df_matrix.columns if c not in ["address", "seed_label"]]
        df_matrix = df_matrix[cols]
        df_matrix.to_csv(OUTPUT_PATH, index=False)
        print(f"Success: {len(df_matrix)} wallets processed -> {OUTPUT_PATH}")
    else:
        print("Warning: No features extracted.")

if __name__ == "__main__":
    main()