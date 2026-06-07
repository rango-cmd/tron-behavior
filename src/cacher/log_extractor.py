import asyncio
import httpx
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TRONSCAN_API = "https://apilist.tronscanapi.com/api"
TRONSCAN_API_KEY = os.getenv("TRONSCAN_API_KEY")
USDT_TOKEN = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

async def fetch_with_retry(client, url, params=None, headers=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(0.1 + attempt * 0.1)
            res = await client.get(url, params=params, headers=headers, timeout=15.0)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
    return {}

async def fetch_spec_usdt(client, wallet_b58, headers):
    url = f"{TRONSCAN_API}/token_trc20/transfers"
    params = {"relatedAddress": wallet_b58, "limit": 50, "start": 0}
    records = []
    
    while True:
        res_data = await fetch_with_retry(client, url, params, headers)
        data = res_data.get("token_transfers", [])
        if not data:
            break
        
        for tx in data:
            if tx.get("tokenInfo", {}).get("tokenId") != USDT_TOKEN:
                continue
            
            amt = float(tx.get("quant", 0)) / 1_000_000.0
            if amt < 1.0:
                continue
            
            ts_ms = int(tx.get("block_ts", 0))
            tx_time = datetime.fromtimestamp(ts_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S') if ts_ms else "Unknown"
            
            records.append({
                "Txn Hash": tx.get("transaction_id", ""),
                "Block": tx.get("block", 0),
                "Time": tx_time,
                "From": tx.get("from_address", ""),
                "To": tx.get("to_address", ""),
                "Amount": amt,
                "Asset": "USDT"
            })
            
        if len(data) < 50:
            break
        params["start"] += 50
    return records

async def fetch_spec_trx(client, wallet_b58, headers):
    url = f"{TRONSCAN_API}/transfer"
    params = {"address": wallet_b58, "limit": 50, "start": 0}
    records = []
    
    while True:
        res_data = await fetch_with_retry(client, url, params, headers)
        data = res_data.get("data", [])
        if not data:
            break
        
        for tx in data:
            amt = float(tx.get("amount", 0)) / 1_000_000.0
            if amt < 1.0:
                continue
            
            ts_ms = int(tx.get("timestamp", 0))
            tx_time = datetime.fromtimestamp(ts_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S') if ts_ms else "Unknown"
            
            records.append({
                "Txn Hash": tx.get("transactionHash", ""),
                "Block": tx.get("block", 0),
                "Time": tx_time,
                "From": tx.get("transferFromAddress", ""),
                "To": tx.get("transferToAddress", ""),
                "Amount": amt,
                "Asset": "TRX"
            })
            
        if len(data) < 50:
            break
        params["start"] += 50
    return records

async def extract_all_seeds():
    output_dir = "data/stage1_raw_logs/"
    os.makedirs(output_dir, exist_ok=True)
    
    df_seeds = pd.concat([
        pd.read_csv("data/raw_seeds/laundering_addresses.csv"),
        pd.read_csv("data/raw_seeds/normal_addresses.csv")
    ], ignore_index=True)
    
    headers = {"TRONSCAN-API-KEY": TRONSCAN_API_KEY} if TRONSCAN_API_KEY else {}
    limits = httpx.Limits(max_keepalive_connections=3, max_connections=5)
    
    async with httpx.AsyncClient(http2=False, limits=limits) as client:
        print(f"Starting data collection for {len(df_seeds)} seeds. API Key loaded: {bool(TRONSCAN_API_KEY)}")
        
        for idx, row in df_seeds.iterrows():
            seed_wallet = str(row["address"]).strip()
            print(f"[{idx+1}/{len(df_seeds)}] Processing: {seed_wallet}")
            
            try:
                usdt_txs = await fetch_spec_usdt(client, seed_wallet, headers)
                trx_txs = await fetch_spec_trx(client, seed_wallet, headers)
                all_seed_txs = usdt_txs + trx_txs
                
                if all_seed_txs:
                    df_out = pd.DataFrame(all_seed_txs)
                    df_out["seed_label"] = row["label"]
                    df_out = df_out.sort_values(by="Time").reset_index(drop=True)
                    df_out.to_csv(os.path.join(output_dir, f"{seed_wallet}_1hop.csv"), index=False)
                    print(f"  Success: {len(df_out)} rows (USDT: {len(usdt_txs)} | TRX: {len(trx_txs)})")
                else:
                    print("  No matching transactions.")
            except Exception as e:
                print(f"  Error on {seed_wallet}: {e}")

if __name__ == "__main__":
    asyncio.run(extract_all_seeds())