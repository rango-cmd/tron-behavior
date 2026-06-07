import asyncio
import httpx
import pandas as pd
import os

TRONSCAN_API = "https://apilist.tronscanapi.com/api"
USDT_TOKEN = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
CEX_HOT = [
    "TQrY8tryqsYVCYS3MFbtffiPp2ccyn4STm", # Binance
    "TAzsQ9Gx8eqFNFSKbeXrbi45CuVPHzA8wr", # Binance
    "TYASr5UV6HEcXatwdFQfmLVUqQQQMUxHLS", # Binance
    "TDqSquXBgUCLYvYC4XZgrprLK589dkhSCf", # Binance
    "TM1zzNDZD2DPASbKcgdVoTYhfmYgtfwx9R", # OKX
    "THHhNUxqanKFAXmjjPGLBpPLK2KYXR9YVv", # BitoPro
]

OUTPUT_FILENAME = "data/raw_seeds/normal_addresses.csv"
TARGET_COUNT = 660


async def fetch_cex_transfers(client, cex_address):
    url = f"{TRONSCAN_API}/token_trc20/transfers"
    params = {"relatedAddress": cex_address, "limit": 50, "start": 0}
    addresses = set()
    
    try:
        res = await client.get(url, params=params, timeout=15.0)
        if res.status_code == 200:
            for tx in res.json().get("token_transfers", []):
                if tx.get("tokenInfo", {}).get("tokenId") == USDT_TOKEN:
                    frm, to = tx.get("from_address", ""), tx.get("to_address", "")
                    if frm and frm not in CEX_HOT: addresses.add(frm)
                    if to and to not in CEX_HOT: addresses.add(to)
    except Exception:
        pass
    return addresses

async def collect_normal_users(target_count=TARGET_COUNT):
    normal_addresses = set()
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    
    async with httpx.AsyncClient(limits=limits) as client:
        print(f"Starting collection from {len(CEX_HOT)} exchange hot wallets...")
        tasks = [fetch_cex_transfers(client, cex) for cex in CEX_HOT]
        results = await asyncio.gather(*tasks)
        
        for addr_set in results:
            normal_addresses.update(addr_set)
            if len(normal_addresses) >= target_count:
                break

    final_list = list(normal_addresses)[:target_count]
    print(f"Collected {len(final_list)} unique addresses.")
    return final_list

def save_to_csv(address_list, filename=OUTPUT_FILENAME):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame({"address": address_list, "label": 0})
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    captured_users = asyncio.run(collect_normal_users(target_count=660))
    if captured_users:
        save_to_csv(captured_users)