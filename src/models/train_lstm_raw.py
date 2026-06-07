import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

LOGS_DIR = "data/stage1_raw_logs/"
# 💡 修正：輸出檔名明確指定為 lstm_raw_model.pt
MODEL_SAVE_PATH = "data/lstm_raw_model.pt"

SEQ_LEN = 50      
FEATURE_DIM = 5    

def build_raw_lstm_dataset():
    csv_files = glob.glob(os.path.join(LOGS_DIR, "*_1hop.csv"))
    if not csv_files:
        print("Error: No raw CSV logs found.")
        return None, None

    X_list = []
    y_list = []

    for file_path in tqdm(csv_files, desc="Parsing raw transaction streams for LSTM", unit="file"):
        df = pd.read_csv(file_path)
        if df.empty: continue
        if "seed_label" not in df.columns: continue
        label = int(df["seed_label"].iloc[0])
        
        all_addresses = pd.concat([df["From"], df["To"]])
        wallet_address = all_addresses.value_counts().index[0]
        
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values(by="Time").reset_index(drop=True)
        df["time_delta"] = df["Time"].diff().dt.total_seconds().fillna(0.0)
        
        seq_features = []
        for _, row in df.iterrows():
            amount = float(row["Amount"])
            is_usdt = 1.0 if row["Asset"] == "USDT" else 0.0
            is_trx = 1.0 if row["Asset"] == "TRX" else 0.0
            direction = 1.0 if row["To"] == wallet_address else -1.0
            time_delta = float(row["time_delta"])
            seq_features.append([amount, is_usdt, is_trx, direction, time_delta])
            
        seq_features = seq_features[-SEQ_LEN:]
        if len(seq_features) < SEQ_LEN:
            pad_len = SEQ_LEN - len(seq_features)
            padded = [[0.0] * FEATURE_DIM] * pad_len + seq_features
            seq_features = padded
            
        X_list.append(seq_features)
        y_list.append(label)

    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)

class TronRawLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        rnn_out, _ = self.lstm(x)
        last_step_features = rnn_out[:, -1, :] 
        logits = self.fc(last_step_features)
        return logits.squeeze(-1)

def main():
    print("🎬 啟動純 Raw Data 驅動型 LSTM 時序運算引擎...")
    X, y = build_raw_lstm_dataset()
    if X is None: return
    print(f"Dataset Built -> Sequences: {X.shape[0]}, Steps: {X.shape[1]}, Raw Variables: {X.shape[2]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    hidden_dim = 64
    batch_size = 32
    epochs = 15
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Loading LSTM Model onto -> {device}")

    model = TronRawLSTMClassifier(FEATURE_DIM, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting LSTM Sequential Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches

        model.eval()
        with torch.no_grad():
            test_x, test_y = X_test.to(device), y_test.to(device)
            test_logits = model(test_x)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            
            test_auc = roc_auc_score(y_test.numpy(), test_probs)
            test_f1 = f1_score(y_test.numpy(), [1 if p > 0.5 else 0 for p in test_probs])
            print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Success: LSTM model saved to -> {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()