# TRON Laundering Behavior

This project utilizes Machine Learning and Deep Learning to detect money laundering activities on the TRON blockchain.
We collect 1,100 wallet addresses: 660 normal user addresses (Label 0) from CEX counterparties without any risk label, and 440 known illicit addresses from 165 website (Label 1).

---

## Project Pipeline & Architecture

### 1. Data Collection

We fetches transaction histories (USDT & TRX logs) for all 1,100 targets via the Tronscan API, storing them in `data/stage1_raw_logs/`

### 2. Feature Engineering

* **What's we find while collecting illicit addresses ?**
    - Funds come in and quickly go out.
    - The wallet is active for only a short time.
    - The daily ending balance is often zero.
    - OTC cash trades are made with unknown parties, often with a small test transfer first.
    - TRX mainly comes from one large source.

* Convert logs into 13 core behavioral metrics per wallet.

| ID | Feature Name | Description |
|:-: | :----------: | :---------: |
| 01 | `lifespan_days` | Active days elapsed between the first and last transaction.
| 02 | `usdt_active_day_intensity` | USDT transaction volume divided by active days.
| 03 | `dwell_time_avg_sec` | Average time USDT sits in the wallet before outgoing transfer.
| 04 | `in_out_amount_ratio` | Ratio of total USDT deposited to total withdrawn.
| 05 | `peeling_chain_count` | Single macro input split into multiple sequential micro outputs (layering/peeling).
| 06 | `aggregation_count` | Multiple minor inputs consolidated into a single macro output within 2 hours.
| 07 | `transit_pass_through_count` | Instant identical forwarding (USDT matching input within 2 hours, error < 0.01).
| 08 | `daily_net_retention_avg` | Average daily capital retention rate. Measures if the account accumulates or drains funds.
| 09 | `balance_zero_days_ratio` | Percentage of active days where the wallet balance is wiped to zero.
| 10 | `balance_sawtooth_score` | Metric for volatile balance fluctuations. Captures rapid filling and clearing cycles.
| 11 | `otc_ping_pong_count` | High-frequency back-and-forth trades with the same counterpart within 10 mins (>50x size shift).
| 12 | `trx_in_max_amount` | Maximum single TRX deposit volume.
| 13 | `trx_fuel_density` | Total TRX received divided by unique source addresses.

### 3. Machine Learning Classification

Use a **Random Forest** classifier on the 13-feature matrix.

```
=== Baseline Model Performance ===
              precision    recall  f1-score   support

           0     0.9416    0.9773    0.9591       132
           1     0.9639    0.9091    0.9357        88

    accuracy                         0.9500       220
   macro avg     0.9527    0.9432    0.9474       220
weighted avg     0.9505    0.9500    0.9497       220

ROC AUC Score: 0.9928

=== Feature Importance Ranking ===
01. trx_fuel_density               : 0.2010
02. trx_in_max_amount              : 0.1968
03. in_out_amount_ratio            : 0.1490
04. otc_ping_pong_count            : 0.0876
05. daily_net_retention_avg        : 0.0612
06. balance_sawtooth_score         : 0.0575
07. lifespan_days                  : 0.0571
08. dwell_time_avg_sec             : 0.0518
09. usdt_active_day_intensity      : 0.0480
10. balance_zero_days_ratio        : 0.0354
11. transit_pass_through_count     : 0.0188
12. aggregation_count              : 0.0178
13. peeling_chain_count            : 0.0178
```

### 4. Sequential Deep Learning

Bypasses the 13 manual features. Trains recurrent networks (**GRU / LSTM**) directly on the raw chronological stream of the last **50** transactions, utilizing transaction amounts, directions, and time differentials (`time_delta`).

```
GRU Sequential Training
Epoch 01 | Train Loss: 0.6447 | Test AUC: 0.7766 | Test F1: 0.6355
Epoch 02 | Train Loss: 0.5478 | Test AUC: 0.9030 | Test F1: 0.7285
Epoch 03 | Train Loss: 0.3956 | Test AUC: 0.9132 | Test F1: 0.6619
Epoch 04 | Train Loss: 0.3466 | Test AUC: 0.9366 | Test F1: 0.8128
Epoch 05 | Train Loss: 0.2760 | Test AUC: 0.9520 | Test F1: 0.8125
Epoch 06 | Train Loss: 0.2474 | Test AUC: 0.9338 | Test F1: 0.8049
Epoch 07 | Train Loss: 0.2435 | Test AUC: 0.9486 | Test F1: 0.8323
Epoch 08 | Train Loss: 0.2273 | Test AUC: 0.9547 | Test F1: 0.8466
Epoch 09 | Train Loss: 0.2255 | Test AUC: 0.9473 | Test F1: 0.8439
Epoch 10 | Train Loss: 0.2250 | Test AUC: 0.9469 | Test F1: 0.8323
Epoch 11 | Train Loss: 0.2192 | Test AUC: 0.9410 | Test F1: 0.8049
Epoch 12 | Train Loss: 0.1918 | Test AUC: 0.9468 | Test F1: 0.8523
Epoch 13 | Train Loss: 0.1871 | Test AUC: 0.9424 | Test F1: 0.8362
Epoch 14 | Train Loss: 0.1851 | Test AUC: 0.9456 | Test F1: 0.8508
Epoch 15 | Train Loss: 0.1952 | Test AUC: 0.9384 | Test F1: 0.8352
```

```
LSTM Sequential Training
Epoch 01 | Train Loss: 0.6377 | Test AUC: 0.8170 | Test F1: 0.0444
Epoch 02 | Train Loss: 0.5353 | Test AUC: 0.8871 | Test F1: 0.6154
Epoch 03 | Train Loss: 0.4448 | Test AUC: 0.8772 | Test F1: 0.7089
Epoch 04 | Train Loss: 0.4114 | Test AUC: 0.8617 | Test F1: 0.7435
Epoch 05 | Train Loss: 0.4210 | Test AUC: 0.8831 | Test F1: 0.7716
Epoch 06 | Train Loss: 0.3795 | Test AUC: 0.8838 | Test F1: 0.7789
Epoch 07 | Train Loss: 0.3565 | Test AUC: 0.9032 | Test F1: 0.7576
Epoch 08 | Train Loss: 0.3754 | Test AUC: 0.8625 | Test F1: 0.7644
Epoch 09 | Train Loss: 0.3322 | Test AUC: 0.8925 | Test F1: 0.7701
Epoch 10 | Train Loss: 0.3157 | Test AUC: 0.8753 | Test F1: 0.7821
Epoch 11 | Train Loss: 0.3181 | Test AUC: 0.8954 | Test F1: 0.7835
Epoch 12 | Train Loss: 0.2886 | Test AUC: 0.8860 | Test F1: 0.7640
Epoch 13 | Train Loss: 0.3008 | Test AUC: 0.8858 | Test F1: 0.7708
Epoch 14 | Train Loss: 0.2785 | Test AUC: 0.8783 | Test F1: 0.7845
Epoch 15 | Train Loss: 0.2351 | Test AUC: 0.9083 | Test F1: 0.7654
```

---

## Model Benchmark 

| Model | Test AUC | F1-Score |
|:----: | :------: | :------: |
| Random Forest | 0.9928 | 0.9474 (Macro) |
| GRU | 0.9384 | 0.8352 |
| LSTM | 0.9083 | 0.7654 |