from recbole.quick_start import load_data_and_model
import torch
import pandas as pd

# Load model dan dataset
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='ckpt/CKE.pth'
)

# Dapatkan mapping ID
uid2idx = dataset.field2id_token['user_id']
iid2idx = dataset.field2id_token['item_id']
idx2iid = {v: k for k, v in iid2idx.items()}

# Cari item long tail dari interaction (misalnya dari train_data)
interaction_df = dataset.inter_feat.copy()
item_freq = interaction_df['item_id'].value_counts()
long_tail_items = item_freq[item_freq == 1].index.tolist()
long_tail_item_ids = [iid2idx[i] for i in long_tail_items if i in iid2idx]

# Pilih user untuk diuji (misalnya user pertama)
user_id = 0
user_tensor = torch.tensor([user_id]).to(config['device'])

# Prediksi semua item untuk user tersebut
model.eval()
with torch.no_grad():
    scores = model.full_sort_predict({model.USER_ID: user_tensor})
top_k = torch.topk(scores, k=50).indices.cpu().numpy()

# Cek apakah item long tail muncul
recommended_items = [idx2iid[i] for i in top_k]
long_tail_recommended = [item for item in recommended_items if iid2idx[item] in long_tail_item_ids]

print(f"Top-K Recommendation for user {user_id}:")
print(recommended_items)
print("\nLong Tail Items Recommended:")
print(long_tail_recommended)

