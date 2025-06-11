from pathlib import Path

# Let's prepare the cleaned version of running_cke.py content
cleaned_code = """
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.knowledge_aware_recommender import CKE
from recbole.trainer import Trainer
from recbole.utils import init_logger, set_color
from recbole.data.interaction import Interaction

import torch
import os
import pickle
import csv
from collections import Counter
import numpy as np

# === STEP 1: Load config ===
config = Config(model='CKE', config_file_list=['/home/tasi2425107/MyProjectExport/cke_config2.yaml'])

print("\\n========= FINAL CONFIG LOADED =========")
print(config)
print("========= END CONFIG =========\\n")
print("📁 Dataset path:", config['data_path'])
print("🔎 Field separator:", repr(config['field_separator']))
print("✅ Field separator terbaca sebagai:", repr(config['field_separator']))

# === STEP 2: Load dataset and logger ===
init_logger(config)
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# === STEP 3: Load or Train Model ===
model_file = os.path.join(config['checkpoint_dir'], f"{config['model']}.pth")
if os.path.exists(model_file):
    print("📦 Loading model from checkpoint...")
    model = CKE(config, train_data.dataset)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(config['device'])))
    model = model.to(config['device'])
else:
    print("🛠️ Training new model...")
    model = CKE(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    print(set_color('best valid result: ', 'blue') + f'{best_valid_result}')
    torch.save(model.state_dict(), model_file)

# === STEP 4: Evaluate model ===
trainer = Trainer(config, model)
test_result = trainer.evaluate(test_data)
print(set_color('test result: ', 'blue') + f'{test_result}')

# === STEP 5: Analyze sparsity & tail items ===
item_ids = dataset.inter_feat['item_id'].tolist()
item_freq = Counter(item_ids)
freqs = np.array(list(item_freq.values()))
threshold = np.percentile(freqs, 80)
tail_items = set([item_id for item_id, freq in item_freq.items() if freq <= threshold])
print(f"📊 Threshold tail item (80% terbawah): {threshold:.2f}")
print(f"📊 Jumlah tail items (<= threshold): {len(tail_items)} dari total {dataset.item_num} items")

# === STEP 6: Load item readable names ===
def load_item_names(link_file_path):
    item_name_map = {}
    with open(link_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\\t')
        for row in reader:
            item_id = row['item_id:token']
            entity_id = row['entity_id:token']
            readable_name = entity_id.replace('res:', '').replace('_', ' ')
            item_name_map[item_id] = readable_name
    return item_name_map

link_file = os.path.join(config['data_path'], f"{config['file_name']}.link")
item_name_map = load_item_names(link_file)

# === STEP 7: Interactive Recommendation ===
def get_recommendations_for_user(user_raw_id):
    try:
        user_internal_id = dataset.token2id(dataset.uid_field, [user_raw_id])[0]
    except KeyError:
        print("❌ User ID tidak ditemukan dalam dataset!")
        return

    interaction = Interaction({model.USER_ID: torch.tensor([user_internal_id])})
    interaction = interaction.to(config['device'])
    scores = model.full_sort_predict(interaction)
    topk_indices = torch.topk(scores, k=10).indices.squeeze().tolist()
    topk_raw_ids = dataset.id2token(dataset.iid_field, topk_indices)
    tail_in_topk = [item_id for item_id in topk_raw_ids if item_id in tail_items]

    print(f"\\n✅ Top-10 rekomendasi item_id untuk user {user_raw_id}:")
    for idx, item_id in enumerate(topk_raw_ids, 1):
        name = item_name_map.get(item_id, '(nama tidak ditemukan)')
        tail_marker = " (tail)" if item_id in tail_in_topk else ""
        print(f"{idx}. {item_id} - {name}{tail_marker}")

    print(f"\\n📊 Jumlah tail items di Top-10: {len(tail_in_topk)}\\n")

while True:
    user_raw_id = input("Masukkan user_id (atau ketik 'exit'): ")
    if user_raw_id.lower() == 'exit':
        break
    get_recommendations_for_user(user_raw_id)
"""

# Save it as running_cke_cleaned.py
path = Path("/mnt/data/running_cke_cleaned.py")
path.write_text(cleaned_code)

path.name

