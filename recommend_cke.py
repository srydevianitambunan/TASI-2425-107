# running_cke.py

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.knowledge_aware_recommender import CKE
from recbole.trainer import Trainer
from recbole.utils import init_logger, get_model, set_color
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model

import torch
import os
import csv
from collections import Counter
import numpy as np

CONFIG_PATH = './cke_config2.yaml'
LINK_FILE = '/home/tasi2425107/MyProjectExport/Dataset/movie/Amazon_Movies_and_TV_GPT/Amazon_Movies_and_TV/Amazon_Movies_and_TV.link'
MODEL_SAVE_PATH = './saved/CKE_model.pth'

# === STEP 1: Load config and initialize logger ===
config = Config(model='CKE', config_file_list=[CONFIG_PATH])
init_logger(config)
print("\n========= FINAL CONFIG LOADED =========")
print(config)
print("========= END CONFIG =========\n")

# === STEP 2: Load and prepare dataset ===
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
test_data.set_mode('test')

# === STEP 3: Load or Train Model ===
if os.path.exists(MODEL_SAVE_PATH):
    print("\n📦 Memuat model dari cache...")
    model_class = get_model(config["model"])
    model = model_class(config, dataset).to(config["device"])
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=config['device']))
else:
    print("\n🛠 Melatih model baru...")
    model = CKE(config, dataset).to(config['device'])
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(set_color('best valid result: ', 'blue') + f'{best_valid_result}')

# === ALT: Quick Start Mode if needed ===
# config, model, dataset, train_data, valid_data, test_data = load_data_and_model(CONFIG_PATH, MODEL_SAVE_PATH)

# === STEP 4: Analisis long-tail item ===
item_ids = dataset.inter_feat['item_id'].tolist()
item_freq = Counter(item_ids)
freqs = np.array(list(item_freq.values()))
threshold = np.percentile(freqs, 80)
tail_items = set([item_id for item_id, freq in item_freq.items() if freq <= threshold])

print(f"\n📊 Jumlah tail items: {len(tail_items)} dari total {dataset.item_num} items")

# === STEP 5: Load item name map ===
def load_item_names(link_file_path):
    item_name_map = {}
    with open(link_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            raw_item_id = row.get('item_id:token') or row.get('item_id')
            entity_id = row.get('entity_id:token') or row.get('entity_id')
            name = entity_id.replace('res:', '').replace('_', ' ') if entity_id else '(unknown)'
            item_name_map[raw_item_id] = name
    return item_name_map

item_name_map = load_item_names(LINK_FILE)

# === STEP 6: Interactive input ===
def get_tail_recommendations(user_raw_id, model, dataset, config, tail_items):
    user_internal_id = dataset.token2id(dataset.uid_field, [user_raw_id])[0]
    interaction = Interaction({model.USER_ID: torch.tensor([user_internal_id])})
    interaction = interaction.to(config['device'])
    scores = model.full_sort_predict(interaction)
    topk_indices = torch.topk(scores, k=10).indices.squeeze().tolist()
    topk_raw_ids = dataset.id2token(dataset.iid_field, topk_indices)
    tail_in_topk = [item_id for item_id in topk_raw_ids if item_id in tail_items]
    return topk_raw_ids, tail_in_topk

while True:
    user_raw_id = input("\nMasukkan user_id (atau ketik 'exit' untuk keluar): ")
    if user_raw_id.lower() == 'exit':
        break
    try:
        dataset.token2id(dataset.uid_field, [user_raw_id])[0]
    except KeyError:
        print("❌ User ID tidak ditemukan dalam dataset!")
        continue

    topk_raw_ids, tail_in_topk = get_tail_recommendations(user_raw_id, model, dataset, config, tail_items)

    print(f"\n✅ Top-10 rekomendasi untuk user {user_raw_id}:")
    for idx, item_id in enumerate(topk_raw_ids, 1):
        name = item_name_map.get(item_id, '(nama tidak ditemukan)')
        tail_mark = " (tail)" if item_id in tail_in_topk else ""
        print(f"{idx}. {item_id} - {name}{tail_mark}")

    print(f"\n📊 Jumlah tail items dalam rekomendasi: {len(tail_in_topk)}\n")
