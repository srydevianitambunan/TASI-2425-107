# 1. train_model.py
# Melatih model CKE dan menyimpan model ke folder `saved/`
from recbole.quick_start import run_recbole

run_recbole(model='CKE', config_file_list=['cke_config2.yaml'])

