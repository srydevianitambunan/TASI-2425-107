# run_cke.py

from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(model='CKE', config_file_list=['cke_config.yaml'])