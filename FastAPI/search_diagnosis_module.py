import numpy as np

def initialize(root_path):
    global diagnose_list
    with open(f'{root_path}/unique_diagnoses.txt', encoding='cp949') as f:
        diagnose_list = np.genfromtxt(f, dtype=str, delimiter='\n')

def search_diagnoses(keyword):
    return [str(diagnosis) for diagnosis in diagnose_list if keyword in diagnosis]
