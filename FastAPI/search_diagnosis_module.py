import numpy as np

def initialize(root_path):
    global diagnose_list
    diagnose_list = np.genfromtxt(f'{root_path}/unique_diagnoses.txt', dtype=str, delimiter='\n')

def search_diagnoses(keyword):
    return [str(diagnosis) for diagnosis in diagnose_list if keyword in diagnosis]