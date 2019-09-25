import pandas as pd

def make_quintiles(x):
    # Create labels and bins of the same size
    labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    quintiles = [[] for l in labels]
    mapping = pd.qcut(x, len(labels), labels=False)
    for idx_input, idx_output in enumerate(result):
        #idx_qcut = labels.index(associated_label)  # Find label's index
        quintiles[idx_output].append(x[idx_input])
    return quintiles
