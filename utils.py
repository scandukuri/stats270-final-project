import pandas as pd

def preprocess_data(raw_data: pd.DataFrame) -> list:
    # Load data
    data = []
    for i, row in raw_data.iterrows():
        data.append(((row[1], row[2]), int(row[0])))
    return data