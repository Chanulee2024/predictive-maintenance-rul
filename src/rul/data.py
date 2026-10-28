from pathlib import Path
import pandas as pd 

COLS = ['unit', 'cycle', 'op1','op2', 'op3'] + [f"s{i}" for i in range(1,22)]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_fd(subset):
    '''Load train, test, and RUL data for a given subset (e.g., FD001)'''
    read = lambda name: pd.read_csv(DATA_DIR/f"{name}_{subset}.txt", sep = r"\s+", header = None, names = COLS)
    train, test = read("train"), read("test")
    rul = pd.read_csv(DATA_DIR/f"RUL_{subset}.txt", header = None, names = ['RUL'])
    rul.index = rul.index+1
    rul.index.name = 'unit'
    return train, test, rul

def add_rul(df, cap=None):
    '''Add RUL column to a dataframe with unit and cycle columns'''
    out = df.copy()
    out['RUL'] = out.groupby('unit')['cycle'].transform(max) - out['cycle']
    if cap is not None:
        out['RUL'] = out['RUL'].clip(upper=cap)
    return out

def constant_columns(df, tol = 1e-6):
    '''Return a list of columns that are constant (std < tol)'''
    stds = df.drop(columns = ['unit','cycle']).std()
    return stds[stds < tol].index.tolist()

