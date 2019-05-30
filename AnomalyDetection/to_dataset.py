import pandas as pd
from utils import utilities
from pathlib import Path

features_selected = ['voltage [V]', 'acceleration (actual) [m/(s*s)]']
pathlist = Path("/home/aggelos-i3/Downloads/simu Elbas/7h33NO").glob(
    '**/*.xls')

df = utilities.feature_selection(features=features_selected, pathlist=pathlist)

df.to_csv("normal_dataset.csv")
