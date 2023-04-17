"""
Программа: Сохранение данных
Версия: 1.0
"""

import pandas as pd

def load_csv(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index_label=True)