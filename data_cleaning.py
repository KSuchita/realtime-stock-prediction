import pandas as pd

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock dataset
    """

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    df.drop_duplicates(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    # Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.dropna(inplace=True)

    return df