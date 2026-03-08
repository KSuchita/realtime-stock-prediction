import pandas as pd

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw stock data:
    - resets index
    - removes duplicates
    - handles missing values
    - ensures correct data types
    """

    # Reset index (Date)
    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Forward-fill missing values (market safe)
    df.ffill(inplace=True)
    df.bfill(inplace=True)


    # Drop remaining NaNs if any
    df.dropna(inplace=True)

    return df