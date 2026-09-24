import pandas as pd
import numpy as np

def calculate_rfm(df: pd.DataFrame):
    """
    Aggregates transactional data into Customer-level RFM features.
    """
    # Force pandas to parse day-first mixed date formats safely
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True, errors='coerce')
    
    # Drop rows where date couldn't be parsed
    df = df.dropna(subset=['Order Date'])
    
    reference_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (reference_date - x.max()).days,  # Recency
        'Order ID': 'nunique',                                    # Frequency
        'Sales': 'sum'                                            # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    
    return rfm

def prepare_time_series_data(df: pd.DataFrame):
    """
    Aggregates transactional data by date for time series forecasting.
    """
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Order Date'])
    
    # Group by date and sum sales
    ts_df = df.groupby('Order Date')['Sales'].sum().reset_index()
    ts_df = ts_df.sort_values('Order Date')
    
    # Set date as index and reindex to fill missing dates with 0 sales
    ts_df.set_index('Order Date', inplace=True)
    ts_df = ts_df.resample('D').sum().fillna(0)
    ts_df.reset_index(inplace=True)
    
    return ts_df