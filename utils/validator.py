import pandas as pd

REQUIRED_COLUMNS = ['Customer ID', 'Order Date', 'Sales']

def validate_schema(df: pd.DataFrame):
    """
    Validates if the uploaded dataframe contains the required columns.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Error: Missing required columns -> {', '.join(missing_cols)}"
    return True, "Success: Schema validation passed."