import yfinance as yf
import pandas as pd
import os
import requests
import io
from datetime import datetime

# Fetch historical exchange rates for GBPNOK
def fetch_exchange_rates(pair, start_date, end_date):
    print(f"Fetching historical exchange rates for {pair} from {start_date} to {end_date}...")
    data = yf.download(pair, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data returned. Check the pair or date range.")
    
    # Reset the index to make 'Date' a column
    data.reset_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]

    # Ensure 'Date' exists after reset_index
    if "Date" not in data.columns:
        raise KeyError("'Date' column is missing after reset_index in fetch_exchange_rates.")
    
    print("Rates DataFrame after resetting index:")
    print(data.head())  # Debugging output to check the structure

    # Rename the columns explicitly to make sure they are consistent
    data.rename(columns={
        "Date": "Date",  # Ensure Date is preserved
        "Adj Close": "Adj Close_GBPNOK=X",
        "Close": "Close_GBPNOK=X",
        "High": "High_GBPNOK=X",
        "Low": "Low_GBPNOK=X",
        "Open": "Open_GBPNOK=X",
        "Volume": "Volume_GBPNOK=X"
    }, inplace=True)

    print("Rates DataFrame columns after renaming:")
    print(data.columns)
    
    return data

def fetch_oil_prices(start_date, end_date):
    print("Fetching oil prices...")
    oil_data = yf.download("BZ=F", start=start_date, end=end_date)
    if oil_data.empty:
        oil_data = pd.DataFrame(columns=["Date", "Oil_Open", "Oil_High", "Oil_Low", "Oil_Close"])
    else:
        # Reset the index to make 'Date' a column
        oil_data.reset_index(inplace=True)

        # Flatten the MultiIndex if present
        if isinstance(oil_data.columns, pd.MultiIndex):
            oil_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in oil_data.columns.values]

        # Rename columns for consistency
        oil_data.rename(columns={
            "Date_": "Date", 
            "Open_": "Oil_Open", 
            "High_": "Oil_High", 
            "Low_": "Oil_Low", 
            "Close_": "Oil_Close"
        }, inplace=True)

        # Ensure 'Date' exists
        if "Date" not in oil_data.columns:
            raise KeyError("'Date' column is missing from the oil prices data.")

    return oil_data


def fetch_wti_prices(start_date, end_date):
    print("Fetching WTI oil prices...")
    wti_data = yf.download("CL=F", start=start_date, end=end_date)
    if wti_data.empty:
        wti_data = pd.DataFrame(columns=["Date", "WTI_Open", "WTI_High", "WTI_Low", "WTI_Close"])
    else:
        # Reset the index to make 'Date' a column
        wti_data.reset_index(inplace=True)

        # Flatten the MultiIndex if present
        if isinstance(wti_data.columns, pd.MultiIndex):
            wti_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in wti_data.columns.values]

        # Rename columns for consistency
        wti_data.rename(columns={
            "Date_": "Date", 
            "Open_": "WTI_Open", 
            "High_": "WTI_High", 
            "Low_": "WTI_Low", 
            "Close_": "WTI_Close"
        }, inplace=True)

        # Ensure 'Date' exists
        if "Date" not in wti_data.columns:
            raise KeyError("'Date' column is missing from the WTI prices data.")

    return wti_data


def fetch_stock_index(start_date, end_date):
    print("Fetching FTSE 100 data...")
    ftse_data = yf.download("^FTSE", start=start_date, end=end_date)
    if ftse_data.empty:
        ftse_data = pd.DataFrame(columns=["Date", "FTSE_Open", "FTSE_High", "FTSE_Low", "FTSE_Close"])
    else:
        # Reset the index to make 'Date' a column
        ftse_data.reset_index(inplace=True)

        # Flatten the MultiIndex if present
        if isinstance(ftse_data.columns, pd.MultiIndex):
            ftse_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in ftse_data.columns.values]

        # Rename columns for consistency
        ftse_data.rename(columns={
            "Date_": "Date", 
            "Open_": "FTSE_Open", 
            "High_": "FTSE_High", 
            "Low_": "FTSE_Low", 
            "Close_": "FTSE_Close"
        }, inplace=True)

        # Ensure 'Date' exists
        if "Date" not in ftse_data.columns:
            raise KeyError("'Date' column is missing from the FTSE data.")

    return ftse_data


def fetch_norges_bank_rate(start_date, end_date):
    print("Fetching Norges Bank interest rate...")
    url = f"https://data.norges-bank.no/api/data/IR/B.KPRA.SD.?format=csv&startPeriod=2001-01-01&endPeriod={end_date}&locale=en"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch Norges Bank interest rate data.")
    norges_data = pd.read_csv(io.StringIO(response.text), sep=';')

    norges_data.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Norges_Rate"}, inplace=True)

    if "Date" not in norges_data.columns:
        raise KeyError("The 'Date' column is missing after renaming. Check the data structure.")

    norges_data["Date"] = pd.to_datetime(norges_data["Date"])

    # Correct Norges Rate if needed
    if not norges_data.empty:
        norges_data["Norges_Rate"] = norges_data["Norges_Rate"].apply(lambda x: 4.5 if x == 3.5 else x)

    return norges_data

def fetch_boe_rate(file_path):
    print("Fetching Bank of England interest rate from local CSV file...")
    boe_data = pd.read_csv(file_path)
    boe_data.rename(columns={"Date Changed": "Date", "Rate": "BOE_Rate"}, inplace=True)
    boe_data["Date"] = pd.to_datetime(boe_data["Date"], dayfirst=True)
    return boe_data

# Merge the datasets and drop unnecessary columns
def merge_data(start_date, end_date, boe_file_path):
    print("Fetching data...")
    exchange_rates = fetch_exchange_rates("GBPNOK=X", start_date, end_date)
    oil_prices = fetch_oil_prices(start_date, end_date)
    wti_prices = fetch_wti_prices(start_date, end_date)
    ftse_data = fetch_stock_index(start_date, end_date)
    norges_rate = fetch_norges_bank_rate(start_date, end_date)
    boe_rate = fetch_boe_rate(boe_file_path)

    # Reset index for all DataFrames to avoid multi-index issues
    exchange_rates.reset_index(inplace=True)
    oil_prices.reset_index(inplace=True)
    wti_prices.reset_index(inplace=True)
    ftse_data.reset_index(inplace=True)
    norges_rate.reset_index(inplace=True)  # Ensure this is also reset
    boe_rate.reset_index(inplace=True)

    # Rename columns to avoid conflicts in the merge
    exchange_rates.rename(columns={"Close": "GBP_to_NOK_Close"}, inplace=True)
    oil_prices.rename(columns={"Close": "Oil_Close", "Open": "Oil_Open"}, inplace=True)
    wti_prices.rename(columns={"Close": "WTI_Close", "Open": "WTI_Open"}, inplace=True)
    ftse_data.rename(columns={"Close": "FTSE_Close", "Open": "FTSE_Open"}, inplace=True)

    # Initialize merged_data as the exchange_rates DataFrame
    merged_data = exchange_rates

    # Now you can safely merge all the other dataframes
    merged_data = merged_data.merge(oil_prices, on="Date", how="left", suffixes=('', '_Oil'))
    merged_data = merged_data.merge(wti_prices, on="Date", how="left", suffixes=('', '_WTI'))
    merged_data = merged_data.merge(ftse_data, on="Date", how="left", suffixes=('', '_FTSE'))
    merged_data = merged_data.merge(norges_rate, on="Date", how="left", suffixes=('', '_Norges'))
    merged_data = merged_data.merge(boe_rate, on="Date", how="left", suffixes=('', '_BOE'))

    # Forward fill missing values for interest rates (and any other non-interpolated fields)
    merged_data['Norges_Rate'] = merged_data['Norges_Rate'].fillna(method='ffill')
    merged_data['BOE_Rate'] = merged_data['BOE_Rate'].fillna(method='ffill')

    # Drop unnecessary columns
    columns_to_drop = [
        'CALC_METHOD', 'Calculation Method', 'index_BOE', 'FREQ', 'Frequency', 
        'INSTRUMENT_TYPE', 'Instrument Type', 'TENOR', 'Tenor', 'UNIT_MEASURE', 
        'Unit of Measure', 'DECIMALS', 'COLLECTION', 'Collection Indicator', 
        'index	Date', 'Volume GBPNOK=X', 'index_Oil', 'Volume_BZ=F', 'index_WTI', 'Volume_CL=F',
        'index_FTSE', 'Volume_^FTSE', 'index_Norges'
    ]
    
    merged_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Impute missing data with linear interpolation (mean of the previous and next available values)
    merged_data.interpolate(method='linear', axis=0, inplace=True)

    # Convert all numeric columns to float type
    for column in merged_data.columns:
        # Only apply the conversion if the column is numeric
        if pd.api.types.is_numeric_dtype(merged_data[column]):
            merged_data[column] = merged_data[column].astype(float)

    return merged_data

# Function to calculate missing data percentage
def calculate_missing_data_percentage(df):
    missing_data = df.isna().sum()  # Count of missing values per column
    total_data = len(df)  # Total number of rows in the DataFrame
    missing_percentage = (missing_data / total_data) * 100  # Calculate percentage of missing data
    missing_percentage_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Percentage': missing_percentage
    })
    return missing_percentage_df

# Save the merged data to CSV
def save_merged_data(merged_data, file_name):
    if not merged_data.empty:
        merged_data.to_csv(file_name, index=False)
        print(f"Merged data saved to {file_name}")
    else:
        print("No data to save.")

def main():
    # Set the date range for the data
    start_date = '2007-12-06'
    end_date = datetime.today().strftime('%Y-%m-%d')
    boe_file_path = "boe_policy_rate.csv"  # Adjust path as needed

    # Merge the data from different sources
    merged_data = merge_data(start_date, end_date, boe_file_path)

    # Calculate missing data percentage
    missing_percentage_df = calculate_missing_data_percentage(merged_data)

    # Display missing data percentage
    print("\nPercentage of Missing Data per Column:")
    print(missing_percentage_df)

    # Save the backfilled merged data to a CSV file
    save_merged_data(merged_data, "GBP_NOK_input_data.csv")

    # Load the existing CSV file
    df = pd.read_csv('GBP_NOK_Input_data.csv')

    # Rename columns to match the neural network's expected column names
    df.rename(columns={
        'Date': 'Date',
        'Close GBPNOK=X': 'GBP_to_NOK_Close',
        'High GBPNOK=X': 'GBP_to_NOK_High',
        'Low GBPNOK=X': 'GBP_to_NOK_Low',
        'Open GBPNOK=X': 'GBP_to_NOK_Open',
        'Close_BZ=F': 'Oil_Close',
        'High_BZ=F': 'Oil_High',
        'Low_BZ=F': 'Oil_Low',
        'Open_BZ=F': 'Oil_Open',
        'Close_CL=F': 'WTI_Close',
        'High_CL=F': 'WTI_High',
        'Low_CL=F': 'WTI_Low',
        'Open_CL=F': 'WTI_Open',
        'Close_^FTSE': 'FTSE_Close',
        'High_^FTSE': 'FTSE_High',
        'Low_^FTSE': 'FTSE_Low',
        'Open_^FTSE': 'FTSE_Open',
        'Norges_Rate': 'Norges_Rate',
        'BOE_Rate': 'BOE_Rate'
    }, inplace=True)

    # Save the updated DataFrame back to the same file
    df.to_csv('GBP_NOK_Input_data.csv', index=False)

    print("Column names have been updated and saved to the same file.")

    # Step 2: Split into training and testing datasets
    # Load the existing CSV file again
    df = pd.read_csv('GBP_NOK_Input_data.csv')

    split_ratio = 0.9  # 90% training, 10% testing
    split_index = int(len(merged_data) * split_ratio)

    train_df = merged_data.iloc[:split_index]
    test_df = merged_data.iloc[split_index:]

    # Save training and testing datasets
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    # Load the existing CSV file
    df = pd.read_csv('train_data.csv')

    # Rename columns to match the neural network's expected column names
    df.rename(columns={
        'Date': 'Date',
        'Close GBPNOK=X': 'GBP_to_NOK_Close',
        'High GBPNOK=X': 'GBP_to_NOK_High',
        'Low GBPNOK=X': 'GBP_to_NOK_Low',
        'Open GBPNOK=X': 'GBP_to_NOK_Open',
        'Close_BZ=F': 'Oil_Close',
        'High_BZ=F': 'Oil_High',
        'Low_BZ=F': 'Oil_Low',
        'Open_BZ=F': 'Oil_Open',
        'Close_CL=F': 'WTI_Close',
        'High_CL=F': 'WTI_High',
        'Low_CL=F': 'WTI_Low',
        'Open_CL=F': 'WTI_Open',
        'Close_^FTSE': 'FTSE_Close',
        'High_^FTSE': 'FTSE_High',
        'Low_^FTSE': 'FTSE_Low',
        'Open_^FTSE': 'FTSE_Open',
        'Norges_Rate': 'Norges_Rate',
        'BOE_Rate': 'BOE_Rate'
    }, inplace=True)

    # Save the updated DataFrame back to the same file
    df.to_csv('train_data.csv', index=False)

    # Load the existing CSV file
    df = pd.read_csv('test_data.csv')

    # Rename columns to match the neural network's expected column names
    df.rename(columns={
        'Date': 'Date',
        'Close GBPNOK=X': 'GBP_to_NOK_Close',
        'High GBPNOK=X': 'GBP_to_NOK_High',
        'Low GBPNOK=X': 'GBP_to_NOK_Low',
        'Open GBPNOK=X': 'GBP_to_NOK_Open',
        'Close_BZ=F': 'Oil_Close',
        'High_BZ=F': 'Oil_High',
        'Low_BZ=F': 'Oil_Low',
        'Open_BZ=F': 'Oil_Open',
        'Close_CL=F': 'WTI_Close',
        'High_CL=F': 'WTI_High',
        'Low_CL=F': 'WTI_Low',
        'Open_CL=F': 'WTI_Open',
        'Close_^FTSE': 'FTSE_Close',
        'High_^FTSE': 'FTSE_High',
        'Low_^FTSE': 'FTSE_Low',
        'Open_^FTSE': 'FTSE_Open',
        'Norges_Rate': 'Norges_Rate',
        'BOE_Rate': 'BOE_Rate'
    }, inplace=True)

    # Save the updated DataFrame back to the same file
    df.to_csv('test_data.csv', index=False)

    print(f"\n- Training dataset saved as train_data.csv ({len(train_df)} rows)"
          f"\n- Testing dataset saved as test_data.csv ({len(test_df)} rows)")

if __name__ == "__main__":
    main()

