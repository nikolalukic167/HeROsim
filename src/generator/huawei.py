import os
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_days_of_month_from_minutes(minutes_passed_tuple):
    # Convert each minute number to a day number (0-based)
    days = []
    for minutes_passed in minutes_passed_tuple:
        # Calculate how many days have passed (integer division)
        days_passed = minutes_passed // (24 * 60)
        days.append(days_passed)

    # Return unique days in order
    return sorted(list(set(days)))


def preprocess_day_file(file_path, fn):
    df = pd.read_csv(file_path)
    df = df.drop(['day'], axis=1)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Split column names into function ID, user, and pool
    column_info = [col.split('---') for col in df.columns if '---' in col]
    new_columns = new_columns = [f"{func}" for func, _, _ in column_info]

    # Assign the new MultiIndex to the DataFrame columns
    df.columns = new_columns

    # Group by function ID and user, summing over pools
    grouped = df.groupby(df.columns, axis=1).sum().fillna(0)

    return grouped[[fn]]


def prepare_weekly_data(preprocessed_data):
    # Combine all days into a single DataFrame
    combined_df = pd.concat(preprocessed_data.values())

    # Ensure the data starts on a Monday
    start_date = combined_df.index.min()
    # while start_date.dayofweek != 0:  # 0 represents Monday
    #     start_date += pd.Timedelta(days=1)

    # Truncate data to start from the first Monday
    combined_df = combined_df.loc[start_date:]

    # Resample to minute frequency and fill missing values
    combined_df = combined_df.resample('T').asfreq().fillna(0)

    # Calculate the number of complete weeks
    minutes_per_week = 7 * 24 * 60
    total_minutes = len(combined_df)
    complete_weeks = total_minutes // minutes_per_week

    # Reshape the data into weekly chunks for each function-user group
    weekly_data = {}
    for column in combined_df.columns:
        series = combined_df[column].values[:complete_weeks * minutes_per_week]
        # weekly_data[column] = np.mean(series.reshape(complete_weeks, minutes_per_week), axis=0).reshape(1,-1)
        weekly_data[column] = series.reshape(complete_weeks, minutes_per_week)

    return weekly_data


def read_and_preprocess_files(directory, from_to, fn):
    preprocessed_data = {}
    for day_id in range(31):  # Assuming 7 days of data
        file_name = f'day_{day_id:02d}.csv'
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            preprocessed_data[day_id] = preprocess_day_file(file_path, fn)
        else:
            print(f"File not found: {file_name}")
    return preprocessed_data


def normalize_fn_patterns(df, fn):
    copy_df = df.copy()
    mean = np.mean(copy_df[fn])
    std = np.std(copy_df[fn])
    # Avoid division by zero
    if std == 0:
        copy_df[fn] = copy_df[fn] - mean
    else:
        copy_df[fn] = (copy_df[fn] - mean) / std
    return copy_df


def read_huawei_dfs(days, time_window, fn, region):
    start = pd.to_datetime(time_window[0], unit='m')
    end = pd.to_datetime(time_window[1], unit='m')
    home = Path.home()
    directory = str(home / f'resources/datasets/huawei_public_cloud/{region}/requests')
    data = read_and_preprocess_files(directory, days, fn)
    combined_df = pd.concat(data.values()).fillna(0)
    combined_df = normalize_fn_patterns(combined_df, fn)
    combined_df = combined_df.loc[start:end]
    return combined_df


def fetch_huawei_arrival_times(fn, region, time_window, simulation_duration, new_average_rps, new_std_rps):
    days = get_days_of_month_from_minutes(time_window)
    df = read_huawei_dfs(days, time_window, fn, region)
    df[fn] = (df[fn] * new_std_rps) + new_average_rps
    sns.scatterplot(x=df.index, y=fn, data=df)
    plt.show()


def main():
    fn = '49'
    region = 'R1'
    # first week
    time_window = (0, 10080)
    simulation_duration = 20 * 60
    new_average_rps = 30
    new_std_rps = 15
    arrival_times = fetch_huawei_arrival_times(fn, region, time_window, simulation_duration, new_average_rps,
                                               new_std_rps)


if __name__ == '__main__':
    main()
