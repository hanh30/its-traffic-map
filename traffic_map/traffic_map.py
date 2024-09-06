import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
from scipy.spatial import KDTree
import pyproj
import itertools


NORTH = 21.09
SOUTH = 20.94
EAST = 105.93
WEST = 105.73

current_path = os.path.dirname(os.path.abspath(__file__))
file_path = f'{current_path}/../db/traffic.xlsx'


# sua < TIME, >= TIMESTART
def read_data_vehicle(FILE_PATH_VEHICLE, DATE, TIME, duration=5):
    TIME_START = (datetime.strptime(TIME, '%H:%M:%S') - timedelta(minutes=duration)).strftime('%H:%M:%S')

    df = pd.read_csv(FILE_PATH_VEHICLE)
    print(df.shape)

    df['datetime'] = pd.to_datetime(df['datetime'])

    df = df[(df['datetime'].dt.date.astype(str) == DATE) & (df['time']<=TIME) & (df['time']>TIME_START)]
    print(df.shape)

    df = df.sort_values(by=['vehicle', 'datetime', 'arrival_time']).reset_index(drop=True)

    df.drop_duplicates(subset=['vehicle', 'datetime'], inplace=True)
    print(df.shape)

    df.rename(columns={'x':'y', 'y':'x'}, inplace=True)

    df = df[(df['x']<=NORTH) & (df['x']>=SOUTH) & (df['y']>=WEST) & (df['y']<=EAST)].reset_index(drop=True)
    
    df['speed'] = np.where(df['speed']<0, 0, df['speed'])
    print(df.shape)
    
    return df


# Vectorized function to calculate geodesic distance
def _vectorized_geodesic(lat1, lon1, lat2, lon2):
    distances = np.vectorize(lambda x1, y1, x2, y2: geodesic((x1, y1), (x2, y2)).meters if not any(pd.isnull([x1, y1, x2, y2])) else np.nan)
    return distances(lat1, lon1, lat2, lon2)


def preprocessing(df):
    df['datetime_last'] = df.groupby('vehicle')['datetime'].shift(1)
    df['x_last'] = df.groupby('vehicle')['x'].shift(1)
    df['y_last'] = df.groupby('vehicle')['y'].shift(1)
    df['speed_last'] = df.groupby('vehicle')['speed'].shift(1)
    df['duration_last'] = (df['datetime'] - df['datetime_last']).dt.total_seconds()

    df['datetime_next'] = df.groupby('vehicle')['datetime'].shift(-1)
    df['x_next'] = df.groupby('vehicle')['x'].shift(-1)
    df['y_next'] = df.groupby('vehicle')['y'].shift(-1)
    df['speed_next'] = df.groupby('vehicle')['speed'].shift(-1)
    df['duration_next'] = (df['datetime_next'] - df['datetime']).dt.total_seconds()

    df['distance_last'] = _vectorized_geodesic(df['x'], df['y'], df['x_last'], df['y_last'])
    df['distance_next'] = _vectorized_geodesic(df['x'], df['y'], df['x_next'], df['y_next'])

    df['speed_avg_last'] = (df['distance_last']/1000)/(df['duration_last']/3600)
    df['speed_avg_next'] = (df['distance_next']/1000)/(df['duration_next']/3600)

    df['is_accelerate'] = np.where(
        (df['speed_avg_last'] == df['speed_avg_next']) | ((df['speed']==0) & (df['speed_last']==0) & (df['speed_next']==0)),
        0, np.where(
            df['speed_avg_last'] < df['speed_avg_next'], 1, -1
        )
    )

    df['accelerate'] = (df['speed_avg_next'] - df['speed_avg_last'])/((df['duration_last'] + df['duration_next'])/2)

    distance_threshold = 200
    df['speed_avg'] = np.where(
        (df['speed']==0) & (df['speed_last']==0) & (df['speed_next']==0),
        0, 
        np.where(
            (df['distance_last'] <= distance_threshold) & (df['distance_next'] <= distance_threshold),
            (df['speed'] + df['speed_last'] + df['speed_next'] + df['speed_avg_last'] + df['speed_avg_next'])/5,
            np.where(
                (df['distance_last'] <= distance_threshold) & (df['distance_next'] > distance_threshold),
                (df['speed'] + df['speed_last'] + df['speed_avg_last'])/3,
                np.where(
                    (df['distance_last'] > distance_threshold) & (df['distance_next'] <= distance_threshold),
                    (df['speed'] + df['speed_next'] + df['speed_avg_next'])/3,
                    df['speed']
                )
            )
        )
    )

    # Compute heading of vehicles
    geod = pyproj.Geod(ellps='WGS84')
    heading, _, _ = geod.inv(df['y'], df['x'], df['y_next'], df['x_next'])

    df['heading'] = np.where(
        (df['x']==df['x_next']) & (df['y']==df['y_next']),
        0, # set heading = 0 if latitude = longitude, in case the above calculation returns 360
        heading
    )

    # Convert negative heading to positive value
    df['heading'] = np.where(
        df['heading'] >= 0,
        df['heading'],
        df['heading'] + 360
    )

    df.drop(columns=['hour', 'min', 'sec', 'datetime_last', 'x_last', 'y_last', 'datetime_next', 'x_next', 'y_next'], inplace=True)

    df = df[df['speed_avg']!=0]

    return df


def insert_heading_col(df):
    df = df.sort_values(by=['street', 'direction', 'order']).reset_index(drop=True)

    df['x_next'] = df.groupby(['street', 'direction'])['x'].shift(-1)
    df['y_next'] = df.groupby(['street', 'direction'])['y'].shift(-1)

    # Compute heading
    geod = pyproj.Geod(ellps='WGS84')
    heading, _, _ = geod.inv(df['y'], df['x'], df['y_next'], df['x_next'])

    df['heading'] = np.where(
        (df['x']==df['x_next']) & (df['y']==df['y_next']),
        0, # set heading = 0 if two consecutive coords are the same, in case the above calculation returns 360
        heading
    )

    # Convert negative heading to positive value
    df['heading'] = np.where(
        df['heading'] >= 0,
        df['heading'],
        df['heading'] + 360
    )

    df.drop(columns=['x_next', 'y_next'], inplace=True)
    df['heading'].fillna(method='ffill', inplace=True)

    return df


# def find_closest_point(df, df_street, angle_threshold=45):
#     closest_points = []

#     for _, row in df.iterrows():
#         if row['heading'] < angle_threshold:
#             df_street_filter = df_street[(df_street['heading'] <= row['heading'] + angle_threshold) | (df_street['heading'] >= row['heading'] - angle_threshold + 360)]
#         elif row['heading'] >= 360 - angle_threshold:
#             df_street_filter = df_street[(df_street['heading'] >= row['heading'] - angle_threshold) | (df_street['heading'] <= row['heading'] + angle_threshold - 360)]
#         else:
#             df_street_filter = df_street[(df_street['heading'] >= row['heading'] - angle_threshold) & (df_street['heading'] <= row['heading'] + angle_threshold)]
        
#         coors = list(zip(df_street['x'], df_street['y']))
#         kd_tree = KDTree(coors)

#         dist, idx = kd_tree.query((row['x'], row['y']))
#         closest_points.append(coors[idx])

#     return closest_points


def find_closest_point(df_vehicle, df_street, angle_threshold=45, chunk_size=5000):
    closest_points = []

    for start_idx in range(0, len(df_vehicle), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df_vehicle))
        vehicle_chunk = df_vehicle.iloc[start_idx:end_idx]

        # Extract coordinates and headings
        vehicle_coords = vehicle_chunk[['x', 'y']].values
        vehicle_headings = vehicle_chunk['heading'].values
        
        street_coords = df_street[['x', 'y']].values
        street_headings = df_street['heading'].values

        # Compute the angle differences
        angle_diff = np.abs(vehicle_headings[:, None] - street_headings[None, :])
        
        # Adjust angles larger than 180 degrees
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        
        # Create a mask for valid angles within the threshold
        valid_mask = angle_diff <= angle_threshold

        # Handle cases where there are no valid matches
        if not np.any(valid_mask):
            return np.full((len(df_vehicle), 2), np.nan)

        # Filter route coordinates based on valid mask
        filtered_coords = np.where(valid_mask[:, :, None], street_coords[None, :, :], np.inf)

        # Compute the squared distances for valid points
        dist_squared = np.sum((filtered_coords - vehicle_coords[:, None, :]) ** 2, axis=2)
        
        # Find the index of the minimum distance
        closest_indices = np.argmin(dist_squared, axis=1)

        # Get the closest points from the original route coordinates
        closest_points.append(street_coords[closest_indices])
    
    closest_points = np.vstack(closest_points)

    df_vehicle['x_closest'] = [c[0] for c in closest_points]
    df_vehicle['y_closest'] = [c[1] for c in closest_points]
    df_vehicle['distance_closest'] = _vectorized_geodesic(df_vehicle['x'], df_vehicle['y'], df_vehicle['x_closest'], df_vehicle['y_closest'])

    return df_vehicle


def get_map_color(df_vehicle, df_street, window_size=10):
    df_street_merged = pd.merge(
        df_street, df_vehicle[['x_closest', 'y_closest', 'speed_avg']].rename(columns={'x_closest':'x', 'y_closest':'y'}), 
        on=['x', 'y'], how='left'
    )

    df_street_group = df_street_merged.groupby(df_street.columns.tolist())['speed_avg'].apply(list).reset_index()
    df_street_group = df_street_group.sort_values(by=['street', 'direction', 'order']).reset_index(drop=True)

    print(f'df_street_group = {df_street_group.shape}')

    df_rolling = pd.DataFrame({
        'street': df_street_group['street'],
        'type': df_street_group['type'],
        'direction': df_street_group['direction'],
        'order': [window.to_list() for window in df_street_group.groupby(['street', 'direction'])['order'].rolling(window_size)],
        'speed_avg': [window.to_list() for window in df_street_group.groupby(['street', 'direction'])['speed_avg'].rolling(window_size)]
    })
    print(f'df_rolling = {df_rolling.shape}')

    df_rolling = df_rolling[df_rolling['order'].apply(lambda x: len(x)) == window_size].reset_index(drop=True)
    print(f'df_rolling = {df_rolling.shape}')

    df_rolling['speed_avg'] = df_rolling['speed_avg'].apply(lambda x: list(itertools.chain.from_iterable(x)))
    df_rolling['speed_avg'] = df_rolling['speed_avg'].apply(lambda x: [i for i in x if not pd.isna(i)])
    df_rolling['speed_median'] = df_rolling['speed_avg'].apply(lambda x: np.median(x) if x else np.nan)

    df_color_mapping = pd.read_excel(file_path, sheet_name='color')

    df_rolling['color'] = np.where(
        df_rolling['speed_median'] >= 15,
        'green',
        np.where(
            df_rolling['speed_median'] >= 10,
            'yellow',
            np.where(
                df_rolling['speed_median'] >= 0,
                'red',
                'blue'
            )
        )
    )

    df_explode = df_rolling[['street', 'type', 'direction', 'order', 'color']].explode('order').reset_index(drop=True)

    priority_order = ['green', 'yellow', 'red', 'darkred']

    # Group by the 'group' column and get the mode, respecting priority order
    def get_mode_with_priority(series):
        # Get the mode(s)
        modes = series.mode()
        
        if len(modes) > 1:
            # If there is a tie, choose based on priority
            for value in priority_order:
                if value in modes.values:
                    return value
        else:
            # If no tie, return the single mode
            return modes[0]

    # Apply the function to get the mode with priority for each group
    df_mode = df_explode.groupby(['street', 'direction', 'order'])['color'].apply(get_mode_with_priority).reset_index()
    print(f'df_mode = {df_mode.shape}')

    df_street_color = df_street_group.merge(df_mode, how='left', on=['street', 'direction', 'order'])
    print(f'df_street_color = {df_street_color.shape}')
    
    return df_street_color, df_rolling
