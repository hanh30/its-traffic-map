import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
from scipy.spatial import KDTree
import pyproj
import itertools


# Define geographical boundaries for Hanoi
NORTH = 21.09
SOUTH = 20.94
EAST = 105.93
WEST = 105.73


def read_data_vehicle(FILE_PATH_VEHICLE, DATE, TIME, duration=5):
    '''
    Read and process vehicle data (including GPS info).

    Args:
        FILE_PATH_VEHICLE (str): Path to the vehicle data CSV file.
        DATE (str): The date for filtering the data (in 'YYYY-MM-DD' format).
        TIME (str): The end time for the time window (in 'HH:MM:SS' format).
        duration (int): The time window duration in minutes before the specified TIME (default is 5 minutes).
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with vehicle data.
    '''
    
    # Calculate the start time based on the duration (minutes) before the given TIME.
    TIME_START = (datetime.strptime(TIME, '%H:%M:%S') - timedelta(minutes=duration)).strftime('%H:%M:%S')

    df = pd.read_csv(FILE_PATH_VEHICLE)
    # print(df.shape)

    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter data based on the specified date and time range
    df = df[(df['datetime'].dt.date.astype(str) == DATE) & (df['time']<TIME) & (df['time']>=TIME_START)]
    # print(df.shape)

    df = df.sort_values(by=['vehicle', 'datetime', 'arrival_time']).reset_index(drop=True)
    df.drop_duplicates(subset=['vehicle', 'datetime'], inplace=True)
    # print(df.shape)

    # df.rename(columns={'x':'y', 'y':'x'}, inplace=True)

    # Filter data based on geographic boundaries
    df = df[(df['x']<=NORTH) & (df['x']>=SOUTH) & (df['y']>=WEST) & (df['y']<=EAST)].reset_index(drop=True)
    # print(df.shape)

    # Ensure non-negative speeds by setting negative speed values to 0
    df['speed'] = np.where(df['speed']<0, 0, df['speed'])
    
    return df


# Vectorized function to calculate geodesic distance between two coordinates
def _vectorized_geodesic(lat1, lon1, lat2, lon2):
    distances = np.vectorize(lambda x1, y1, x2, y2: geodesic((x1, y1), (x2, y2)).meters if not any(pd.isnull([x1, y1, x2, y2])) else np.nan)
    return distances(lat1, lon1, lat2, lon2)


def preprocessing(df):
    # Create lagged and lead columns for datetime, vehicle coordinates and speed
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

    # Calculate average speeds (in km/h) using the distances and durations between consecutive points
    df['speed_avg_last'] = (df['distance_last']/1000)/(df['duration_last']/3600)
    df['speed_avg_next'] = (df['distance_next']/1000)/(df['duration_next']/3600)

    df['is_accelerate'] = np.where(
        (df['speed_avg_last'] == df['speed_avg_next']) | ((df['speed']==0) & (df['speed_last']==0) & (df['speed_next']==0)),
        0, np.where(
            df['speed_avg_last'] < df['speed_avg_next'], 1, -1
        )
    )

    df['accelerate'] = (df['speed_avg_next'] - df['speed_avg_last'])/((df['duration_last'] + df['duration_next'])/2)

    # Calculate speed_avg based on speed, speed_last, speed_avg_last, speed_next, speed_avg_next
    # If speed = speed_last = speed_next = 0 then speed_avg = 0
    # else calculate speed_avg as average of above values of speed,
    # on condition that distance last/next < distance_threshold (in meter)
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

    # Compute heading of vehicles (angle between vehicle's direction and the north)
    geod = pyproj.Geod(ellps='WGS84')
    heading, _, _ = geod.inv(df['y'], df['x'], df['y_next'], df['x_next'])

    df['heading'] = np.where(
        (df['x']==df['x_next']) & (df['y']==df['y_next']),
        0, # set heading = 0 if current coordinate = next coordinate, in case the above calculation returns 360
        heading
    )

    # Convert negative heading to positive value
    df['heading'] = np.where(
        df['heading'] >= 0,
        df['heading'],
        df['heading'] + 360
    )

    # Drop unnecessary columns
    df.drop(columns=['hour', 'min', 'sec', 'datetime_last', 'x_last', 'y_last', 'datetime_next', 'x_next', 'y_next'], inplace=True)

    # Filter out rows where speed_avg = 0 to remove stopped vehicles
    df = df[df['speed_avg']!=0]

    return df


def insert_heading_col(df):
    '''
    Calculate the heading for streets based on consecutive points, grouping by street and direction.
    '''
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

    # Forward fill missing heading values (the last point on the street)
    df['heading'] = df['heading'].ffill()

    # Set heading to NaN for streets without direction
    df['heading'] = np.where(
        df['direction'] == 0,
        np.nan,
        df['heading']
    )

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


def find_closest_point(df_vehicle, df_street, angle_threshold=45, distance_closest_threshold = 10, chunk_size=5000):
    '''
    Finds the closest point on the street for each vehicle data point based on spatial proximity and heading direction.
    
    Args:
        df_vehicle (pd.DataFrame): Vehicle data with 'x' (latitude), 'y' (longitude), and 'heading' columns.
        df_street (pd.DataFrame): Street data with 'x' (latitude), 'y' (longitude), and 'heading' columns.
        angle_threshold (float): Maximum angle difference (in degrees) to consider for matching the vehicle heading to street heading.
        distance_closest_threshold (float): Maximum distance (in meters) to consider a point as the closest.
        chunk_size (int): Number of vehicle records to process in each chunk to manage memory usage.

    Returns:
        pd.DataFrame: Updated vehicle DataFrame with the closest point's coordinates and distance.
    '''
    
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

        # For streets without direction, fill the angle differences = 0
        angle_diff[np.isnan(angle_diff)] = 0
        
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

    # Add columns for closest point coordinates and distance to df_vehicle
    df_vehicle['x_closest'] = [c[0] for c in closest_points]
    df_vehicle['y_closest'] = [c[1] for c in closest_points]
    df_vehicle['distance_closest'] = _vectorized_geodesic(df_vehicle['x'], df_vehicle['y'], df_vehicle['x_closest'], df_vehicle['y_closest'])

    # Filter to keep rows where the closest distance is within the threshold
    df_vehicle = df_vehicle[df_vehicle['distance_closest']<=distance_closest_threshold]

    return df_vehicle


def get_map_color(df_vehicle, df_street, window_size=10, weight=0.5):
    '''
    Determines the color for each street point based on average speeds and predefined color mappings.
    
    Args:
        df_vehicle (pd.DataFrame): Vehicle data
        df_street (pd.DataFrame): Street data
        window_size (int): Size of the rolling window to compute average speed for each street segment.

    Returns:
        pd.DataFrame: Street DataFrame with added color information based on speed and predefined color mappings.
    '''

    # Merge vehicle data with street data based on closest points
    df_street_merged = df_street.merge(
        df_vehicle[['x_closest', 'y_closest', 'speed_avg']].rename(columns={'x_closest':'x', 'y_closest':'y'}), 
        on=['x', 'y'], how='left'
    )

    # Group by street and direction and compute lists of average speeds
    df_street_group = df_street_merged.groupby(['street', 'direction', 'order'])['speed_avg'].apply(list).reset_index()
    df_street_group = df_street.merge(df_street_group, how='left', on=['street', 'direction', 'order'])
    df_street_group = df_street_group.sort_values(by=['street', 'direction', 'order']).reset_index(drop=True)

    # print(f'df_street_group = {df_street_group.shape}')

    # Create rolling windows for order and speed_avg
    df_rolling = pd.DataFrame({
        'street': df_street_group['street'],
        'type': df_street_group['type'],
        'direction': df_street_group['direction'],
        'order': [window.to_list() for window in df_street_group.groupby(['street', 'direction'])['order'].rolling(window_size)],
        'speed_avg': [window.to_list() for window in df_street_group.groupby(['street', 'direction'])['speed_avg'].rolling(window_size)]
    })
    # print(f'df_rolling = {df_rolling.shape}')

    # Filter out rows where rolling window size does not match the specified window_size
    df_rolling = df_rolling[df_rolling['order'].apply(lambda x: len(x)) == window_size].reset_index(drop=True)
    # print(f'df_rolling = {df_rolling.shape}')

    # Flatten the list of speed_avg
    df_rolling['speed_avg'] = df_rolling['speed_avg'].apply(lambda x: list(itertools.chain.from_iterable(x)))
    
    # Remove NaN item from each list of speed_avg
    df_rolling['speed_avg'] = df_rolling['speed_avg'].apply(lambda x: [i for i in x if not pd.isna(i)])
    
    # Compute median speed for each street point
    df_rolling['speed_median'] = df_rolling['speed_avg'].apply(lambda x: np.median(x) if x else np.nan)

    # Load color mapping data
    df_color_mapping = pd.read_excel(file_path, sheet_name='color')
    df_color_mapping['speed_in_traffic_min'] = weight * df_color_mapping['speed_in_traffic_min']
    df_color_mapping['speed_in_traffic_max'] = weight * df_color_mapping['speed_in_traffic_max']

    # Merge speed data with color mapping and filter based on speed median
    df_rolling_nonnull = df_rolling[df_rolling['speed_median'].notnull()]
    df_rolling_nonnull = df_rolling_nonnull.merge(df_color_mapping, how='left', on='type')
    df_rolling_nonnull = df_rolling_nonnull[
        (df_rolling_nonnull['speed_median']>=df_rolling_nonnull['speed_in_traffic_min']) & 
        (df_rolling_nonnull['speed_median']<df_rolling_nonnull['speed_in_traffic_max'])
    ]
    df_rolling_nonnull.drop(columns=['speed_in_traffic_min', 'speed_in_traffic_max'], inplace=True)

    # df_rolling_null = df_rolling[df_rolling['speed_median'].isnull()]
    # df_rolling_null['color'] = np.nan

    # df_rolling = pd.concat([df_rolling_nonnull, df_rolling_null]).reset_index(drop=True)

    # Explode the 'order' column to convert each element in order list to one row
    df_explode = df_rolling_nonnull[['street', 'direction', 'order', 'color']].explode('order').reset_index(drop=True)

    # Function to get the most presented color (with priority) from a series of color
    def get_mode_with_priority(series):
        priority_order = ['green', 'yellow', 'red', 'darkred']
        modes = series.mode().tolist()
        modes_priority = [x for x in priority_order if x in modes]
        return modes_priority[0]

    # Apply the function to get the most presented color with priority for each street point
    df_mode = df_explode.groupby(['street', 'direction', 'order'])['color'].apply(get_mode_with_priority).reset_index()
    # print(f'df_mode = {df_mode.shape}')

    # Merge the most presented colors back into the street data
    df_street_color = df_street_group.merge(df_mode, how='left', on=['street', 'direction', 'order'])
    df_street_color['color'] = df_street_color['color'].fillna('blue')
    # print(f'df_street_color = {df_street_color.shape}')
    
    return df_street_color


if __name__ == '__main__':
    # Set variables

    # DATE = datetime.now().strftime('%Y-%m-%d')
    # TIME = datetime.now().strftime('%H:%M:00')

    DATE = '2024-09-10'
    TIME = '17:03:00'

    # print(f'DATE = {DATE}')
    # print(f'TIME = {TIME}')

    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = f'{current_path}/../db/traffic.xlsx'
    FILE_PATH_VEHICLE = f'{current_path}/../db/f_gps_vehicle_20240912164421.csv'
    FILE_PATH_STREET = f'{current_path}/../db/d_street_old.csv'

    weight=0.5

    # Load vehicle data
    df_vehicle = read_data_vehicle(FILE_PATH_VEHICLE, DATE, TIME, duration=5)
    df_vehicle = preprocessing(df_vehicle)
    # print(df_vehicle.shape)

    # Load street data
    df_street = pd.read_csv(FILE_PATH_STREET)
    # df_street.shape

    #######################################
    df_street_type = pd.read_excel(file_path, sheet_name='street')
    df_street = df_street.merge(df_street_type, how='left', on='street')
    df_street = df_street[df_street['order'].notnull() & (df_street['order'] != 'x')].reset_index(drop=True)
    df_street['order'] = df_street['order'].astype(int)
    # df_street.shape
    #######################################

    # Combine vehicle data and street data, get traffic color
    df_street = insert_heading_col(df_street)
    df_vehicle = find_closest_point(df_vehicle, df_street, angle_threshold=45, chunk_size=5000)
    df_street_color = get_map_color(df_vehicle, df_street, window_size=10, weight=weight)

    # print(f'df_street = {df_street.shape}')
    # print(f'df_vehicle = {df_vehicle.shape}')
    # print(f'df_street_color = {df_street_color.shape}')

    # Export result
    dt = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = f'{current_path}/../db/output_{dt}.csv'
    df_street_color.to_csv(output_path, index=False)
