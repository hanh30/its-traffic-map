import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches





def get_df_speed(path):
    '''
    Read and process result from Distance Matrix API.
    '''
    df_list = []
    for f in glob.glob(f'{path}/*.csv'):
        df = pd.read_csv(f)
        df_list.append(df)
    df_speed = pd.concat(df_list).reset_index(drop=True)

    df_speed['datetime'] = pd.to_datetime(df_speed['datetime'])
    df_speed['datetime'] = df_speed['datetime'].dt.floor('s')
    
    # Get center coordinate between origin and destination,
    # to extract color for the center point in the next step
    df_speed['lat'] = (df_speed['origin_x'] + df_speed['destination_x'])/2
    df_speed['lng'] = (df_speed['origin_y'] + df_speed['destination_y'])/2
    return df_speed


def generate_center_list(nw, se, zoom=18):
    '''
    Generate a grid of points (with latitude and longitude) within
    a given bounding box (northwest and southeast corners)
    based on the specified zoom level.
    Results of this function are used as center points to get traffic color from Traffic Layer API.

    Parameters:
    nw (tuple): Coordinates of the northwest corner (latitude, longitude).
    se (tuple): Coordinates of the southeast corner (latitude, longitude).
    zoom (int): Zoom level to determine the grid spacing (default is 18).

    Returns:
    pd.DataFrame: A DataFrame containing the grid of center points 
    with 'lat' (latitude) and 'lng' (longitude) columns.
    '''

    # Read the 'zoom' sheet from the Excel file to get latitude and longitude delta values for the specified zoom level
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = f'{current_path}/../db/traffic.xlsx'
    df_zoom = pd.read_excel(file_path, sheet_name='zoom')

    # delta_lat: distance from center to horizontal edge
    # delta_lng: distance from center to vertical edge
    delta_lat = df_zoom[df_zoom['zoom']==zoom]['delta_lat'].values[0]
    delta_lng = df_zoom[df_zoom['zoom']==zoom]['delta_lng'].values[0]

    # Generate the centers' latitude list from north to south
    x_list = np.arange(nw[0], se[0], -delta_lat)
    x_list = x_list[[i for i in range(len(x_list)-1) if i%2==1]]

    if x_list[-1] - delta_lat > se[0]:
        x_list = np.append(x_list, x_list[-1] - 2*delta_lat)

    # Generate the centers' longitude list from west to east
    y_list = np.arange(nw[1], se[1], delta_lng)
    y_list = y_list[[i for i in range(len(y_list)-1) if i%2==1]]

    if y_list[-1] + delta_lng < se[1]:
        y_list = np.append(y_list, y_list[-1] + 2*delta_lng)

    # print(len(x_list))
    # print(len(y_list))

    # Generate all combinations (cross join) of latitude and longitude points
    df_grid_x = pd.DataFrame({'lat': x_list})
    df_grid_y = pd.DataFrame({'lng': y_list})
    df_grid = pd.merge(df_grid_x, df_grid_y, how='cross')
    # df_grid.shape

    return df_grid


def get_df_coord(path):
    '''
    Read and process json files from Traffic Layer API result,
    containing center, North East (NE) and South West (SW) coordinates.
    '''

    coord_list = []
    for f in glob.glob(f'{path}/*.json'):
        with open(f, 'r') as json_file:
            coord = json.load(json_file)
            coord['timestamp'] = f.split('\\')[-1][:10]
            coord_list.append(coord)

    def _transform_structure(data):
        transformed_data = []
        for element in data:
            transformed_element = {
                'lat': element['center']['lat'],
                'lng': element['center']['lng'],
                'ne_lat': element['ne']['lat'],
                'ne_lng': element['ne']['lng'],
                'sw_lat': element['sw']['lat'],
                'sw_lng': element['sw']['lng'],
                'timestamp': element['timestamp']
            }
            transformed_data.append(transformed_element)
        return transformed_data

    coord_list = _transform_structure(coord_list)

    df_coord = pd.DataFrame(coord_list)
    return df_coord


# def get_traffic_color(image_path, df_coord, edge=20, plot=False):
#     '''
#     Extracts the dominant traffic color from a cropped square of an image based on geographical coordinates.
    
#     Args:
#         image_path (str): The file path to the image.
#         df_coord (DataFrame): The DataFrame containing NE, SW and center coordinates of the image.
#         edge (int): The edge length of the square region to crop from the image.
#         plot (bool): Whether to display the cropped image and the marked region (for visualization).

#     Returns:
#         most_matching_color (str): The traffic color that matches the most pixels in the cropped area.
#     '''
#     # Load the image
#     image = Image.open(image_path)
#     W, H = image.size

#     timestamp = image_path.split('\\')[-1][:10]
#     coord = df_coord[df_coord['timestamp']==timestamp].iloc[0]

#     # Extract the coordinates
#     lat_nw = coord['ne_lat']
#     lon_nw = coord['sw_lng']
#     lat_se = coord['sw_lat']
#     lon_se = coord['ne_lng']
#     lat_c = coord['lat']
#     lon_c = coord['lng']

#     # Convert the center coordinates to pixel coordinates
#     x_c = int(W / (lon_se - lon_nw) * (lon_c - lon_nw))
#     y_c = int(H / (lat_nw - lat_se) * (lat_nw - lat_c))

#     # Determine the pixel coordinates of the square
#     half_edge = int(edge/2)
#     x_nw_square = max(0, x_c - half_edge)
#     y_nw_square = max(0, y_c - half_edge)
#     x_se_square = min(W, x_c + half_edge + 1)
#     y_se_square = min(H, y_c + half_edge + 1)

#     # Crop the square from the image
#     cropped_square = image.crop((x_nw_square, y_nw_square, x_se_square, y_se_square))
#     cropped_image = np.array(cropped_square)[:,:,:3]

#     # Define colors as numpy arrays
#     colors = {
#         'green': np.array([22, 224, 152]),
#         'yellow': np.array([255, 207, 67]),
#         'red': np.array([242, 78, 66]),
#         'darkred': np.array([169, 39, 39])
#     }

#     # Initialize a dictionary to count pixels matching each color
#     color_counts = {color_name: 0 for color_name in colors}

#     threshold = 10  # Threshold to find matching color for each pixel

#     # Loop through each pixel in the image
#     for color_name, color_value in colors.items():
#         # Compute the distance of each pixel to the color
#         distances = np.linalg.norm(cropped_image - color_value, axis=2)
#         # Count pixels within the threshold
#         count = np.sum(distances < threshold)
#         color_counts[color_name] = count

#     # Find the color with the maximum count
#     most_matching_color = max(color_counts, key=color_counts.get)

#     if plot is True:
#         # Plot the image
#         fig, ax = plt.subplots()
#         ax.imshow(image)

#         # Create a rectangle patch for the square
#         rect = patches.Rectangle(
#             (x_nw_square, y_nw_square),  # (x, y) of top-left corner of square
#             x_se_square - x_nw_square,  # Width of the square
#             y_se_square - y_nw_square,  # Height of the square
#             linewidth=1,
#             edgecolor='b',
#             facecolor='none'
#         )

#         # Add the rectangle patch to the plot
#         ax.add_patch(rect)

#         # Display the plot
#         plt.show()

#         # Plot the cropped image
#         plt.imshow(cropped_square)

#         # Display the plot
#         plt.show()

#     return most_matching_color

def find_coord_in_grid(coord, df_coord):
    '''
    Find the closest center point in a grid (df_coord) to a given coordinate (coord).

    Parameters:
    coord (tuple): The target coordinate as a tuple (latitude, longitude).
    df_coord (pd.DataFrame): A DataFrame containing grid cells, with north-east and 
                             south-west coordinates (columns 'ne_lat', 'ne_lng', 'sw_lat', 'sw_lng'),
                             and the center point columns 'lat' and 'lng'.

    Returns:
    pd.Series: The row corresponding to the closest center point to the given coordinate.
    '''

    # Filter to find the cells that contain the target coordinate
    df_coord_filter = df_coord[
        (coord[0] <= df_coord['ne_lat']) &
        (coord[0] >= df_coord['sw_lat']) &
        (coord[1] <= df_coord['ne_lng']) &
        (coord[1] >= df_coord['sw_lng'])
    ]

    # Calculate the distance between the target coordinate and the center points of each cell
    df_coord_filter['distance'] = np.sqrt((df_coord_filter['lat'] - coord[0]) ** 2 + (df_coord_filter['lng'] - coord[1]) ** 2)
    
    # Find the row with the minimum distance
    nearest_row = df_coord_filter.loc[df_coord_filter['distance'].idxmin()]

    return nearest_row


def get_traffic_color(coord, df_coord, traffic_api_result_path, edge=20, plot=False):
    '''
    Extracts the dominant traffic color from a cropped square of an image based on geographical coordinates.
    
    Args:
        coord (tuple): The geographical coordinates (latitude, longitude) of the point of interest.
        df_coord (DataFrame): DataFrame containing the north-east (NE), south-west (SW), and center coordinates of the images (result of Traffic Layer API).
        traffic_api_result_path (str): Path to the directory containing traffic images fetched from the API.
        edge (int, optional): The edge length of the square region to crop from the image, in pixels. Default is 20.
        plot (bool, optional): Whether to display the cropped image and the marked region for visualization.

    Returns:
        most_matching_color (str): The traffic color (e.g., 'green', 'yellow', 'red', 'darkred')
        that matches the most pixels in the cropped area.
    '''
    
    nearest_row = find_coord_in_grid(coord, df_coord)
    timestamp = nearest_row['timestamp']

    # Load the image
    image_path = f'{traffic_api_result_path}/{timestamp}_image.png'
    image = Image.open(image_path)
    W, H = image.size

    # Extract the coordinates of north-west, south-east, and point of interest
    lat_nw = nearest_row['ne_lat']
    lon_nw = nearest_row['sw_lng']
    lat_se = nearest_row['sw_lat']
    lon_se = nearest_row['ne_lng']
    lat_c = coord[0]
    lon_c = coord[1]

    # Convert the coordinate to pixel coordinate
    x_c = int(W / (lon_se - lon_nw) * (lon_c - lon_nw))
    y_c = int(H / (lat_nw - lat_se) * (lat_nw - lat_c))

    # Determine the pixel coordinate of the square
    half_edge = int(edge/2)
    x_nw_square = max(0, x_c - half_edge)
    y_nw_square = max(0, y_c - half_edge)
    x_se_square = min(W, x_c + half_edge + 1)
    y_se_square = min(H, y_c + half_edge + 1)

    # Crop the square from the image
    cropped_square = image.crop((x_nw_square, y_nw_square, x_se_square, y_se_square))
    cropped_image = np.array(cropped_square)[:,:,:3]

    # Define colors as numpy arrays
    colors = {
        'green': np.array([22, 224, 152]),
        'yellow': np.array([255, 207, 67]),
        'red': np.array([242, 78, 66]),
        'darkred': np.array([169, 39, 39])
    }

    # Initialize a dictionary to count pixels matching each color
    color_counts = {color_name: 0 for color_name in colors}

    threshold = 10  # Threshold to find matching color for each pixel

    # Loop through each pixel in the image
    for color_name, color_value in colors.items():
        # Compute the distance of each pixel to the color
        distances = np.linalg.norm(cropped_image - color_value, axis=2)
        # Count pixels within the threshold
        count = np.sum(distances < threshold)
        color_counts[color_name] = count

    # Find the color with the maximum count
    most_matching_color = max(color_counts, key=color_counts.get)

    if plot is True:
        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        # Create a rectangle patch for the square
        rect = patches.Rectangle(
            (x_nw_square, y_nw_square),  # (x, y) of top-left corner of square
            x_se_square - x_nw_square,  # Width of the square
            y_se_square - y_nw_square,  # Height of the square
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )

        # Add the rectangle patch to the plot
        ax.add_patch(rect)

        # Display the plot
        plt.show()
        
        # Plot the cropped image
        plt.imshow(cropped_square)

        # Display the plot
        plt.show()
        
    return most_matching_color


def get_df_color(df, df_coord, traffic_api_result_path):
    '''
    Get the traffic color for multiple points.

    Args:
        df (DataFrame): The input DataFrame containing coordinates of points of interest ('lat', 'lng' columns).
        df_coord (DataFrame): DataFrame containing the north-east (NE), south-west (SW), and center coordinates of the images (result of Traffic Layer API).
        traffic_api_result_path (str): Path to the directory containing traffic images fetched from the API.
        
    Returns:
        DataFrame: The input DataFrame with an additional 'color' column indicating the dominant traffic color for each point, 
                   and a 'datetime' column converting the 'timestamp' to a timezone-adjusted datetime format.
    '''
    df['color'] = df.apply(lambda row: get_traffic_color((row['lat'], row['lng']), df_coord, traffic_api_result_path, edge=20, plot=False), axis=1)
    df['datetime'] = pd.to_datetime(df['timestamp'].astype('int'), unit='s') + pd.Timedelta(hours=7)

    return df


def get_df_traffic(df_speed, df_color):
    '''
    Merge the results of Distance Matrix API and the results of Traffic Layer API.
    '''
    df_traffic = df_speed.merge(
        df_color[['lat', 'lng', 'datetime', 'color']].rename(columns={'datetime': 'datetime_color'}), 
        how='left', 
        on=['lat', 'lng']
    )

    # Filter the DataFrame to keep only rows where datetimes of two APIs are closest 
    # and the time difference is within the defined threshold.
    datetime_delta_threshold = 10 * 60
    df_traffic['datetime_delta'] = (df_traffic['datetime'] - df_traffic['datetime_color']).dt.total_seconds().abs()
    df_traffic['rank'] = df_traffic.groupby(['lat', 'lng', 'datetime'])['datetime_delta'].rank(method='min')
    df_traffic = df_traffic[(df_traffic['rank']==1) & (df_traffic['datetime_delta']<=datetime_delta_threshold)]

    return df_traffic


def traffic_map_eval(output_path, nw, se, df_coord, traffic_api_result_path, num_point_sample=10, random_state=42):
    '''
    Evaluates the accuracy of predicted traffic colors by comparing them to actual traffic colors 
    extracted from traffic images within a specified bounding box.

    Args:
        output_path (str): The file path to the CSV containing the traffic prediction results.
        nw (tuple): Coordinates of the northwest corner (latitude, longitude) of the bounding box.
        se (tuple): Coordinates of the southeast corner (latitude, longitude) of the bounding box.
        df_coord (DataFrame): A DataFrame containing the NE, SW, and center coordinates of the images (result of Traffic Layer API).
        traffic_api_result_path (str): Path to the directory containing traffic images fetched from the API.
        num_point_sample (int, optional): The maximum number of sample points to take per street and direction group. Default is 10.
        random_state (int, optional): Random seed for sampling to ensure reproducibility. Default is 42.

    Returns:
        DataFrame: A DataFrame containing precision and recall for each traffic color ('green', 'yellow', 'red', 'darkred').
        Last row is total accuracy.
    '''

    # Load the predicted traffic data from the CSV file
    df_output = pd.read_csv(output_path)
    # print(df_output.shape)

    # Filter the dataset to include points within the bounding box and exclude points with a 'blue' color
    df_output_filter = df_output[
        (df_output['x']>=se[0]) & (df_output['x']<=nw[0]) &
        (df_output['y']>=nw[1]) & (df_output['y']<=se[1]) &
        (df_output['color'] != 'blue')
    ][['street', 'x', 'y', 'direction', 'order', 'color']]
    # print(df_output_filter.shape)

    # Sample a maximum of num_point_sample points per street and direction group
    df_output_sample = df_output_filter.groupby(['street' ,'direction']).apply(lambda x: x.sample(min(len(x), num_point_sample), random_state=random_state)).reset_index(drop=True)
    # print(df_output_sample.shape)

    # For each sample point, determine the actual traffic color
    df_output_sample['color_actual'] = df_output_sample.apply(lambda row: get_traffic_color((row['x'], row['y']), df_coord, traffic_api_result_path, edge=20, plot=False), axis=1)

    # Evaluate
    df_predict = df_output_sample['color'].value_counts().reset_index().rename(columns={'count': 'predict'})
    df_actual = df_output_sample['color_actual'].value_counts().reset_index().rename(columns={'count': 'actual', 'color_actual': 'color'})
    df_tp = df_output_sample[df_output_sample['color']==df_output_sample['color_actual']]['color'].value_counts().reset_index().rename(columns={'count': 'tp'})

    df_eval = df_predict.merge(df_actual, how='left', on='color').merge(df_tp, how='left', on='color')
    df_eval.loc[len(df_eval)] = ('total', df_eval['predict'].sum(), df_eval['actual'].sum(), df_eval['tp'].sum())
    
    order = ['green', 'yellow', 'red', 'darkred', 'total']
    df_eval['color'] = pd.Categorical(df_eval['color'], categories=order, ordered=True)
    df_eval = df_eval.sort_values(by='color').reset_index(drop=True)

    df_eval['precision'] = df_eval['tp']/df_eval['predict']*100
    df_eval['recall'] = df_eval['tp']/df_eval['actual']*100

    return df_eval
