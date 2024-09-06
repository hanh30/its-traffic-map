import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches



current_path = os.path.dirname(os.path.abspath(__file__))
api_result_path = f'{current_path}/../api_result'

def get_df_speed():
    df_list = []
    for f in glob.glob(f'{api_result_path}/distance_matrix_api/*.csv'):
        df = pd.read_csv(f)
        df_list.append(df)
    df_speed = pd.concat(df_list).reset_index(drop=True)

    df_speed['datetime'] = pd.to_datetime(df_speed['datetime'])
    df_speed['datetime'] = df_speed['datetime'].dt.floor('S')
    
    df_speed['lat'] = (df_speed['origin_x'] + df_speed['destination_x'])/2
    df_speed['lng'] = (df_speed['origin_y'] + df_speed['destination_y'])/2
    return df_speed


def get_df_coord():
    coord_list = []
    for f in glob.glob(f'{api_result_path}/traffic_api/*.json'):
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


def get_traffic_color(image_path, df_coord, edge=15, plot=False):
    # Load the image
    image = Image.open(image_path)
    W, H = image.size

    timestamp = image_path.split('\\')[-1][:10]
    coord = df_coord[df_coord['timestamp']==timestamp].iloc[0]

    # Extract the coordinates
    lat_nw = coord['ne_lat']
    lon_nw = coord['sw_lng']
    lat_se = coord['sw_lat']
    lon_se = coord['ne_lng']
    lat_c = coord['lat']
    lon_c = coord['lng']

    # Calculate the pixels per degree in latitude and longitude
    pixels_per_lat = H / (lat_nw - lat_se)
    pixels_per_lon = W / (lon_se - lon_nw)

    # Convert the center coordinates to pixel coordinates
    x_c = int((lon_c - lon_nw) * pixels_per_lon)
    y_c = int((lat_nw - lat_c) * pixels_per_lat)

    # Determine the pixel coordinates of the square (5x5 pixel square)
    half_edge = int(edge/2)  # Half-edge of 5 pixels (5/2 = 2.5, but we use 2 for integer pixel coords)
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

    threshold = 10  # Distance threshold

    # Initialize a dictionary to count pixels matching each color
    color_counts = {color_name: 0 for color_name in colors}

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
        # Plot the cropped image
        plt.imshow(cropped_square)

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

    return most_matching_color

def get_df_color(df_coord):
    timestamp_list = []
    color_list = []

    for f in glob.glob(f'{api_result_path}/traffic_api/*.png'):
        timestamp_list.append(f.split('\\')[-1][:10])
        color_list.append(get_traffic_color(f, df_coord, edge=20, plot=False))

    df_traffic_color = pd.DataFrame({
        'timestamp': timestamp_list,
        'color': color_list
    })

    df_color = df_coord.merge(df_traffic_color, how='left', on='timestamp')
    df_color['datetime'] = pd.to_datetime(df_color['timestamp'].astype('int'), unit='s') + pd.Timedelta(hours=7)

    return df_color
