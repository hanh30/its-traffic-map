import os
import pandas as pd
import json

current_path = os.path.dirname(os.path.abspath(__file__))
file_path = f'{current_path}/../db/traffic.xlsx'
output_path = f'{current_path}/location.json'

def get_location(file_path):
    df = pd.read_excel(file_path, header=2)[['district', 'type', 'street', 'origin_x', 'origin_y', 'destination_x', 'destination_y']]
    df = df[df['type']!='test'].reset_index(drop=True)

    df['lat'] = (df['origin_x'] + df['destination_x'])/2
    df['lng'] = (df['origin_y'] + df['destination_y'])/2

    location = df[['lat', 'lng']].to_dict('records')

    return location

if __name__ == '__main__':
    location = get_location(file_path)
    with open(output_path, 'w') as json_file:
        json.dump(location, json_file, indent=4)
