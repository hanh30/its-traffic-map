import os
import pandas as pd
import requests
from datetime import datetime


current_path = os.path.dirname(os.path.abspath(__file__))
file_path = f'{current_path}/../db/traffic.xlsx'
output_path = f'{current_path}/../api_result/distance_matrix_api'
api_key = "AIzaSyDs-H0aYalgZR-63_qrw5qNyj3LccQG5Ws"


def read_data(file_path):
    # Read data from path
    df = pd.read_excel(file_path, header=2, sheet_name='data_api')[['district', 'type', 'street', 'origin_x', 'origin_y', 'destination_x', 'destination_y']]

    # Filter to remove test row
    df = df[df['type']!='test'].reset_index(drop=True)

    # Create new column with api link 
    df['link'] = (
        "https://maps.googleapis.com/maps/api/distancematrix/json?departure_time=now&destinations="
        + df['destination_x'].astype(str)
        + "%2C"
        + df['destination_y'].astype(str)
        + "&origins="
        + df['origin_x'].astype(str)
        + "%2C"
        + df['origin_y'].astype(str)
        + "&key=" + api_key
    )

    return df


def request_api(df):
    # Initialize a session for connection reuse
    session = requests.Session()

    output_list = []
    for i in range(len(df)):
        dt = datetime.now()

        resp = session.get(url=df['link'][i])
        data = resp.json() # Check the JSON Response Content documentation below

        data['origin_addresses'] = data['origin_addresses'][0]
        data['destination_addresses'] = data['destination_addresses'][0]
        data['distance'] = data['rows'][0]['elements'][0]['distance']['value']
        data['duration'] = data['rows'][0]['elements'][0]['duration']['value']
        data['duration_in_traffic'] = data['rows'][0]['elements'][0]['duration_in_traffic']['value']
        data['datetime'] = dt

        data = {key: value for key, value in data.items() if key not in ['rows', 'status']}
        
        output_list.append(data)

    df_output = pd.DataFrame(output_list)
    df_api = pd.concat([df, df_output], axis=1)

    df_api['speed'] = df_api['distance']/1000 / (df_api['duration']/3600)
    df_api['speed_in_traffic'] = df_api['distance']/1000 / (df_api['duration_in_traffic']/3600)

    return df_api



if __name__ == '__main__':
    dt = datetime.now().strftime('%Y%m%d%H%M%S')
    df = read_data(file_path)
    df_api = request_api(df)
    df_api.to_csv(f'{output_path}/{dt}.csv', index=False)
    