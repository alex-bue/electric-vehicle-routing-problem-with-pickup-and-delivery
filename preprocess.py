import pandas as pd
from datetime import datetime, time as dt_time, timedelta
import json
from tqdm import tqdm
import argparse
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut


### Geocoding Functions ###

def do_geocode(geolocator, lat, lon, attempt=1, max_attempts=5, timeout=10):
    try:
        return geolocator.reverse((lat, lon), language='en', timeout=timeout)
    except GeocoderTimedOut:
        if attempt <= max_attempts:
            print(f"GeocoderTimedOut: Retrying ({attempt}/{max_attempts})...")
            time.sleep(attempt)  # Exponential backoff
            return do_geocode(geolocator, lat, lon, attempt=attempt + 1, max_attempts=max_attempts, timeout=timeout)
        raise


def geocode_coordinates(geolocator, coordinates, max_retries=5, delay_between_retries=2, timeout=10):
    def get_country(lat, lon):
        retries = 0
        while retries < max_retries:
            try:
                location = do_geocode(geolocator, lat, lon, attempt=retries + 1, max_attempts=max_retries,
                                      timeout=timeout)
                if location:
                    return location.raw.get('address', {}).get('country_code', '').upper()
                return None
            except Exception as e:
                print(f"Error geocoding {lat}, {lon}: {str(e)}. Retrying...")
                retries += 1
                time.sleep(delay_between_retries)
        return None

    geocoded_results = {}
    for (lat, lon) in tqdm(coordinates, desc="Geocoding"):
        geocoded_results[f"{lat},{lon}"] = get_country(lat, lon)
    return geocoded_results


def save_geocoded_results(geocoded_results, filepath):
    with open(filepath, 'w') as file:
        json.dump(geocoded_results, file)


def load_geocoded_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    return None


### Data Processing Functions ###

def add_duration_distance(velocity_df, input_unique_distance, input_unique_duration):
    with open(input_unique_distance, 'r') as f:
        unique_distance_matrix = json.load(f)
    with open(input_unique_duration, 'r') as f:
        unique_duration_matrix = json.load(f)

    # Convert distance to km and duration to hours
    unique_distance_matrix = [[dist / 1000 for dist in row] for row in unique_distance_matrix]
    unique_duration_matrix = [[dur / 3600 for dur in row] for row in unique_duration_matrix]

    velocity_df['Duration'] = None
    velocity_df['Distance'] = None
    velocity_df['Distance_from_depot'] = None
    velocity_df['Distance_to_depot'] = None
    for i in range(len(velocity_df)):
        from_index = velocity_df.loc[i, 'FromIndex']
        to_index = velocity_df.loc[i, 'ToIndex']
        duration = unique_duration_matrix[from_index][to_index]
        distance = unique_distance_matrix[from_index][to_index]
        velocity_df.loc[i, "Duration"] = duration
        velocity_df.loc[i, "Distance"] = distance
        velocity_df.loc[i, "Distance_from_depot"] = unique_distance_matrix[from_index][0]
        velocity_df.loc[i, "Distance_to_depot"] = unique_distance_matrix[0][to_index]

    return velocity_df


def aggregate_top_days_bookings(velocity_df, top_n=50):
    # Aggregate the number of bookings per day
    bookings_per_day = velocity_df.groupby('StartRequestedDate').size().reset_index(name='NumberOfBookings')
    sorted_days = bookings_per_day.sort_values(by='NumberOfBookings', ascending=False).head(top_n)

    top_days = sorted_days['StartRequestedDate']
    top_days_bookings = velocity_df[velocity_df['StartRequestedDate'].isin(top_days)].reset_index()

    print(f'Number of bookings: {len(top_days_bookings)}')

    return top_days_bookings


def split_weights_into_multiple_rows(df, weight_column='Weight', max_capacity=33000):
    rows = []

    for index, row in df.iterrows():
        weight = row[weight_column]
        num_full_trucks = weight // max_capacity
        remainder_weight = weight % max_capacity

        for _ in range(int(num_full_trucks)):
            new_row = row.copy()
            new_row[weight_column] = max_capacity
            rows.append(new_row)

        if remainder_weight > 0:
            new_row = row.copy()
            new_row[weight_column] = remainder_weight
            rows.append(new_row)

    return pd.DataFrame(rows)


def extract_unique_locations_from_velocity(velocity_df):
    unique_from_locations = velocity_df[['FromLatitude', 'FromLongitude']].drop_duplicates()
    unique_to_locations = velocity_df[['ToLatitude', 'ToLongitude']].drop_duplicates()
    unique_from_locations.columns = ['Latitude', 'Longitude']
    unique_to_locations.columns = ['Latitude', 'Longitude']
    unique_locations = pd.concat([unique_from_locations, unique_to_locations]).drop_duplicates().reset_index(drop=True)
    unique_locations['Source'] = 'velocity'

    return unique_locations


def filter_stations_by_proximity(station_df, customer_df, distance_threshold):
    # Calculate distance between each station and the closest customer
    def min_distance_to_customers(station, customers):
        distances = customers.apply(lambda customer: great_circle((station['Latitude'], station['Longitude']),
                                                                  (customer['Latitude'], customer['Longitude'])).meters,
                                    axis=1)
        return distances.min()

    # Add the MinDistanceToCustomer column
    station_df['MinDistanceToCustomer'] = station_df.apply(
        lambda station: min_distance_to_customers(station, customer_df), axis=1)

    # Filter out stations beyond the distance threshold from any customer
    filtered_stations = station_df[station_df['MinDistanceToCustomer'] <= distance_threshold]

    # Cleanup
    filtered_stations = filtered_stations.drop(columns=['MinDistanceToCustomer'])

    print(f"Number of stations after filtering by distance: {len(filtered_stations)}")

    return filtered_stations


def cluster_and_reduce_stations(filtered_stations, customer_coordinates, eps=0.05, min_samples=2,
                                fraction_to_retain=0.5):
    original_columns = filtered_stations.columns.tolist()

    if isinstance(customer_coordinates, pd.DataFrame):
        customer_coordinates = customer_coordinates[['Latitude', 'Longitude']].values.tolist()

    # Combine customer and filtered station coordinates
    combined_coordinates = customer_coordinates + filtered_stations[['Latitude', 'Longitude']].values.tolist()
    combined_labels = ['customer'] * len(customer_coordinates) + ['station'] * len(filtered_stations)

    # Convert combined coordinates to DataFrame
    combined_df = pd.DataFrame(combined_coordinates, columns=['Latitude', 'Longitude'])
    combined_df['Type'] = combined_labels

    # Normalize the combined data
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(combined_df[['Latitude', 'Longitude']])

    # Apply DBSCAN with specified parameters
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    combined_df['Cluster'] = db.labels_

    # Filter out the noise points (label -1) and customer points
    clustered_stations = combined_df[(combined_df['Cluster'] != -1) & (combined_df['Type'] == 'station')]

    # Proportionally reduce points by retaining a fraction of each cluster based on its size
    def proportional_reduction(df, fraction):
        cluster_sizes = df['Cluster'].value_counts()
        reduced_df = pd.DataFrame(columns=df.columns)

        for cluster, size in cluster_sizes.items():
            cluster_points = df[df['Cluster'] == cluster]
            num_points_to_retain = max(1, int(size * fraction))
            retained_points = cluster_points.sample(n=num_points_to_retain, random_state=42)
            reduced_df = pd.concat([reduced_df, retained_points], ignore_index=True)

        return reduced_df

    reduced_clustered_stations = proportional_reduction(clustered_stations, fraction_to_retain)
    print(f"Number of stations after clustering: {len(reduced_clustered_stations)}")

    # Filter the reduced stations back to the original columns and return
    reduced_original_stations = filtered_stations.merge(
        reduced_clustered_stations[['Latitude', 'Longitude']],
        on=['Latitude', 'Longitude'],
        how='inner'
    )

    return reduced_original_stations[original_columns]


def clean_time_windows(df, unit='half_hours', working_start='08:00:00', working_end='18:00:00', buffer_minutes=30):
    # Combine date and time columns into datetime with improved naming convention
    df['PickupStartDatetime'] = pd.to_datetime(df['StartRequestedDate'] + ' ' + df['StartRequestedFromTime'],
                                               dayfirst=True, errors='coerce')
    df['PickupEndDatetime'] = pd.to_datetime(df['StartRequestedDate'] + ' ' + df['StartRequestedToTime'], dayfirst=True,
                                             errors='coerce')
    df['DeliveryStartDatetime'] = pd.to_datetime(df['StartRequestedDate'] + ' ' + df['EndRequestedFromTime'],
                                                 dayfirst=True, errors='coerce')
    df['DeliveryEndDatetime'] = pd.to_datetime(df['StartRequestedDate'] + ' ' + df['EndRequestedToTime'], dayfirst=True,
                                               errors='coerce')

    # Define working hours and end of the day time
    working_start = datetime.strptime(working_start, '%H:%M:%S').time()
    working_end = datetime.strptime(working_end, '%H:%M:%S').time()
    end_of_day = datetime.strptime('23:59:59', '%H:%M:%S').time()  # Ensure end_of_day is of type dt_time

    # Define conversion functions for different units
    def to_half_hour_units(t):
        return (t.hour * 60 + t.minute) // 30

    def to_minute_units(t):
        return t.hour * 60 + t.minute

    def to_second_units(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    def from_half_hour_units(u):
        return dt_time((u * 30) // 60, (u * 30) % 60)

    def from_minute_units(u):
        return dt_time(u // 60, u % 60)

    def from_second_units(u):
        return dt_time(u // 3600, (u % 3600) // 60, u % 60)

    # Set the conversion functions based on the unit
    if unit == 'half_hours':
        to_units = to_half_hour_units
        from_units = from_half_hour_units
        buffer_units = buffer_minutes // 30
        units_per_day = 48
    elif unit == 'minutes':
        to_units = to_minute_units
        from_units = from_minute_units
        buffer_units = buffer_minutes
        units_per_day = 1440
    elif unit == 'seconds':
        to_units = to_second_units
        from_units = from_second_units
        buffer_units = buffer_minutes * 60
        units_per_day = 86400
    else:
        raise ValueError("Invalid unit. Choose from 'half_hours', 'minutes', or 'seconds'.")

    # Function to adjust time windows
    def adjust_time_window(start_unit, end_unit):
        # start and end NA -> assume working hours
        if start_unit is None and end_unit is None:
            start_unit = to_units(working_start)
            end_unit = to_units(working_end)
        # end time NA -> assume working hours for end. If results in start > end time set to EOD
        elif end_unit is None:
            end_unit = to_units(working_end)
            if start_unit > end_unit:
                end_unit = units_per_day - 1
        # same but other way around
        elif start_unit is None:
            start_unit = to_units(working_start)
            if start_unit > end_unit:
                start_unit = 0

        # insert buffer to give some slack if start and end times are the same
        if start_unit == end_unit:
            if end_unit == units_per_day - 1:
                start_unit = max(0, start_unit - buffer_units)
            else:
                end_unit = min(units_per_day - 1, end_unit + buffer_units)
        if start_unit > end_unit:
            end_unit = units_per_day - 1

        return start_unit, end_unit

    # Adjust time windows for pickup and delivery
    for index, row in df.iterrows():
        pickup_start_unit = to_units(row['PickupStartDatetime'].time()) if not pd.isnull(
            row['PickupStartDatetime']) else None
        pickup_end_unit = to_units(row['PickupEndDatetime'].time()) if not pd.isnull(row['PickupEndDatetime']) else None
        adjusted_pickup_start_unit, adjusted_pickup_end_unit = adjust_time_window(pickup_start_unit, pickup_end_unit)
        df.at[index, 'PickupStartDatetime'] = pd.to_datetime(
            row['StartRequestedDate'] + ' ' + from_units(adjusted_pickup_start_unit).strftime("%H:%M:%S"),
            dayfirst=True)
        df.at[index, 'PickupEndDatetime'] = pd.to_datetime(
            row['StartRequestedDate'] + ' ' + from_units(adjusted_pickup_end_unit).strftime("%H:%M:%S"), dayfirst=True)

        delivery_start_unit = to_units(row['DeliveryStartDatetime'].time()) if not pd.isnull(
            row['DeliveryStartDatetime']) else None
        delivery_end_unit = to_units(row['DeliveryEndDatetime'].time()) if not pd.isnull(
            row['DeliveryEndDatetime']) else None
        adjusted_delivery_start_unit, adjusted_delivery_end_unit = adjust_time_window(delivery_start_unit,
                                                                                      delivery_end_unit)
        df.at[index, 'DeliveryStartDatetime'] = pd.to_datetime(
            row['StartRequestedDate'] + ' ' + from_units(adjusted_delivery_start_unit).strftime("%H:%M:%S"),
            dayfirst=True)
        df.at[index, 'DeliveryEndDatetime'] = pd.to_datetime(
            row['StartRequestedDate'] + ' ' + from_units(adjusted_delivery_end_unit).strftime("%H:%M:%S"), dayfirst=True)

        # Additional logic to handle DeliveryEndDatetime - PickupStartDatetime > df['Duration']
        duration_timedelta = timedelta(hours=row['Duration']) + timedelta(hours=2)
        work_start_time = datetime.combine(df.at[index, 'PickupStartDatetime'].date(), working_start)
        work_end_time = datetime.combine(df.at[index, 'DeliveryEndDatetime'].date(), working_end)

        current_duration = df.at[index, 'DeliveryEndDatetime'] - df.at[index, 'PickupStartDatetime']

        if current_duration < duration_timedelta:
            # Try adjusting DeliveryEndDatetime to 7 PM
            if work_end_time > df.at[index, 'DeliveryStartDatetime']:
                df.at[index, 'DeliveryEndDatetime'] = work_end_time
                current_duration = df.at[index, 'DeliveryEndDatetime'] - df.at[index, 'PickupStartDatetime']
        else:
            continue

        # If condition still holds, try adjusting PickupStartDatetime to 7 AM
        if current_duration < duration_timedelta:
            if work_start_time < df.at[index, 'PickupEndDatetime']:
                df.at[index, 'PickupStartDatetime'] = work_start_time
                current_duration = df.at[index, 'DeliveryEndDatetime'] - df.at[index, 'PickupStartDatetime']
        else:
            continue

        # If condition still holds, set the time window to all day
        if current_duration < duration_timedelta:
            df.at[index, 'PickupStartDatetime'] = datetime.combine(df.at[index, 'PickupStartDatetime'].date(),
                                                                   dt_time(0, 0))
            df.at[index, 'DeliveryEndDatetime'] = datetime.combine(df.at[index, 'DeliveryEndDatetime'].date(),
                                                                       dt_time(23, 59, 59))

    # Convert time windows to specified units from the start of the day
    df['PickupStartTime'] = df['PickupStartDatetime'].apply(lambda x: to_units(x.time()))
    df['PickupEndTime'] = df['PickupEndDatetime'].apply(lambda x: to_units(x.time()))
    df['DeliveryStartTime'] = df['DeliveryStartDatetime'].apply(lambda x: to_units(x.time()))
    df['DeliveryEndTime'] = df['DeliveryEndDatetime'].apply(lambda x: to_units(x.time()))

    return df


def map_locations_to_indices(df, combined_df):
    location_index = {tuple(v): k for k, v in combined_df[['Latitude', 'Longitude']].iterrows()}
    df['FromIndex'] = df.apply(lambda row: location_index[(row['FromLatitude'], row['FromLongitude'])], axis=1)
    df['ToIndex'] = df.apply(lambda row: location_index[(row['ToLatitude'], row['ToLongitude'])], axis=1)
    return df


def map_stations_to_indices(stations_df, combined_df):
    location_index = {tuple(v): k for k, v in combined_df[['Latitude', 'Longitude']].iterrows()}
    stations_df['Index'] = stations_df.apply(lambda row: location_index[(row['Latitude'], row['Longitude'])], axis=1)
    return stations_df


### Main Function ###

def main():
    parser = argparse.ArgumentParser(description='Preprocess velocity data.')
    parser.add_argument('--input_unique_distance', type=str, default='evrp/data/distance_matrix.json')
    parser.add_argument('--input_unique_duration', type=str, default='evrp/data/duration_matrix.json')
    parser.add_argument('--input_velocity', type=str, default='data/velocity_raw.csv',
                        help='Path to the raw velocity data file.')
    parser.add_argument('--input_stations', type=str, default='data/stations_raw.csv',
                        help='Path to the raw stations data file.')
    parser.add_argument('--output_geocode_cache', type=str, default='data/geocoded_results.json',
                        help='Path to save geocode cache.')
    parser.add_argument('--output_unique_locations', type=str, default='evrp/data/unique_locations.csv',
                        help='Path to save unique locations.')
    parser.add_argument('--output_bookings_mapping', type=str, default='evrp/data/bookings_mapping_all_dates.csv',
                        help='Path to save bookings mapping.')
    parser.add_argument('--output_stations_mapping', type=str, default='evrp/data/stations_mapping.csv',
                        help='Path to save stations mapping.')
    parser.add_argument('--output_instances', type=str, default='evrp/data/instances/', help='Directory to save instance files.')

    args = parser.parse_args()

    # Load and preprocess velocity data
    print(f"Loading raw velocity data from {args.input_velocity}...")
    velocity_df = pd.read_csv(args.input_velocity)
    print(f"Initial dataframe size: {velocity_df.shape}")
    # velocity_df.drop(columns=['Name', 'ActualDistanceInKilometers'], inplace=True)
    velocity_df = velocity_df.rename(
        columns={'Tolatitude': 'ToLatitude', 'Tolongitude': 'ToLongitude', 'Shortname': 'ShortName'})
    velocity_df = velocity_df[velocity_df['ShortName'] == 'GOT']
    print(f"Size after filtering for Gothenburg: {velocity_df.shape}")

    # Extract unique coordinates from velocity
    velocity_coords = extract_unique_locations_from_velocity(velocity_df)
    # Load geocoded results if they exist
    geocoded_results = load_geocoded_results(args.output_geocode_cache)
    if geocoded_results is None:
        geolocator = Nominatim(user_agent="thesis-analysis")
        geocoded_results = geocode_coordinates(geolocator, velocity_coords[['Latitude', 'Longitude']].values.tolist())
        save_geocoded_results(geocoded_results, args.output_geocode_cache)
    else:
        print("Loaded cached geocoded results.")

    # Add geocoded information to velocity data
    velocity_df['FromCountry'] = velocity_df.apply(
        lambda row: geocoded_results.get(f"{row['FromLatitude']},{row['FromLongitude']}", None), axis=1)
    velocity_df['ToCountry'] = velocity_df.apply(
        lambda row: geocoded_results.get(f"{row['ToLatitude']},{row['ToLongitude']}", None), axis=1)

    # Filter data for Sweden
    velocity_df = velocity_df[(velocity_df['FromCountry'] == 'SE') & (velocity_df['ToCountry'] == 'SE')]
    print(f"Size after filtering for Sweden: {velocity_df.shape}")

    # Aggregate by top_n days with most bookings and only keep those
    # n_days = 50
    # velocity_df = aggregate_top_days_bookings(velocity_df, top_n=n_days)
    # print(f"Number of bookings after filtering for top {n_days} days of bookings: {velocity_df.shape}")
    # velocity_df = split_weights_into_multiple_rows(velocity_df, weight_column='GrossWeight', max_capacity=33000)
    velocity_df = velocity_df.reset_index(drop=True)
    # # FROM HERE ON WE START PREPARING THE DEPOT, UNIQUE CUSTOMERS AND UNIQUE STATIONS DATAFRAME
    # # Define depot node to be inserted at the beginning
    depot_coords = pd.DataFrame({'Latitude': [57.70105], 'Longitude': [11.81312], 'Source': ['depot']})

    # Extract unique coordinates again since now all transformations are done
    velocity_coords = extract_unique_locations_from_velocity(velocity_df)

    # Load and reduce number of stations (no need to grab unique because they are all unique already)
    print(f"Loading stations data from {args.input_stations}...")
    stations_df = pd.read_csv(args.input_stations)
    print(f"Number of stations loaded: {len(stations_df)}")
    stations_df = stations_df[stations_df['ConnectionType'] == 'CCS (Type 2)']
    print(f"Number of stations after filtering for CCS (Type 2) connection types: {len(stations_df)}")

    stations_df = filter_stations_by_proximity(stations_df, velocity_coords, distance_threshold=50000)
    stations_df = cluster_and_reduce_stations(stations_df, velocity_coords, eps=0.05, min_samples=2,
                                              fraction_to_retain=0.5)
    stations_df['Source'] = 'station'

    # Combine depot, velocity coords and station coords in order
    depot_customers_stations_df = pd.concat(
        [depot_coords, velocity_coords,
         stations_df[['Latitude', 'Longitude', 'Source']]]).drop_duplicates().reset_index(
        drop=True)

    # depot_customers_stations_df.to_csv(args.output_unique_locations, index_label='Index')

    # Map bookings and stations to unique depot, customers, stations indices
    bookings_mapping = map_locations_to_indices(velocity_df, depot_customers_stations_df)
    stations_mapping = map_stations_to_indices(stations_df, depot_customers_stations_df)

    # Add Actual duration and distance between customers
    bookings_mapping = add_duration_distance(bookings_mapping, args.input_unique_distance, args.input_unique_duration)
    # Clean and prepare the time windows to be right units and handle NA's
    bookings_mapping = clean_time_windows(bookings_mapping, unit='half_hours', working_start='08:00:00', working_end='18:00:00')

    # Save them
    bookings_mapping.to_csv(args.output_bookings_mapping, index=False)
    # stations_mapping.to_csv(args.output_stations_mapping, index=False)

    # Create directory if it doesn't exist
    if not os.path.exists(args.output_instances):
        os.makedirs(args.output_instances)
    #
    # # Split the bookings data by day using 'PickUpStartDatetime'
    # daily_data = {date: day_df for date, day_df in
    #               bookings_mapping.groupby(bookings_mapping['PickupStartDatetime'].dt.date)}
    #
    # for date, day_df in daily_data.items():
    #     formatted_date = date.strftime("%Y-%m-%d")
    #     filename = os.path.join(args.output_instances, f"instance_{formatted_date}.csv")
    #
    #     # Ensure parent directory exists
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #
    #     day_df.to_csv(filename, index=False)
    #     print(f"Saved instance for {formatted_date} to {filename}")


if __name__ == "__main__":
    main()
