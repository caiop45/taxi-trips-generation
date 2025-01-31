import pandas as pd
import geopandas as gpd
import numpy as np
import time

# 1. Data Preparation

parquet_files = [
    '/home-ext/caioloss/data/yellow_tripdata_2024-01.parquet'
]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

taxi_zones = gpd.read_file('/home-ext/caioloss/data/taxi-zones')
taxi_zones = taxi_zones.drop("geometry", axis=1)

# Create a dictionary to map LocationID to zone
location_zone_dict = dict(zip(taxi_zones['LocationID'], taxi_zones['zone']))

# Replace Location IDs with zone names
df['pickup_location'] = df['PULocationID'].map(location_zone_dict)
df['dropoff_location'] = df['DOLocationID'].map(location_zone_dict)

# Create the 'hour' column by rounding the pickup time
df['hour'] = df['tpep_pickup_datetime'].dt.round('h').dt.hour

# Select only the needed columns
trips_per_hour = df[['hour', 'pickup_location', 'dropoff_location']]

# 5 peak hours with the highest number of trips (before filtering and removing duplicates)
peak_hours_values = (
    trips_per_hour
    .groupby('hour')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='hour_count')[:5]
)
peak_hours = peak_hours_values['hour']
print("Peak hours before adjustment:")
print(peak_hours_values)

# Filter the dataframe to include only the trips within the peak hours and drop duplicates
trips_per_hour_peak = trips_per_hour[trips_per_hour['hour'].isin(peak_hours)].drop_duplicates()

# Save the real peak data to a parquet file
trips_per_hour_peak.to_parquet("real_data_peak.parquet")

# Recalculate the peak hours based on the filtered data
peak_hours_values = (
    trips_per_hour_peak
    .groupby('hour')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='hour_count')
)
peak_hours = peak_hours_values['hour']
print("\nPeak hours after adjustment:")
print(peak_hours_values)

# 2. Calculation of Pickup and Dropoff Probabilities

# Calculate the total number of pickups per hour
total_count_per_hour = (
    trips_per_hour_peak
    .groupby('hour')
    .size()
    .reset_index(name='total_hour_count')
)

# Calculate the total number of pickups per hour and pickup_location
total_count_per_hour_pickup = (
    trips_per_hour_peak
    .groupby(['hour', 'pickup_location'])
    .size()
    .reset_index(name='total_hour_count_pickup')
)

# Calculate counts for each combination of hour, pickup_location, and dropoff_location
counts = (
    trips_per_hour_peak
    .groupby(['hour', 'pickup_location', 'dropoff_location'])
    .size()
    .reset_index(name='count')
)

# Calculate the pickup probability based on hour
prob_df_pickup = total_count_per_hour.merge(total_count_per_hour_pickup, on='hour')
prob_df_pickup['pickup_probability'] = (
    prob_df_pickup['total_hour_count_pickup'] / prob_df_pickup['total_hour_count']
)

# Calculate the dropoff probability based on hour and pickup_location
prob_df_dropoff = counts.merge(total_count_per_hour, on=['hour'])
prob_df_dropoff['dropoff_probability'] = (
    prob_df_dropoff['count'] / prob_df_dropoff['total_hour_count']
)

# Pivot to create the pickup probability matrix
pickup_probability = prob_df_pickup.pivot_table(
    index='hour',
    columns='pickup_location',
    values='pickup_probability'
).fillna(0)

# Re-normalize to ensure probabilities sum to 1 per hour
pickup_probability = pickup_probability.div(pickup_probability.sum(axis=1), axis=0).fillna(0)

# Get all unique dropoff locations
all_dropoff_locations = taxi_zones['zone'].unique()

# Pivot to create the dropoff probability matrix
dropoff_probability = prob_df_dropoff.pivot_table(
    index=['hour', 'pickup_location'],
    columns='dropoff_location',
    values='dropoff_probability'
).fillna(0)

# Reindex to include all dropoff locations
dropoff_probability = dropoff_probability.reindex(columns=all_dropoff_locations, fill_value=0)

# 3. Optimized Pre-calculation of Probabilities

def pre_calculate_probabilities(prob_pickup, prob_dropoff, dropoff_zones):
    """
    Pre-calculate and store pickup and dropoff probabilities for all combinations of hour and zones.
    
    Args:
        prob_pickup (DataFrame): Pickup probability DataFrame (indexed by hour, columns are pickup zones).
        prob_dropoff (DataFrame): Dropoff probability DataFrame (indexed by [hour, pickup_zone], columns are dropoff zones).
        dropoff_zones (array-like): List of all dropoff zones.
    
    Returns:
        Tuple: Numpy arrays of pickup and dropoff probability matrices.
    """
    # Pickup probabilities by hour and pickup zone
    pickup_hours = prob_pickup.index.unique()
    pickup_zones = prob_pickup.columns.values

    pickup_prob_matrix = np.zeros((len(pickup_hours), len(pickup_zones)))

    for i, hour in enumerate(pickup_hours):
        pickup_prob_matrix[i] = prob_pickup.loc[hour].values

    # Dropoff probabilities by hour, pickup zone, and dropoff zone
    dropoff_prob_matrix = np.zeros((len(pickup_hours), len(pickup_zones), len(dropoff_zones)))

    for i, hour in enumerate(pickup_hours):
        for j, zone_pickup in enumerate(pickup_zones):
            try:
                prob_values = prob_dropoff.loc[(hour, zone_pickup)].reindex(dropoff_zones).fillna(0).values
                total = prob_values.sum()
                if total > 0:
                    prob_values = prob_values / total
                else:
                    # If there's no data, distribute uniformly
                    prob_values = np.ones(len(dropoff_zones)) / len(dropoff_zones)
                dropoff_prob_matrix[i, j] = prob_values
            except KeyError:
                # If the combination doesn't exist, distribute uniformly
                dropoff_prob_matrix[i, j] = np.ones(len(dropoff_zones)) / len(dropoff_zones)

    return pickup_prob_matrix, dropoff_prob_matrix

# Pre-calculate probabilities
dropoff_zones = dropoff_probability.columns.values
pickup_prob_matrix, dropoff_prob_matrix = pre_calculate_probabilities(
    pickup_probability, dropoff_probability, dropoff_zones
)

# 4. Optimized Sampling Functions

def sample_pickup_vector(hour, num_samples, pickup_prob_matrix, pickup_zones, min_hour):
    """
    Vectorized function to sample pickup locations for a given hour.
    
    Args:
        hour (int): Hour to sample.
        num_samples (int): Number of samples.
        pickup_prob_matrix (np.ndarray): Matrix of pickup probabilities.
        pickup_zones (np.ndarray): Array of pickup zone names.
        min_hour (int): Minimum hour used to adjust matrix index if needed.
    
    Returns:
        np.ndarray: Array of sampled pickup locations.
    """
    # Adjust the index of the hour if necessary (here we locate the exact row index in the DataFrame index)
    hour_index = np.where(pickup_probability.index == hour)[0][0]
    probs = pickup_prob_matrix[hour_index]

    # Ensure probabilities sum to 1
    if not np.isclose(probs.sum(), 1):
        print(f"Pickup probabilities for hour {hour} do not sum to 1. Adjusting.")
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Uniform distribution if no data
            probs = np.ones(len(pickup_zones)) / len(pickup_zones)

    return np.random.choice(pickup_zones, size=num_samples, p=probs)

def sample_dropoff_vector(hour, pickups, dropoff_prob_matrix, dropoff_zones, pickup_zones, min_hour):
    """
    Vectorized function to sample dropoff locations based on a pre-calculated probability matrix.
    
    Args:
        hour (int): The trip hour.
        pickups (np.ndarray): Array of sampled pickup zones.
        dropoff_prob_matrix (np.ndarray): Matrix of dropoff probabilities.
        dropoff_zones (np.ndarray): Array of dropoff zone names.
        pickup_zones (np.ndarray): Array of pickup zone names.
        min_hour (int): Minimum hour used to adjust matrix index if needed.
    
    Returns:
        np.ndarray: Array of sampled dropoff locations.
    """
    hour_index = np.where(pickup_probability.index == hour)[0][0]
    dropoffs = np.empty_like(pickups, dtype=object)

    # Map pickup zone names to indices
    pickup_zone_indices = {zone: idx for idx, zone in enumerate(pickup_zones)}

    for i, pickup in enumerate(pickups):
        pickup_idx = pickup_zone_indices.get(pickup, None)
        
        if pickup_idx is not None:
            probs = dropoff_prob_matrix[hour_index, pickup_idx]
            # Ensure probabilities sum to 1
            if not np.isclose(probs.sum(), 1):
                print(f"Dropoff probabilities for hour {hour}, pickup '{pickup}' do not sum to 1. Adjusting.")
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # Uniform distribution if no data
                    probs = np.ones(len(dropoff_zones)) / len(dropoff_zones)
        else:
            # Uniform distribution if the pickup zone is not found
            print(f"Pickup zone '{pickup}' not found. Using uniform distribution for dropoff.")
            probs = np.ones(len(dropoff_zones)) / len(dropoff_zones)
        
        dropoffs[i] = np.random.choice(dropoff_zones, p=probs)

    return dropoffs

# 5. Synthetic Data Generation

# Initialize a list to store synthetic trips
synthetic_trips = []

# Determine the smallest peak hour
min_hour = peak_hours_values['hour'].min()

# Start a global timer
total_start_time = time.time()
print("\nPeak hours used for synthetic data generation:")
print(peak_hours_values)

# Iterate over each peak hour
for _, row in peak_hours_values.iterrows():
    hour = row['hour']
    num_trips = row['hour_count']
    print(f"\nGenerating data for hour {hour} with {num_trips} trips.")

    # Start timer for this hour
    start_time = time.time()

    # Sample all pickups at once
    pickups = sample_pickup_vector(
        hour, num_trips, pickup_prob_matrix, pickup_probability.columns.values, min_hour
    )

    # Sample all dropoffs at once
    dropoffs = sample_dropoff_vector(
        hour, pickups, dropoff_prob_matrix, dropoff_probability.columns.values,
        pickup_probability.columns.values, min_hour
    )

    temp_df = pd.DataFrame({
        'hour': [hour] * num_trips,
        'pickup_location': pickups,
        'dropoff_location': dropoffs
    })

    synthetic_trips.append(temp_df)

    # Stop timer for this hour
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Show time taken for this hour
    print(f"Time spent generating data for hour {hour}: {elapsed_time:.2f} seconds")

# Concatenate all synthetic trips into one DataFrame
synthetic_trips_df = pd.concat(synthetic_trips, ignore_index=True)
synthetic_trips_df.to_parquet("synthetic_data_v2.parquet")

# Stop the global timer
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time
print(f"\nTotal time spent generating synthetic data: {total_elapsed_time:.2f} seconds")

# Optional: Check the distribution of synthetic trips
print("\nExamples of generated synthetic trips:")
print(synthetic_trips_df.head())

# Check if all dropoff probabilities sum to 1
sums = dropoff_prob_matrix.sum(axis=2)
if not np.allclose(sums, 1):
    print("There are hour and pickup combinations that do not sum to 1 in the dropoff probabilities.")
else:
    print("All hour and pickup combinations sum to 1 in the dropoff probabilities.")

# Check if the total number of synthetic trips matches the total number of real trips
total_synthetic_trips = synthetic_trips_df.shape[0]
total_real_trips = trips_per_hour_peak.shape[0]
print(f"\nTotal synthetic trips generated: {total_synthetic_trips}")
print(f"Total real trips (after filtering and removing duplicates): {total_real_trips}")
