import pandas as pd
import geopandas as gpd
import numpy as np

# 1. Data Preparation

parquet_files = [
    '/home-ext/caioloss/Dados/yellow_tripdata_2024-01.parquet'
]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

taxi_zones = gpd.read_file('/home-ext/caioloss/Dados/taxi-zones')
taxi_zones = taxi_zones.drop("geometry", axis=1)

# Create a dictionary to map LocationID to zone
location_zone_dict = dict(zip(taxi_zones['LocationID'], taxi_zones['zone']))

# Replace Location IDs with zone names
df['pickup_location'] = df['PULocationID'].map(location_zone_dict)
df['dropoff_location'] = df['DOLocationID'].map(location_zone_dict)

# Identify the top 5 peak hours with the highest number of trips
df['hour'] = df['tpep_pickup_datetime'].dt.round('h').dt.hour
trips_per_hour = df[['hour', 'pickup_location', 'dropoff_location']]
peak_hours_values = (
    trips_per_hour
    .groupby('hour')
    .size()
    .sort_values(ascending=False)
    .reset_index(name='hour_count')[:5]
)
peak_hours = peak_hours_values['hour']

# Filter only the trips for peak hours and drop duplicates
trips_per_hour_peak = trips_per_hour[trips_per_hour['hour'].isin(peak_hours)].drop_duplicates()

# 2. Calculation of Pickup and Dropoff Probabilities

# Calculate the total count of pickups by hour
total_count_per_hour = (
    trips_per_hour_peak
    .groupby('hour')
    .size()
    .reset_index(name='total_hour_count')
)

# Calculate the total count of pickups by hour and pickup_location
total_count_per_hour_pickup = (
    trips_per_hour_peak
    .groupby(['hour', 'pickup_location'])
    .size()
    .reset_index(name='total_hour_count_pickup')
)

# Calculate the counts for each combination of hour, pickup_location, and dropoff_location
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

# Re-normalize to ensure probabilities sum to 1 by hour
pickup_probability = pickup_probability.div(pickup_probability.sum(axis=1), axis=0).fillna(0)
sum_by_hour = pickup_probability.sum(axis=1)

# Get all unique dropoff locations
all_dropoff_locations = df['dropoff_location'].unique()
all_dropoff_locations = taxi_zones['zone'].unique()

# Pivot to create the dropoff probability matrix
dropoff_probability = prob_df_dropoff.pivot_table(
    index=['hour', 'pickup_location'],
    columns='dropoff_location',
    values='dropoff_probability'
).fillna(0)

# Reindex columns to include all dropoff locations
dropoff_probability = dropoff_probability.reindex(columns=all_dropoff_locations, fill_value=0)

# Re-normalize to ensure dropoff probabilities sum to 1 for each hour and pickup_location
dropoff_probability = dropoff_probability.div(dropoff_probability.sum(axis=1), axis=0).fillna(0)

# 3. Sampling Functions

def sample_pickup(hour, num_samples, pickup_probability):
    """
    Samples pickup locations for a given hour.
    
    Args:
        hour (int): The hour for which to sample.
        num_samples (int): Number of samples to be generated.
        pickup_probability (DataFrame): DataFrame containing pickup probabilities.
        
    Returns:
        np.ndarray: Array of sampled pickup locations.
    """
    # Get probabilities for the specific hour
    probs = pickup_probability.loc[hour].values
    locations = pickup_probability.columns.values
    return np.random.choice(locations, p=probs)

def sample_dropoff(hour, pickups, dropoff_probability):
    """
    Samples dropoff locations based on pickup and hour, using tuples of (location, probability).
    
    Args:
        hour (int): The hour of the trip.
        pickups (np.ndarray): Array of sampled pickup locations.
        dropoff_probability (DataFrame): DataFrame containing dropoff probabilities.
        
    Returns:
        np.ndarray: Array of sampled dropoff locations.
    """
    dropoffs = []
    total_pickups = len(pickups)  # Total number of pickups

    for i, pickup in enumerate(pickups, start=1):
        # Get probabilities for the hour + pickup combination
        try:
            probs = dropoff_probability.loc[(hour, pickup)].values
            locations = dropoff_probability.columns.values
            location_probs = list(zip(locations, probs))  # Create tuples of (location, probability)
            
            # Filter only tuples with probability > 0 to avoid invalid sampling
            filtered_location_probs = [(loc, prob) for loc, prob in location_probs if prob > 0]
            
            # Separate locations and probabilities for sampling
            filtered_locations, filtered_probs = zip(*filtered_location_probs)

            # Sample a dropoff location based on filtered probabilities
            dropoff = np.random.choice(filtered_locations, p=filtered_probs)
        except (KeyError, ValueError):
            print("Error encountered, choosing a random dropoff location")
            # If there's no data for this combination, choose randomly
            dropoff = np.random.choice(dropoff_probability.columns.values)
        
        # If you want multiple samples, you'd collect them all. 
        # As the code is written, it returns on the first iteration. 
        # Adjust as needed if you want all dropoffs.
        return dropoff

# 4. Synthetic Data Generation

# Initialize a list to store synthetic trips
synthetic_trips = []

# Dummy peak hours data for testing
peak_hours_values_similar = pd.DataFrame({
    'hour': range(15, 20),
    'hour_count': [10] * 5
})

# Iterate over each peak hour
for _, row in peak_hours_values_similar.iterrows():
    hour = row['hour']
    num_trips = row['hour_count']
    
    print(f"Started generating data for hour {hour}")
    
    # Generate the specified number of trips in 'hour_count'
    for _ in range(num_trips):
        # Sample a pickup location
        pickup = sample_pickup(hour, 1, pickup_probability)
        # Sample a dropoff location conditioned on pickup and hour
        dropoff = sample_dropoff(hour, [pickup], dropoff_probability)
        
        # Create a temporary DataFrame to store this trip
        temp_df = pd.DataFrame({
            'hour': [hour],
            'pickup_location': [pickup],
            'dropoff_location': [dropoff]
        })
        
        # Add to the main DataFrame
        synthetic_trips.append(temp_df)

# Concatenate all synthetic trips into a single DataFrame
synthetic_trips_df = pd.concat(synthetic_trips, ignore_index=True)
synthetic_trips_df.to_csv("visualize.csv")

# Optional: Check the distribution of synthetic trips
print("\nExamples of generated synthetic trips:")
print(synthetic_trips_df.head())

exit()
print("\nDistribution of synthetic trips by hour:")
print(synthetic_trips_df['hour'].value_counts().sort_index())
