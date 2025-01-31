import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load pickup and dropoff frequency data
df_pickup = pd.read_csv('frequencias_embarque.csv')
df_dropoff = pd.read_csv('frequencias_desembarque.csv')

def plot_comparative_bar_chart(df, column_location, top_n, title, filename):
    """
    Plots a comparative bar chart of real vs. synthetic frequencies and saves it locally.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least two columns (real and synthetic frequencies).
        column_location (str): Column name representing the location (e.g., 'pickup_location' or 'dropoff_location').
        top_n (int): Number of top locations to display in the chart.
        title (str): Chart title.
        filename (str): Filename to save the chart.
    """
    # Sort the DataFrame by real frequency in descending order and select the top N
    df_sorted = df.sort_values('frequência_real', ascending=False).head(top_n)
    
    # Define positions and bar width
    positions = np.arange(len(df_sorted))
    bar_width = 0.4
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot real frequencies
    bars_real = ax.bar(
        positions - bar_width / 2,
        df_sorted['frequência_real'],
        width=bar_width,
        label='Real',
        color='skyblue'
    )
    
    # Plot synthetic frequencies
    bars_synthetic = ax.bar(
        positions + bar_width / 2,
        df_sorted['frequência_sintética'],
        width=bar_width,
        label='Synthetic',
        color='salmon'
    )
    
    # Configure labels and title
    ax.set_xlabel('Location')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(df_sorted[column_location], rotation=45, ha='right')
    ax.legend()
    
    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the chart locally
    plt.savefig(filename, dpi=300)
    
    # Show the chart (optional)
    plt.show()

# Plot and save the pickup frequency chart
plot_comparative_bar_chart(
    df_pickup,
    column_location='embarque_location',  # Column name in your CSV
    top_n=20,
    title='Comparison of Pickup Frequencies - Top 20 Locations',
    filename='pickup_frequencies_comparison.png'
)

# Plot and save the dropoff frequency chart
plot_comparative_bar_chart(
    df_dropoff,
    column_location='desembarque_location',  # Column name in your CSV
    top_n=20,
    title='Comparison of Dropoff Frequencies - Top 20 Locations',
    filename='dropoff_frequencies_comparison.png'
)
