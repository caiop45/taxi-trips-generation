The files **synthetic_data_taxi_trip_v1** and **synthetic_data_taxi_trip_v2** are two initial versions for generating synthetic taxi trip data in NYC. Both use **pickup-dropoff probability matrices** to simulate real travel patterns, and the generated data is later used for **training a machine learning model**.  

The quality of the generated data can be evaluated through the **pickup and dropoff frequency per neighborhood**, which are stored in the **pickup_frequency** and **dropoff_frequency** files. These files allow for a comparison between synthetic and real data.  

To run the code, it is necessary to **download real taxi trip data** directly from the NYC government website: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### **Libraries Used**  
- **Pandas**: Data manipulation (file reading, filtering, grouping, etc.).  
- **GeoPandas**: Handling geospatial data (mapping location IDs to boroughs).  
- **Numpy**: Mathematical calculations and vectorization for optimized trip sampling.  
- **Matplotlib**: Creating comparative charts between synthetic and real data.  
- **Time**: Measuring performance during data generation.  

### **Summary of Code Functionalities**  
- **synthetic_data_taxi_trip_v1**: A simpler implementation where trip sampling occurs sequentially.  
- **synthetic_data_taxi_trip_v2**: Optimized with **precomputed matrices** and vectorized functions to speed up data generation.  
- **pickup_frequency & dropoff_frequency**: Compare the frequency of real and synthetic trips, generating visual analysis charts.

- ### **Graphs**
- ![pickup_frequency](https://github.com/user-attachments/assets/4414f482-5fb9-4859-b947-5a521c0530a6)
- ![dropoff_frequency](https://github.com/user-attachments/assets/cccf7608-c566-416b-bf5c-3afe2d9e7c8a)
