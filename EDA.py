import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#correlation matrix as heatmap
data = pd.read_csv('merged_pca_11.csv', delimiter=',')
##print(data.shape)
print(data.columns)
#climate_variables = data.iloc[:, 6:]
climate_variables = data.drop(data.columns[[0, 1, 2, 3, 4, 5]], axis=1)
#print(climate_variables.head(3))

coefficient_matrix = climate_variables.corr()
print(coefficient_matrix)
plt.figure(figsize=(10, 10))
sns.heatmap(coefficient_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.savefig('correlation_matrix.png')
plt.show()

# time series over all stations for specific variable
plt.figure(figsize=(12, 6))
for station in data['STATION'].unique():
    station_data = data[data['STATION'] == station]
    plt.plot(station_data['DATE'], station_data['TAVG'], label=f'Station {station}')

plt.title(f'All Stations - TAVG')
plt.xlabel('Year')
plt.ylabel('TAVG')
plt.legend(loc='best')
plt.savefig(f'TAVG_timeseries.png')
plt.close()

# scatter plot
plt.figure(figsize=(12, 6))
for station in data['STATION'].unique():
    station_data = data[data['STATION'] == station]
    plt.scatter(station_data['PRCP'], station_data['SNOW'], label=f'Station {station}')

plt.title(f'PRCP vs SNOW')
plt.xlabel('PRCP')
plt.ylabel('SNOW')
plt.legend(loc='best')
plt.savefig(f'PRCP_vs_SNOW_scatterplot.png')
plt.close()

# histogram
hist1 = data['SNOW']
hist2 = data['TAVG']

plt.figure(figsize=(10, 6))
sns.histplot(hist1, bins=30, color="blue", alpha=0.5, label='SNOW')
sns.histplot(hist2, bins=30, color="red", alpha=0.5, label='TAVG')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Snow and Average Temperature')
plt.legend()
plt.show()