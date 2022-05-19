#%%
import pandas as pd
import geopandas as gpd
import osmnx as ox

df = pd.read_csv('pollution_samples_terrassa.csv').set_index('timestamp')
df_normalized = df.copy()
df_normalized['co'] = (df_normalized['co'] - df_normalized['co'].min()) / (
    df_normalized['co'].max() - df_normalized['co'].min())
df_normalized['no2'] = (df_normalized['no2'] - df_normalized['no2'].min()) / (
    df_normalized['no2'].max() - df_normalized['no2'].min())
df_normalized['pm25'] = (df_normalized['pm25'] - df_normalized['pm25'].min()
                         ) / (df_normalized['pm25'].max() -
                              df_normalized['pm25'].min())
df_normalized['pm10'] = (df_normalized['pm10'] - df_normalized['pm10'].min()
                         ) / (df_normalized['pm10'].max() -
                              df_normalized['pm10'].min())
df_combined = df_normalized.copy()
df_combined['Pollution'] = df_combined.loc[:,
                                           ['co', 'no2', 'pm25', 'pm10']].mean(
                                               axis=1)
df = df_combined.drop(['co', 'no2', 'pm25', 'pm10'], axis=1)
#%%
gdf = gpd.GeoDataFrame(df,
                       geometry=gpd.points_from_xy(df['longitude'],
                                                   df['latitude']),
                       crs='EPSG:4326')
# %%
G = ox.graph_from_place("terrassa", network_type="drive")
nodes, edges = ox.graph_to_gdfs(G)