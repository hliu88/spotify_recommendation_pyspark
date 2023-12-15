import os
import csv
import pandas as pd


folder = 'csv'
folder_list = sorted(os.listdir(folder))
df = pd.DataFrame()
for i in folder_list:
    df1 = pd.read_csv(os.path.join(folder, i))
    df = pd.concat([df, df1], axis=0)

# save the dataframe to csv
df.to_csv('all_data_raw.csv', index=False)

# create a dictionary that maps trackid to track_num
df_trackid_pid = df.copy()
df_trackid_pid = df_trackid_pid.drop(columns=['artist_name', 'track_name'])
df_trackid_pid['trackid'] = df_trackid_pid['trackid'].astype('category').cat.codes
trackid_dict = dict(zip(df['trackid'], df_trackid_pid['trackid']))

# save the dictionary to csv just in case
with open('trackid_dict.csv', 'w') as f:
    w = csv.DictWriter(f, trackid_dict.keys())
    w.writeheader()
    w.writerow(trackid_dict)

# create a dataframe that contains track_num and trackid
df_track_info = df.copy()
df_track_info = df_track_info.drop(columns=['pid'])
df_track_info = df_track_info.drop_duplicates(subset=['trackid'])
df_track_info['track_num'] = df_track_info['trackid'].map(trackid_dict)

# save the dataframe to csv
df_track_info.to_csv('df_track_info.csv', index=False)