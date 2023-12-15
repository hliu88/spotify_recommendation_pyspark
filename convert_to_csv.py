import json
import pandas as pd

def jsonToCSV():
    start = 0
    end = 1000
    while end != 1000000:
        lst = []
        path = "mpd.slice.%d-%d.json" % (start, end-1)
        j = json.load(open(path, 'r'))
        j = pd.DataFrame.from_dict(j['playlists'], orient='columns')
        for _, row in j.iterrows():
            for track in row['tracks']:
                lst.append([track['track_uri'], track['artist_name'], track['track_name'], row['pid']])
        df = pd.DataFrame(lst, columns=['trackid', 'artist_name', 'track_name', 'pid'])
        df.to_csv('csv/%d-%d.csv' % (start, end-1), index=False)
        start, end = start + 1000, end + 1000


if __name__ == '__main__':
    jsonToCSV()