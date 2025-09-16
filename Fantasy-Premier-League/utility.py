import os
import csv

def load_player_fbref_id_map():
    """Load master player_fbref_id_map.csv as a dict."""
    player_map = {}
    if os.path.exists("player_fbref_id_map.csv"):
        with open("player_fbref_id_map.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_map[row["Player_Name_fbref"]] = row["FBRef_ID"]
    return player_map
import sys

def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    """ Wrapper function around print from Stackoverflow
    https://stackoverflow.com/questions/14630288/unicodeencodeerror-charmap-codec-cant-encode-character-maps-to-undefined/16120218
    """
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)
