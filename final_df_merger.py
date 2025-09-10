import os
import re
import unicodedata
import pandas as pd
import glob
from final_fbref_scraper import get_epl_players
from difflib import SequenceMatcher

seasons = [
    ("2023-2024", "Fantasy-Premier-League/data/2023-24", "new_data/2023-2024/fbref"),
    ("2024-2025", "Fantasy-Premier-League/data/2024-25", "new_data/2024-2025/fbref"),
    ("2025-2026", "Fantasy-Premier-League/data/2025-26", "new_data/2025-2026/fbref"),
]

def fill_missing_matchweeks(player_df, season, current_gw):
    player_timeline = []
    if season == "2025-2026":
        x = current_gw + 1
    else:
        x = 39
    for gw in range(1, x):
        player_timeline.append({'Season': season, 'round': f'Matchweek {gw}'})
    player_timeline = pd.DataFrame(player_timeline)
    player_timeline['FBRef_ID'] = player_df['FBRef_ID'].iloc[0]
    merged_df = pd.merge(player_timeline, player_df, on=['Season', 'round', 'FBRef_ID'], how='left')
    # print(player_df['minutes'], merged_df['minutes'])
    merged_df['minutes'] = merged_df['minutes'].fillna(0)
    return merged_df

def add_rolling_minutes(player_df):
    player_df['minutes_weightings'] = (
        player_df.groupby('FBRef_ID')['minutes_fbref']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    return player_df

def normalize_name(name):
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    cleaned = re.sub(r'[^a-zA-Z\s]', '', str(name))
    return ' '.join(cleaned.lower().split())

def find_fpl_files(season_folder, matched_id):
    return glob.glob(os.path.join(season_folder, "players", f"*_{matched_id}"))

def name_parts_in_other(shorter, longer):
    nickname_map = {
        "maximilian": "max",
        "max": "maximilian",
        "oliver": "ollie",
        "ollie": "oliver",
        "benjamin": "ben",
        "ben": "benjamin",
        "alexander": "alex",
        "alex": "alexander",
        "michael": "mike",
        "mike": "michael",
        "william": "will",
        "will": "william",
        "james": "jim",
        "jim": "james",
        "nicholas": "nick",
        "nick": "nicholas",
    }
    shorter_parts = shorter.split()
    longer_parts = longer.split()
    for part in shorter_parts:
        # Direct match or nickname match
        if part not in longer_parts:
            # Check nickname mapping
            if part in nickname_map and nickname_map[part] in longer_parts:
                continue
            # Check substring match (e.g., "ollie" in "oliver")
            if any(part in lp or lp in part for lp in longer_parts):
                continue
            return False
    return True


# Load current season FPL player list
current_season = "2025-2026"
idlist_path = f"Fantasy-Premier-League/data/2025-26/player_idlist.csv"
id_df = pd.read_csv(idlist_path)
id_df['Normalized_Name'] = (id_df['first_name'] + ' ' + id_df['second_name']).apply(normalize_name)
id_df['FBRef_id'] = None
MANUAL_FBREF_OVERRIDES = {
    normalize_name("Joao Pedro Junqueira de Jesus"): "e8832875",
    normalize_name("Joao Pedro Ferreira da Silva"): "4d77e622",
}

for norm_name, fbref_id in MANUAL_FBREF_OVERRIDES.items():
    mask = id_df['Normalized_Name'] == norm_name
    if mask.any():
        id_df.loc[mask, 'FBRef_id'] = fbref_id

for url in [
    "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats"
]:
    players, stats = get_epl_players(url)
    for player_id, player_obj in players.items():
        player_name = player_obj.data[0]['player']
        normalized_name = normalize_name(player_name)
        # Exact normalized name match first
        mask_exact = (id_df['Normalized_Name'] == normalized_name)
        if mask_exact.any():
            # Do not overwrite manual assignment
            id_df.loc[mask_exact & id_df['FBRef_id'].isna(), 'FBRef_id'] = player_id
            continue
                # Fallback: relaxed match (keep previous logic but skip already set IDs)
        mask_relaxed = id_df['Normalized_Name'].apply(
            lambda x: all(part in x.split() for part in normalized_name.split())
        )
        if mask_relaxed.any():
            id_df.loc[mask_relaxed & id_df['FBRef_id'].isna(), 'FBRef_id'] = player_id

        id_df.loc[mask, 'FBRef_id'] = player_id

# Sanity check: any FBRef_id used by multiple distinct normalized names?
dupes = (id_df[id_df['FBRef_id'].notna()]
         .groupby('FBRef_id')['Normalized_Name']
         .nunique()
         .reset_index())
dupes = dupes[dupes['Normalized_Name'] > 1]
if not dupes.empty:
    print("Warning: FBRef IDs shared by multiple names:")
    print(dupes)

all_players_df = []

fpl_id_map = {}
for season, season_folder, _ in seasons:
    idlist_path = os.path.join(season_folder, "player_idlist.csv")
    if os.path.exists(idlist_path):
        season_id_df = pd.read_csv(idlist_path)
        season_id_df['Normalized_Name'] = (season_id_df['first_name'] + ' ' + season_id_df['second_name']).apply(normalize_name)
        for _, row in season_id_df.iterrows():
            norm_name = row['Normalized_Name']
            fpl_id = row['id']
            if norm_name not in fpl_id_map:
                fpl_id_map[norm_name] = []
            fpl_id_map[norm_name].append((season, fpl_id))

# Create a list of keys for iteration
keys = list(fpl_id_map.keys())

# Use a dictionary to track merged keys
merged_keys = {}

for i, key1 in enumerate(keys):
    if key1 in merged_keys:  # Skip if key1 has already been merged
        continue
    for j in range(i + 1, len(keys)):  # Only compare keys that come after key1
        key2 = keys[j]
        if key2 in merged_keys:  # Skip if key2 has already been merged
            continue

        # Calculate similarity ratio
        a = SequenceMatcher(None, key1, key2).ratio()
        shorter, longer = (key1, key2) if len(key1.split()) <= len(key2.split()) else (key2, key1)

        # Check similarity and name parts
        if a > 0.7 and a != 1.0 and name_parts_in_other(shorter, longer):
            # Merge key2 into key1, keeping both names
            fpl_id_map[key1] = sorted(fpl_id_map[key1] + fpl_id_map[key2], key=lambda x: x[0])
            if key1 not in merged_keys:
                merged_keys[key1] = [key1]  # Initialize with key1
            merged_keys[key1].append(key2)  # Add key2 to the list of merged names
            merged_keys[key2] = merged_keys[key1]  # Point key2 to the same list

# Update fpl_id_map to include all merged names
for key, merged_names in merged_keys.items():
    for name in merged_names:
        fpl_id_map[name] = fpl_id_map[key]

def main(current_gw):
    for idx, row in id_df.iterrows():
        player_fpl_id = row['id']
        
        player_name = f"{row['first_name']} {row['second_name']}"
        normalized_name = row['Normalized_Name']
        fbref_player_id = row['FBRef_id']
        # print(f"Processing: {player_name} (FPL ID: {player_fpl_id}, FBref ID: {fbref_player_id})")

        player_season_dfs = []
        for season, season_folder, fbref_folder in seasons:
            # Find FPL file(s) for this player in this season
            season_fpl_ids = [fpl_id for s, fpl_id in fpl_id_map[normalized_name] if s == season]
            cleaned_players = pd.read_csv(os.path.join(season_folder, "cleaned_players.csv"))
            cleaned_players['Normalized_Name'] = (cleaned_players['first_name'] + ' ' + cleaned_players['second_name']).apply(normalize_name)

            matched_positions = []
            for cp_name in cleaned_players['Normalized_Name']:
                a = SequenceMatcher(None, normalized_name, cp_name).ratio()
                shorter, longer = (normalized_name, cp_name) if len(normalized_name.split()) <= len(cp_name.split()) else (cp_name, normalized_name)
                if a > 0.7 and name_parts_in_other(shorter, longer):
                    pos = cleaned_players.loc[cleaned_players['Normalized_Name'] == cp_name, 'element_type']
                    if not pos.empty:
                        matched_positions.append(pos.iloc[0])
            if matched_positions:
                player_position = matched_positions[0]
            else:
                player_position = None
                # print(player_name)

            fpl_dfs = []
            if season_fpl_ids:
                fpl_files = find_fpl_files(season_folder, season_fpl_ids[0])
                for fpl_folder in fpl_files:
                    gw_path = os.path.join(fpl_folder, "gw.csv")
                    if os.path.exists(gw_path):
                        fpl_df = pd.read_csv(gw_path)
                        fpl_df['FPL_ID'] = player_fpl_id
                        fpl_df['Player_Name'] = player_name
                        fpl_df['Season'] = season
                        fpl_df['date'] = pd.to_datetime(fpl_df['kickoff_time'], errors='coerce').dt.date

                        fpl_dfs.append(fpl_df)
                if fpl_dfs:
                    fpl_df = pd.concat(fpl_dfs, ignore_index=True)
                    # Try to find fbref file using matched fbref_player_id
                    if fbref_player_id:
                        fbref_path = os.path.join(fbref_folder, f"{fbref_player_id}.csv")
                        if os.path.exists(fbref_path):
                            fbref_df = pd.read_csv(fbref_path)
                            fbref_df['FBRef_ID'] = fbref_player_id
                            fbref_df['Season'] = season
                            fbref_df['Player_Name'] = player_name
                            # print(fbref_df.columns)
                            fbref_df['date'] = pd.to_datetime(fbref_df['date'], errors='coerce').dt.date #align fbref and fpl dates
                            fbref_df = fill_missing_matchweeks(fbref_df, season, current_gw=current_gw) # very important to update each week
                            merged_df = pd.merge(
                                fbref_df,
                                fpl_df,
                                on=['date'],
                                how='left',
                                suffixes=('_fbref', '_fpl')
                            )
                            merged_df['FPL Position'] = player_position
                            player_season_dfs.append(merged_df)
                            # print(merged_df['saves'])
                        else:
                            continue
                            # print(f"No FBRef file found for player {player_name} (FBref ID: {fbref_player_id}) in {season}")
                    else:
                        pass
                        # print(f"No FBRef overview match for player {player_name}")
                else:
                    pass
                    # print(f"No FPL gw.csv found for player {player_name} in {season}")
            else:
                continue
                # print(f"No FPL data found for player {player_name} in {season}")

        if player_season_dfs:
            player_df = pd.concat(player_season_dfs, ignore_index=True)
            player_df['date'] = pd.to_datetime(player_df['date'], errors='coerce')
            player_df = player_df.sort_values('date').reset_index(drop=True)
            player_df = add_rolling_minutes(player_df)
            all_players_df.append(player_df)

    # Concatenate all player dataframes into one big dataframe
    if all_players_df:
        final_df = pd.concat(all_players_df, ignore_index=True)
        for col in final_df.columns:
            if 'mng' in col:
                final_df = final_df.drop(columns=[col])
        final_df.drop(['fouled', 'pens_conceded', 'offsides', 'crosses', 'fouls', 'pens_won', 'own_goals_fpl', 'own_goals_fbref'], axis=1, inplace=True, errors='ignore')
        final_df.to_csv("all_players_merged.csv", index=False)
        print("Saved all players to all_players_merged.csv")
    else:
        print("No merged data to save.")

if __name__ == "__main__":
    current_gw = 3  # Update this as needed
    main(current_gw)