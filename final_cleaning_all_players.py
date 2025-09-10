import pandas as pd
from bs4 import BeautifulSoup

df = pd.read_csv("all_players_merged.csv")
df['date'] = pd.to_datetime(df['date'])
df['team'] = df.groupby('FBRef_ID')['team'].transform(lambda x: x.ffill().bfill())

min_minutes_threshold = 150 # minimum minutes over entire PL career
minutes_per_player = df.groupby('FBRef_ID')['minutes_fbref'].transform('sum')
before_count = df['FBRef_ID'].nunique()
df = df[minutes_per_player >= min_minutes_threshold].copy()
after_count = df['FBRef_ID'].nunique()


availability_df = df.copy()
# print(availability_df.columns)

seasons = [
    ("2023-2024", "Fantasy-Premier-League/data/2023-24", "new_data/2023-2024/fbref"),
    ("2024-2025", "Fantasy-Premier-League/data/2024-25", "new_data/2024-2025/fbref"),
    ("2025-2026", "Fantasy-Premier-League/data/2025-26", "new_data/2025-2026/fbref"),
]

teams_dict = {
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester Utd": "Man Utd",
    "Newcastle Utd": "Newcastle",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Tottenham": "Spurs",
}

team_cols = ['strength_overall_home_team', 'strength_overall_away_team', 'strength_attack_home_team', 'strength_defense_home_team']
away_team_cols = ['strength_overall_home_opp', 'strength_overall_away_opp', 'strength_attack_home_opp', 'strength_defense_home_opp']
# print(df['opponent'].unique())
# print(df['team'].unique()) # needed to change teams.csv names to match

team_strength_data = {}
for season_fbref, fpl_path, fbref_path in seasons:
    team_strength_data[season_fbref] = pd.read_csv(f"{fpl_path}/teams.csv")

def get_team_strength(row):
    teams_df = team_strength_data[row['Season_fbref']]
    team_name = BeautifulSoup(row['team'], 'html.parser').a.text.replace("\\", "")
    opponent_team_name = row['opponent']
    team_name = teams_dict.get(team_name, team_name)
    opponent_team_name = teams_dict.get(opponent_team_name, opponent_team_name)

    if row['was_home'] == 1:
        team_strength = teams_df.loc[teams_df['name'] == team_name, 'strength_overall_home'].values[0]
        opponent_strength = teams_df.loc[teams_df['name'] == opponent_team_name, 'strength_overall_away'].values[0]
    else:
        team_strength = teams_df.loc[teams_df['name'] == team_name, 'strength_overall_away'].values[0]
        opponent_strength = teams_df.loc[teams_df['name'] == opponent_team_name, 'strength_overall_home'].values[0]

    return pd.Series([team_strength, opponent_strength])


availability_df['was_home'] = availability_df['was_home'].astype(str).map({'True': 1, 'False': 0})
availability_df['starts'] = pd.to_numeric(availability_df['starts'], errors='coerce').fillna(0).astype(int)
availability_df['played_flag'] = (availability_df['minutes_fbref'] > 0).astype(int)
availability_df['started_flag'] = (availability_df['starts'] == 1).astype(int)
# availability_df = availability_df.sort_values(['FBRef_ID', 'Season_fbref', 'date']).reset_index(drop=True) should not have reset_index

for k in [2, 3, 5, 8]:
    availability_df[f'roll_played_rate_{k}'] = (
        availability_df.groupby(['FBRef_ID', 'Season_fbref'], observed=True)['played_flag']
        .transform(lambda x: x.rolling(k, min_periods=1).mean())
    )
    availability_df[f'roll_started_rate_{k}'] = (
        availability_df.groupby(['FBRef_ID', 'Season_fbref'], observed=True)['started_flag']
        .transform(lambda x: x.rolling(k, min_periods=1).mean())
    )

brier_scores = {}
best_k = 3
for k in [2, 3, 5, 8]:
    col = f'roll_started_rate_{k}'
    valid = availability_df[[col, 'started_flag']].dropna()
    if len(valid) == 0:
        continue
    brier = ((valid[col] - valid['started_flag']) ** 2).mean()
    brier_scores[k] = float(brier)

if brier_scores:
    best_k = min(brier_scores, key=brier_scores.get)
    print(f"Best k for started rate: {best_k}")
    availability_df['start_prob'] = availability_df[f'roll_started_rate_{best_k}'].fillna(0.5)
else:
    availability_df['start_prob'] = availability_df['roll_started_rate_3'].fillna(0.5)

availability_model_cols = [
    'Season_fbref', 'FBRef_ID', 'Player_Name_fbref', 'round_fbref', 'date',
    'team', 'opponent',
    'played_flag', 'started_flag',
    'roll_played_rate_2', 'roll_played_rate_3', 'roll_played_rate_5', 'roll_played_rate_8',
    'roll_started_rate_2',  'roll_started_rate_3', 'roll_started_rate_5', 'roll_started_rate_8',
    'start_prob'
]

availability_dataset = availability_df[availability_model_cols]
availability_df = availability_df.sort_values(['FBRef_ID', 'Season_fbref', 'date'])
availability_dataset.to_csv("availability_dataset.csv", index=False)


players = ["Liam Delap", "João Pedro Junqueira de Jesus"]
print(availability_dataset[(availability_dataset['Player_Name_fbref'].isin(players)) & 
                           (availability_dataset['Season_fbref'] == '2025-2026')].sort_values(['Player_Name_fbref', 'date'])[['Player_Name_fbref', 'start_prob', 'roll_started_rate_2']])



df = df[df['minutes_fbref'] > 0]  # keep only matches where player appeared
df[['team_strength', 'opponent_strength']] = df.apply(get_team_strength, axis=1)
df['was_home'] = df['was_home'].astype(str).map({'True': 1, 'False': 0})
df['started_given_played'] = (df['minutes_fbref'] > 60).astype(int)
df['60_played'] = (df['minutes_fbref'] > 60).astype(int)
df['90_played'] = (df['minutes_fbref'] > 90).astype(int)
df['minutes_weightings'] = df['minutes_fbref'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean()) / 90



deprecated_cols = [
    'dayofweek', 'game_started', 'passes_pct', 'take_ons', 'position', 
    'take_ons_won', 'Player_Name_fpl', 'Season_fpl', 'modified',
    'transfers_in', 'transfers_out', 'expected_goal_involvements', 'goals_scored',
    'recoveries', 'tackles_fpl', 'element', 'fixture', 'match_report', 'ict_index',
    'kickoff_time', 'own_goals', 'penalties_missed', '90_played'
]


admin_cols = [
    'Season_fbref', 'Player_Name_fbref',
    'FBRef_ID', 'FPL_ID', 'date', 'value',
    'bench_explain', 'round_fbref', 'round_fpl', 'opponent', 'opponent_team',
    'team', 'was_home', 'total_points', 'bonus', 'bps', 'selected',
    'transfers_balance', 'team_strength', 'opponent_strength', '60_played',
    'minutes_fbref', 'minutes_fpl', 'started_given_played', 'minutes_weightings'
]

gk_cols = admin_cols + [
    'clean_sheets', 'goals_conceded', 'penalties_saved',
    'team_a_score', 'team_h_score', 'result', 'was_home',
    'passes', 'passes_completed', 'passes_pct', 'assists_fpl',
    'clearances_blocks_interceptions', 'defensive_contribution',
    'yellow_cards', 'red_cards', 'cards_red',
    'expected_goals_conceded', 'saves'
]

def_cols = admin_cols + [
    'npxg', 'goals', 'shots', 'blocks', 'tackles_won', 'sca', 'gca',
    'interceptions', 'assists_fpl', 'clean_sheets', 'touches', 'progressive_passes',
    'passes_completed', 'expected_goals_conceded', 'expected_assists',
    'expected_goals', 'goals_conceded', 'influence', 'creativity', 'threat'
]

mid_cols = admin_cols + [
    'npxg', 'goals', 'shots', 'blocks', 'tackles_won', 'sca', 'gca',
    'interceptions', 'assists_fpl', 'clean_sheets', 'touches', 'progressive_passes',
    'passes_completed', 'expected_goals', 'expected_assists',
    'expected_goals', 'goals_conceded', 'influence', 'creativity', 'threat'
]

fwd_cols = admin_cols + [
    'npxg', 'goals', 'shots', 'blocks', 'tackles_won', 'sca', 'gca',
    'interceptions', 'assists_fpl', 'touches', 'progressive_passes',
    'passes_completed', 'expected_goals', 'expected_assists',
    'expected_goals', 'goals_conceded', 'influence', 'creativity', 'threat'
]

gk_df = df[df['FPL Position'] == 'GK']
def_df = df[df['FPL Position'] == 'DEF']
mid_df = df[df['FPL Position'] == 'MID']
fwd_df = df[df['FPL Position'] == 'FWD']

gk_stats = gk_df[gk_cols]
def_stats = def_df[def_cols]
mid_stats = mid_df[mid_cols]
fwd_stats = fwd_df[fwd_cols]

gk_stats.to_csv("merged_gk_stats.csv", index=False)
def_stats.to_csv("merged_def_stats.csv", index=False)
mid_stats.to_csv("merged_mid_stats.csv", index=False)
fwd_stats.to_csv("merged_fwd_stats.csv", index=False)
