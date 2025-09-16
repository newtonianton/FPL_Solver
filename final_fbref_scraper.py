import pandas as pd
from bs4 import BeautifulSoup, Comment
import time, re, os, csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

def load_player_fbref_id_map():
    """Load master player_fbref_id_map.csv as a dict."""
    player_map = {}
    if os.path.exists("player_fbref_id_map.csv"):
        with open("player_fbref_id_map.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_map[row["Player_Name_fbref"]] = row["FBRef_ID"]
    return player_map


class MatchData:
    def __init__(self) -> None:
        self.comp = ""
        self.date = ""
        self.round = ""
        self.data = {}


class PlayerData:
    def __init__(self) -> None:
        self.data = []
        self.base_url = ""
        self.matches_links = []
        self.matches = []
        self.match_stat_set = set()


class SquadData:
    def __init__(self):
        self.squad = ""
        self.stats = {}


driver = None


def setup_driver():
    global driver
    if driver is None:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        # Update this path to your chromedriver executable
        service = Service(r"chromedriver-win64/chromedriver.exe")
        driver = webdriver.Chrome(service=service, options=options)
    return driver


def get_fbref_data_selenium(url):  # returns BeautifulSoup object
    """Use Selenium to bypass Cloudflare, with retry on failure."""
    driver = setup_driver()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 8)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            time.sleep(0.3)
            html = driver.page_source
            time.sleep(3) # maximum 20 requests per minute to avoid being blocked
            soup = BeautifulSoup(html, 'html.parser')
            if soup is not None:
                return soup
        except Exception as e:
            print(f"Failed to load {url} (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    print(f"Giving up on {url} after {max_retries} attempts.")
    return None


def get_data(url):
    parsed_html = get_fbref_data_selenium(url=url)
    if parsed_html is None:
        return []
    comments = parsed_html.find_all(string=lambda text: isinstance(text, Comment))
    tables = []
    for c in comments:
        if '<table' in c:
            table_html = BeautifulSoup(c, 'html.parser')
            tables = table_html.find_all('table')
    return tables


def get_table_data(url):
    parsed_html = get_fbref_data_selenium(url=url)
    if parsed_html is None:
        return None
    tables = parsed_html.find_all('table')
    return tables[0] if tables else None


def get_matches_data(player):
    tables = []
    for l in player.matches_links:
        t = get_table_data(l)
        if t is not None:
            tables.append(t)
    matches = []
    match_stat_set = set()
    for t in tables:
        if t is None or t.tbody is None:
            continue
        for row in t.tbody.find_all('tr'):
            data = {}
            class_name = row.get('class')
            if class_name != None and len(class_name) > 0 and 'unused_sub' not in class_name:
                continue
            columns = row.find_all('td') + row.find_all('th')
            for c in columns:
                data_stat = c.get('data-stat')
                match_stat_set.add(data_stat)
                if data_stat in ['date', 'round', 'comp', 'opponent', 'squad']:
                    for i in range(len(c.contents)):
                        a_html = BeautifulSoup(str(c.contents[i]), 'html.parser')
                        a = a_html.find_all('a')
                        if len(a) > 0:
                            if len(a[0].contents) > 0:
                                data[data_stat] = a[0].contents[0]
                elif data_stat == 'match_report':
                    continue
                else:
                    if len(c.contents) == 0:
                        continue
                    data[data_stat] = c.contents[0]
            if 'comp' not in data or data['comp'] != 'Premier League':
                continue
            match = MatchData()
            match.date = data['date']
            match.round = data['round']
            match.comp = data['comp']
            match.data = data
            matches += [match]
    player.matches = matches
    player.match_stat_set = match_stat_set


def get_epl_players(url):
    tables = get_data(url) # manually run through the seasons "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
    # tables = get_data("https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats")
    table = tables[0]
    players = {}
    stat_names = set()
    for row in table.tbody.find_all('tr'):
        class_name = row.get('class')
        if class_name != None and len(class_name) > 0:
            continue
        columns = row.find_all('td')
        base_url = ""
        matches_link = ""
        player_id = ""
        stats = {}
        for c in columns:
            data_stat = c.get('data-stat')
            if data_stat == 'player':
                a_html = BeautifulSoup(str(c.contents[0]), 'html.parser')
                a = a_html.find_all('a')
                base_url = "https://fbref.com" + a[0].get('href')
                link = a[0].get('href')
                pieces = link.split('/')
                player_id = pieces[3]
                stats[data_stat] = a[0].contents[0]
                stat_names.add(data_stat)
            elif data_stat == 'squad':
                a_html = BeautifulSoup(str(c.contents[0]), 'html.parser')
                a = a_html.find_all('a')
                stats[data_stat] = a[0].contents[0]
                stat_names.add(data_stat)
            elif data_stat == 'minutes':
                mins = c.contents[0]
                if ',' in mins:
                    mins = int(mins.replace(',', ''))
                stats[data_stat] = mins
                stat_names.add(data_stat)
            elif data_stat == "matches":
                a_html = BeautifulSoup(str(c.contents[0]), 'html.parser')
                a = a_html.find_all('a')
                matches_link = "https://fbref.com" + a[0].get('href')
                # print(f"Matches link for player {player_id}: {matches_link}")
            elif data_stat == "nationality":
                continue
            else:
                if c.contents:
                    stats[data_stat] = c.contents[0]
                    stat_names.add(data_stat)
                else:
                    stats[data_stat] = None
                    stat_names.add(data_stat)
        player = PlayerData()
        if player_id in players:
            player = players[player_id]
        player.base_url = base_url
        if len(player.matches_links) == 0:
            player.matches_links += [matches_link]
        player.data += [stats]
        players[player_id] = player
    return players, stat_names


def get_squad_stats(season):
    soup = get_fbref_data_selenium(url=f"https://fbref.com/en/comps/9/{season}/Premier-League-Stats")
    # Find the squad stats table by class name
    # soup = get_fbref_data_selenium(url=f'https://fbref.com/en/comps/9/{season}/stats/{season}-Premier-League-Stats')
    table = soup.find('table', id='stats_squads_standard_for')
    if not table:
        print("Squad stats table not found.")
        return []

    squads = []
    thead_rows = table.thead.find_all('tr')
    header_row = thead_rows[-1] 
    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    for row in table.tbody.find_all('tr'):
        squad_data = SquadData()
        cells = row.find_all(['td', 'th'])
        for idx, cell in enumerate(cells):
            value = cell.get_text(strip=True)
            if headers[idx].lower() == 'squad':
                squad_data.squad = value
            squad_data.stats[headers[idx]] = value
        squads.append(squad_data)
    return squads


def write_squad_stats_to_files(squads, season):
    folder = f"new_data/{season}/fbref_squads"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for squad in squads:
        file_path = f"{folder}/{squad.squad}.csv"
        with open(file_path, "w", newline='', encoding='utf-8') as outf:
            writer = csv.DictWriter(outf, fieldnames=list(squad.stats.keys()))
            writer.writeheader()
            writer.writerow(squad.stats)


def player_csv_maker():
    # must change folder paths manually
    seasons = [
        # {"season": "2023-2024", "url": "https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats"},
        # {"season": "2024-2025", "url": "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats"},
        {"season": "2025-2026", "url": "https://fbref.com/en/comps/9/stats/Premier-League-Stats"}
    ]
    for season_data in seasons:
        season = season_data["season"]
        url = season_data["url"]

        players, stats = get_epl_players(url)
        for id, player in players.items():
            player_file = f'new_data/{season}/fbref/{id}.csv'
            get_matches_data(player)
            with open(player_file, 'w', newline='', encoding='utf-8') as outf:
                writer = csv.DictWriter(outf, fieldnames=list(player.match_stat_set))
                writer.writeheader()
                for match in player.matches:
                    writer.writerow(match.data)
            print(f"Written data for player {id} to {player_file}")

        with open(f'new_data/{season}/fbref_overview.csv', 'w', newline='', encoding='utf-8') as outf:
            writer = csv.DictWriter(outf, fieldnames=list(stats))
            writer.writeheader()
            for id, player in players.items():
                for data in player.data:
                    writer.writerow(data)


def main():
    player_csv_maker()
    # season = "2025-2026"
    # squads = get_squad_stats(season=season)
    # write_squad_stats_to_files(squads, season=season)
    print("Data collection complete.")


if __name__ == '__main__':
    main()