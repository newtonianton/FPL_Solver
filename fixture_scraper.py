import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from final_fbref_scraper import setup_driver

# URL of the Premier League fixtures page
FIXTURE_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

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

# Folder to save fixture files
FIXTURE_FOLDER = "fixture_files"
if not os.path.exists(FIXTURE_FOLDER):
    os.makedirs(FIXTURE_FOLDER)

def scrape_fixtures(url):
    # Send a GET request to the URL
    driver = setup_driver()
    driver.get(url=url)

    # Parse the HTML content
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the fixtures table
    table = soup.find("table", {"id": "sched_2025-2026_9_1"})
    if not table:
        print("Could not find the fixtures table on the page.")
        return

    # Extract table headers
    headers = table.find("thead").find_all("th")
    headers = [th.text.strip() for th in headers]

    # Extract table rows
    rows = []
    for row in table.find("tbody").find_all("tr"):
        cols = [td.text.strip() for td in row.find_all(["th", "td"])]
        if cols:
            rows.append(cols)

    # Convert to a DataFrame
    fixtures_df = pd.DataFrame(rows, columns=headers)

    # Clean and process the DataFrame
    fixtures_df = fixtures_df.dropna(how="all")   # Convert gameweek to integer
    fixtures_df.sort_values("Wk", inplace=True)
    fixtures_df = fixtures_df[fixtures_df['Wk'].apply(lambda x: str(x).isdigit())]
    print(fixtures_df['Wk'].unique())
    return fixtures_df

def save_fixtures_by_gameweek(fixtures_df):
    # Group fixtures by gameweek and save each gameweek to a separate CSV file
    for gw, group in fixtures_df.groupby("Wk"):
        result = {}
        result['team'] = group['Home'].apply(lambda x: teams_dict.get(x, x)).tolist() + group['Away'].apply(lambda x: teams_dict.get(x, x)).tolist()
        result['opponent'] = group['Away'].apply(lambda x: teams_dict.get(x, x)).tolist() + group['Home'].apply(lambda x: teams_dict.get(x, x)).tolist()
        result['was_home'] = [True] * 10 + [False] * 10
        # print(result)
        filename = os.path.join(FIXTURE_FOLDER, f"gameweek_{gw}.csv")
        pd.DataFrame(result).to_csv(filename, index=False)
        print(f"Saved fixtures for Gameweek {gw} to {filename}")

def main():
    print("Scraping fixtures...")
    fixtures_df = scrape_fixtures(FIXTURE_URL)
    if fixtures_df is not None:
        # print("Saving fixtures by gameweek...")
        save_fixtures_by_gameweek(fixtures_df)
        # print("Fixture scraping and saving complete.")

if __name__ == "__main__":
    main()