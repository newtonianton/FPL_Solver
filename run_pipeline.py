import subprocess
import sys
from final_ML_model import ML_predictions
from final_simulation import full_simulation
import os

# List of scripts to run in order for a new gameweek
UPDATE_DATA_SCRIPTS = [
    "Fantasy-Premier-League/global_scraper.py",
    "final_cleaning_all_players.py",
    "final_df_merger.py",
    "final_fbref_scraper.py"
]


def run_script(script):
    print(f"\n--- Running {script} ---")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"[ERROR] {script} STDERR:\n{result.stderr}")
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
    return result.returncode


def update():
    for script in UPDATE_DATA_SCRIPTS:
        code = run_script(script)
        if code != 0:
            print(f"Pipeline stopped due to error in {script}.")
            break
    else:
        print("\nPipeline completed successfully.")


if __name__ == "__main__":
    # update()
    gameweeks = [5, 6, 7, 8, 9, 10, 11]
    # ML_predictions(gameweeks=gameweeks)
    # print("\nML Predictions completed.")
    full_simulation(
        gw_list=gameweeks,
        budget=1000,
        fixed_players=[],
        fixed_in_xi=True,
        initial_free_transfers=2
    )
    print("\nFull simulation completed.")


