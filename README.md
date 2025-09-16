# FPL_Solver
Full FPL pipeline: cleans FBRef + FPL data, predicts player points by combining start probabilities and conditional points. Runs multi-gameweek optimisation with transfers and fixed player options, and utilises Monte Carlo simulations to give outcomes with quantiles. 

# Core components
- Data ingestion: scrape/merge FBRef + FPL, normalize teams, dedupe matches, filter low minutes.
- Availability: leak‑safe per‑season rolling starts -> start_prob (shifted windows, optional league‑only).
- Features: rolling (3/7) performance stats + pre‑match fixture features (home, team/opponent strength).
- ML models: per‑position XGBoost (option NN) predicting conditional points (if player starts).
- Uncertainty: per‑player residual std + start_prob -> unconditional mean & variance (law of total variance).
- Optimiser: PuLP MILP over multi‑GW horizon (squad, XI, captain, transfers, penalties, constraints).
- Simulation: Monte Carlo of starts + conditional points to get distribution (mean, std, quantiles).
- Reporting: per‑GW transfers, captaincy, XI, bench ordering by expected points; player prediction CSV.
- ID hygiene: manual FBRef ID overrides, duplicate match collapse, safe merges.

# Outputs (See example_output.md)
- Expected points per player per future GW (conditional & unconditional).
- Optimised multi‑GW squad plan (squad_ids, xi_ids, captains, transfers).
- Risk metrics (distribution quantiles).
- Transparent intermediate artefacts (residual stats, per‑player uncertainty).

# Pipeline

## Installation

1. Create a Python virtual environment (recommended):
	```cmd
	python -m venv .venv
	.venv\Scripts\activate
	```
2. Install required packages:
	```cmd
	pip install -r requirements.txt
	```

## Pipeline
Prep: fixture_scraper.py, install chromedriver
global_scraper.py (Fantasy-Premier-League/global_scraper.py)
final_fbref_scraper.py
final_df_merger.py
final_cleaning_all_players.py
final_ML_model.py
final_simulation.py

# Credit
- Credit to https://github.com/vaastav/Fantasy-Premier-League for global_scraper.py and FPL Data
