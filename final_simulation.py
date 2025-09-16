import numpy as np
import pandas as pd
import pulp
import os
import csv
from final_fbref_scraper import load_player_fbref_id_map

class FPLSimulator:
    def compute_player_covariance(self, gw_list):
        """
        Compute covariance matrix of simulated points for all players over gw_list.
        Returns: DataFrame with index/columns as FBRef_ID, values as covariances.
        """
        players = self.players_df[self.players_df['gameweek'].isin(gw_list)]['Player_Name_fbref'].unique()
        sim_matrix = {}
        player_map = load_player_fbref_id_map()
        for p in players:
            fbref_id = player_map.get(p, p)
            sim_matrix[fbref_id] = self.simulate_player_horizon(fbref_id, gw_list)
        sim_df = pd.DataFrame(sim_matrix)
        cov_df = sim_df.cov()
        return cov_df
    def __init__(self, players_df, n_sims=10000, random_state=42):
        """
        player_df: DataFrame with columns:
            - Player_Name_fbref, Player_Name_fbref, position, team, price, etc.
            - mean_points_cond, std_points_cond, start_prob
            - expected_points_uncond, resid_std_scaled
        n_sims: Number of Monte Carlo samples
        """
        self.players_df = players_df
        self.n_sims = n_sims
        self.rng = np.random.default_rng(random_state)
    
    def simulate_player_points(self, row):
        """
        Simulate points for a single player based on their statistics.
        """
        p = row['start_prob']
        mu = row['mean_points_cond']
        sigma = row['std_points_cond']

        starts = self.rng.binomial(1, p, self.n_sims) # 1 if starts, 0 if not Bernoulli

        points_if_played = self.rng.normal(mu, sigma, self.n_sims)
        points_if_played = np.maximum(0, points_if_played)
        points = starts * points_if_played
        return points

    def simulate_player_horizon(self, player_id, gw_list):
        """Simulate player points over a list of gameweeks."""
        sims = []
        for gw in gw_list:
            row = self.players_df[(self.players_df['Player_Name_fbref'] == player_id) & (self.players_df['gameweek'] == gw)]
            if row.empty:
                continue
            sims.append(self.simulate_player_points(row.iloc[0]))
        if sims:
            return np.sum(sims, axis=0)
        else:
            return np.zeros(self.n_sims)
    
    def simulate_squad_horizon(self, xi_ids_per_gw, captain_per_gw=None, injured_players=None):
        """
        Simulate total points for a squad across multiple gameweeks.

        Parameters
        ----------
        xi_ids_per_gw : dict
            Dictionary {gw: [Player_Name_fbrefs]} representing the starting XI for each gameweek.
        captain_per_gw : dict, optional
            Dictionary {gw: Player_Name_fbref} representing captain choice for each gameweek.

        Returns
        -------
        np.ndarray
            1D array of total squad points across all GWs for each simulation.
        """
        total_points = np.zeros(self.n_sims)

        injured_set = set(injured_players) if injured_players else set()
        for gw, xi_ids in xi_ids_per_gw.items():
            gw_points = np.zeros(self.n_sims)
            for p_id in xi_ids:
                if p_id in injured_set:
                    continue
                player_row = self.players_df[(self.players_df['Player_Name_fbref'] == p_id) & (self.players_df['gameweek'] == gw)]
                gw_points += self.simulate_player_points(player_row.iloc[0])

            # Apply captain multiplier if provided
            if captain_per_gw and gw in captain_per_gw:
                cap_id = captain_per_gw[gw]
                if cap_id not in injured_set:
                    cap_row = self.players_df[(self.players_df['Player_Name_fbref'] == cap_id) & (self.players_df['gameweek'] == gw)]
                    gw_points += self.simulate_player_points(cap_row.iloc[0])
            
            total_points += gw_points

        return total_points


    def optimise_squad_with_transfers(
        self,
        gw_list,
        budget=1000,
        fixed_players=None,
        fixed_in_xi=False,
        initial_squad=None,  # Accepts list of names or FBREF IDs
        initial_free_transfers=1,
        transfer_cost=4.0,
        clash_pairs=None,
        clash_penalty=10.5,
        risk_lambda=0.75,
        cov_matrix=None,
        injured_players=None,
    ):
        '''
        Optimise a squad for multiple GWs (fixed horizon, no transfers yet).
        
        Parameters:
        - gw_list: list of gameweeks to include in horizon
        - budget: total budget (default 1000 = £100m)
        - n_players: total squad size (default 15)

        Returns:
        - dict with squad, XI, bench, stats on points distribution
        '''
        if fixed_players is None:
            fixed_players = []
        if injured_players is None:
            injured_players = []
        injured_set = set(injured_players)

        df = self.players_df.copy()
        df = df[df['gameweek'].isin(gw_list)].copy()
        players = sorted(df['Player_Name_fbref'].unique().tolist())
        base = (df.sort_values(['Player_Name_fbref', 'gameweek'], ascending=False).drop_duplicates('Player_Name_fbref'))
        price = base.set_index('Player_Name_fbref')['value'].to_dict()
        team = base.set_index('Player_Name_fbref')['team'].to_dict()
        pos = base.set_index('Player_Name_fbref')['position'].to_dict()
        epts = (df.set_index(['Player_Name_fbref','gameweek'])['expected_points_uncond']
          .astype(float).to_dict())

        x = pulp.LpVariable.dicts('squad', [(p, gw) for p in players for gw in gw_list], 0, 1, cat='Binary')
        y = pulp.LpVariable.dicts('xi', [(p, gw) for p in players for gw in gw_list], 0, 1, cat='Binary')
        c = pulp.LpVariable.dicts('captain', [(p, gw) for p in players for gw in gw_list], 0, 1, cat='Binary')

        z_in  = pulp.LpVariable.dicts('tr_in',  [(p, gw) for p in players for gw in gw_list[1:]], 0, 1, cat='Binary')
        z_out = pulp.LpVariable.dicts('tr_out', [(p, gw) for p in players for gw in gw_list[1:]], 0, 1, cat='Binary')
        extra = pulp.LpVariable.dicts('extra_tr', gw_list[1:], lowBound=0, cat='Continuous')
        # Track stored free transfers per GW (rolling, max 5)
        stored_ft = {}
        for gw in gw_list:
            stored_ft[gw] = pulp.LpVariable(f'stored_ft_{gw}', lowBound=0, upBound=5, cat='Integer')

        # Clash variables (only if provided). Each w=1 if BOTH players in the clash pair start that GW.
        if clash_pairs:
            w = pulp.LpVariable.dicts('clash',
                                      [(p1, p2, gw) for (p1, p2, gw) in clash_pairs],
                                      0, 1, cat='Binary')
        else:
            w = {}


        prob = pulp.LpProblem("FPL_Squad_Horizon_Optimization", pulp.LpMaximize)

        # Base points (including captain double count by adding c again)
        base_points = pulp.lpSum(
            (0.0 if p in injured_set else epts.get((p, gw), 0.0)) * (y[(p, gw)] + c[(p, gw)])
            for p in players for gw in gw_list
        )
        transfer_cost_term = pulp.lpSum(transfer_cost * extra[gw] for gw in gw_list[1:])
        clash_term = pulp.lpSum(w[(p1, p2, gw)] for (p1, p2, gw) in clash_pairs) if clash_pairs else 0

        # Risk penalty (variance) for whole squad
        risk_term = 0
        if risk_lambda > 0 and cov_matrix is not None:
            for gw in gw_list:
                for i in players:
                    for j in players:
                        cov_ij = cov_matrix.loc[i, j] if i in cov_matrix.index and j in cov_matrix.columns else 0.0
                        risk_term += risk_lambda * cov_ij * x[(i, gw)] * x[(j, gw)]

        # Objective: maximize expected points minus transfer costs, clash penalties, and risk penalty
        prob += base_points - transfer_cost_term - clash_penalty * clash_term - risk_term

        # 1) Initial squad: allow up to initial_free_transfers changes before first GW
        # Model exact number of initial transfers with integer variable (each transfer produces two unit changes: one out, one in)
        init_changes = None
        if initial_squad:
            missing = [p for p in initial_squad if p not in players]
            if missing:
                raise ValueError(f"Initial squad not in data: {missing}")
            gw0 = gw_list[0]
            for p in players:
                # Fix inclusion variable via conditional equality (player either stays or is transferred in)
                pass  # we will not fix here; the optimizer chooses subject to transfer budget below
            transfer_out_terms = [1 - x[(p, gw0)] for p in initial_squad]
            transfer_in_terms = [x[(p, gw0)] for p in players if p not in initial_squad]
            # Integer variable for number of transfers actually made
            init_changes = pulp.LpVariable('initial_transfers_used', lowBound=0, upBound=initial_free_transfers, cat='Integer')
            # 2 * init_changes == sum(out + in) because each real transfer produces one out and one in term=1
            prob += 2 * init_changes == pulp.lpSum(transfer_out_terms + transfer_in_terms)
            # Cannot exceed available free transfers
            prob += init_changes <= initial_free_transfers
        else:
            # If no initial squad provided, treat initial transfers used as zero via a dummy var
            init_changes = pulp.LpVariable('initial_transfers_used', lowBound=0, upBound=0, cat='Integer')

        # 2) Fixed players across the whole horizon (outside the per-GW loop)
        for p in fixed_players:
            for gw in gw_list:
                prob += x[(p, gw)] == 1
                if fixed_in_xi:
                    prob += y[(p, gw)] == 1

        # Constraints
        for gw in gw_list:
            # Squad size and budget per GW
            prob += pulp.lpSum(x[(p, gw)] for p in players) == 15
            prob += pulp.lpSum(price.get(p, 0.0) * x[(p, gw)] for p in players) <= budget

            # Max 2 per team (would be 3, but 2 for diversification purposes)
            for t in set(team.values()):
                team_players = [p for p in players if team.get(p) == t]
                prob += pulp.lpSum(x[(p, gw)] for p in team_players) <= 2

            # Squad position requirements (fixed by rules)
            squad_min = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
            for position, count in squad_min.items():
                pos_players = [p for p in players if pos.get(p) == position]
                prob += pulp.lpSum(x[(p, gw)] for p in pos_players) == count

            # XI constraints per GW
            prob += pulp.lpSum(y[(p, gw)] for p in players) == 11
            for position, min_count in {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}.items():
                pos_players = [p for p in players if pos.get(p) == position]
                prob += pulp.lpSum(y[(p, gw)] for p in pos_players) >= min_count

            # Bench size always 4
            prob += pulp.lpSum(x[(p, gw)] for p in players) - pulp.lpSum(y[(p, gw)] for p in players) == 4

            # Linking and captain per GW
            for p in players:
                prob += y[(p, gw)] <= x[(p, gw)]
                prob += c[(p, gw)] <= y[(p, gw)]
            prob += pulp.lpSum(c[(p, gw)] for p in players) == 1
        
        # Clash linearization constraints
        if clash_pairs:
            for (p1, p2, gw) in clash_pairs:
                # w <= y_p1; w <= y_p2; w >= y_p1 + y_p2 - 1
                prob += w[(p1, p2, gw)] <= y[(p1, gw)]
                prob += w[(p1, p2, gw)] <= y[(p2, gw)]
                prob += w[(p1, p2, gw)] >= y[(p1, gw)] + y[(p2, gw)] - 1

        # Transfer links between GWs
        for i in range(1, len(gw_list)):
            gw_prev, gw = gw_list[i-1], gw_list[i]
            for p in players:
                # in: 0 -> 1 z_in = max(0, x_t - x_{t-1})
                prob += z_in[(p, gw)] >= x[(p, gw)] - x[(p, gw_prev)]
                prob += z_in[(p, gw)] <= x[(p, gw)]
                prob += z_in[(p, gw)] <= 1 - x[(p, gw_prev)]
                # out: 1 -> 0 z_out = max(0, x_{t-1} - x_t)
                prob += z_out[(p, gw)] >= x[(p, gw_prev)] - x[(p, gw)]
                prob += z_out[(p, gw)] <= x[(p, gw_prev)]
                prob += z_out[(p, gw)] <= 1 - x[(p, gw)]

            # Transfers in must equal transfers out
            prob += pulp.lpSum(z_in[(p, gw)] for p in players) == pulp.lpSum(z_out[(p, gw)] for p in players)

            # Transfers must be position-matched
            for position in ["GK", "DEF", "MID", "FWD"]:
                pos_players = [p for p in players if pos.get(p) == position]
                prob += pulp.lpSum(z_in[(p, gw)] for p in pos_players) == pulp.lpSum(z_out[(p, gw)] for p in pos_players)

            transfers_gw = pulp.lpSum(z_in[(p, gw)] for p in players)

            # Free transfer accumulation logic
            if i == 1:
                # Initial stored free transfers AFTER using init_changes in first considered GW
                # Remaining stored free transfers carried forward (before adding the +1 for the next GW happens via recurrence):
                # stored_ft[gw_prev] = initial_free_transfers - init_changes
                prob += stored_ft[gw_prev] == initial_free_transfers - init_changes
            # Each week: stored_ft = min(5, prev + 1 - transfers_gw)
            prob += stored_ft[gw] <= stored_ft[gw_prev] + 1 - transfers_gw
            prob += stored_ft[gw] >= stored_ft[gw_prev] + 1 - transfers_gw
            prob += stored_ft[gw] <= 5
            prob += stored_ft[gw] >= 0

            # Transfer cost: only pay for transfers above stored free transfers
            prob += extra[gw] >= transfers_gw - stored_ft[gw_prev] - 1
            prob += extra[gw] >= 0

        # Solve and check status
        result_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status_str = pulp.LpStatus[prob.status]
        if status_str != "Optimal":
            print(f"Warning: Solver status is {status_str}. Solution may violate constraints or be incomplete.")

        squad_ids_by_gw = {gw: [p for p in players if x[(p, gw)].value() == 1] for gw in gw_list}
        xi_ids = {gw: [p for p in players if y[(p, gw)].value() == 1] for gw in gw_list}
        captain_choices = {}
        for gw in gw_list:
            found = [p for p in players if c[(p, gw)].value() == 1]
            if found:
                captain_choices[gw] = found[0]
            else:
                captain_choices[gw] = None
                print(f"Warning: No captain assigned for GW {gw} (solver issue or infeasibility)")

        # Simulate XI distribution
        xi_points = self.simulate_squad_horizon(xi_ids, captain_per_gw=captain_choices)

        transfers_by_gw = {}
        free_transfers_by_gw = {}
        # Initial GW transfers (actual number used)
        gw0 = gw_list[0]
        transfers_by_gw[gw0] = int(init_changes.value()) if init_changes is not None else 0
        # Free transfers left after initial changes
        free_transfers_by_gw[gw0] = max(0, initial_free_transfers - transfers_by_gw[gw0])

        for i in range(1, len(gw_list)):
            gw = gw_list[i]
            transfers_by_gw[gw] = int(sum((z_in[(p, gw)].value() or 0) for p in players))
            free_transfers_by_gw[gw] = int(stored_ft[gw].value()) if stored_ft[gw].value() is not None else None

        return {
            "squad_ids": squad_ids_by_gw,
            "xi_ids": xi_ids,
            "captains": captain_choices,
            "mean": np.mean(xi_points),
            "std": np.std(xi_points),
            "quantiles": np.percentile(xi_points, [10, 50, 90]),
            "free_transfers": free_transfers_by_gw,
            "transfers": transfers_by_gw,
            # Preserve the originally supplied initial squad (order as given) for accurate first GW diff
            "initial_squad_original": initial_squad if initial_squad else None,
            "injured_players": injured_players,
        }

def pretty_print_result(result, players_df):
    """Print initial squad, then transfers and captain each week."""
    # Meta
    meta = (
        players_df[[
            'Player_Name_fbref', 'position', 'team', 'value'
        ]]
        .drop_duplicates('Player_Name_fbref')
        .set_index('Player_Name_fbref')
    )
    # Opponent lookup per (player, gw)
    opp_lookup = (
        players_df[['Player_Name_fbref', 'gameweek', 'opponent']]
        .dropna(subset=['opponent'])
        .assign(gameweek=lambda d: d['gameweek'].astype(int))
        .set_index(['Player_Name_fbref', 'gameweek'])['opponent']
        .to_dict()
    )

    # Build expected points lookup per (player, gw)
    epts = (
        players_df[[
            'Player_Name_fbref', 'gameweek', 'expected_points_uncond'
        ]]
        .dropna(subset=['expected_points_uncond'])
        .assign(gameweek=lambda d: d['gameweek'].astype(int))
        .set_index(['Player_Name_fbref', 'gameweek'])['expected_points_uncond']
        .to_dict()
    )

    mean = float(result['mean'])
    std = float(result['std'])
    q10, q50, q90 = [float(x) for x in result['quantiles']]
    print(f"Expected total points: {mean:.2f} ± {std:.2f}  (p10={q10:.2f}, median={q50:.2f}, p90={q90:.2f})\n")

    # Sorted GWs
    gw_sorted = sorted(result['squad_ids'].keys())
    if not gw_sorted:
        print("No gameweeks in result.")
        return

    # Initial squad (post-optimisation for first GW horizon) and original supplied squad
    gw0 = gw_sorted[0]
    initial = result['squad_ids'][gw0]
    original_initial = result.get('initial_squad_original') or initial
    original_initial_set = set(original_initial)
    print(f"Initial squad (GW {gw0}): (size={len(initial)})")
    pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    def sort_key(name):
        pos = meta.loc[name]['position'] if name in meta.index else ''
        return (pos_order.get(pos, 9), name)

    injured_set = set(result.get('injured_players') or [])
    for name in sorted(initial, key=sort_key):
        ep = epts.get((name, int(gw0)), None)
        ep_str = f"{ep:.2f}" if ep is not None and not pd.isna(ep) else "NA"
        opp = opp_lookup.get((name, int(gw0)), "?")
        if name in meta.index:
            pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
            injury_tag = " (INJURED)" if name in injured_set else ""
            adj_ep = "0.00" if name in injured_set else ep_str
            print(f"  - {name} [{pos}] ({team}) · £{price} · E[pts]={adj_ep} · vs {opp}{injury_tag}")
        else:
            print(f"  - {name} · E[pts]={ep_str} · vs {opp}")
    print("")

    # Weekly transfers and captain
    # For the first GW we want to show transfers relative to the originally supplied squad
    prev_set = original_initial_set
    for gw in gw_sorted:
        curr = result['squad_ids'][gw]
        curr_set = set(curr)
        ins = sorted(curr_set - prev_set, key=sort_key)
        outs = sorted(prev_set - curr_set, key=sort_key)
        cap = result.get('captains', {}).get(gw, None)

        xi = result.get('xi_ids', {}).get(gw, [])
        bench = [name for name in curr if name not in set(xi)]
        free_transfers = result.get('free_transfers', {}).get(gw, None)
        transfers = result.get('transfers', {}).get(gw, None)
        print(f"GW {gw}: Squad size={len(curr)}, XI size={len(xi)}, Bench size={len(bench)}, Free transfers left={free_transfers}, Transfers made={transfers}")
        if ins:
            print("  Transfers in:")
            for name in ins:
                ep = epts.get((name, int(gw)), None)
                ep_str = f"{ep:.2f}" if ep is not None and not pd.isna(ep) else "NA"
                opp = opp_lookup.get((name, int(gw)), "?")
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
                    injury_tag = " (INJURED)" if name in injured_set else ""
                    adj_ep = "0.00" if name in injured_set else ep_str
                    print(f"    + {name} [{pos}] ({team}) · £{price} · E[pts]={adj_ep} · vs {opp}{injury_tag}")
                else:
                    print(f"    + {name} · E[pts]={ep_str} · vs {opp}")
        if outs:
            print("  Transfers out:")
            for name in outs:
                ep = epts.get((name, int(gw)), None)
                ep_str = f"{ep:.2f}" if ep is not None and not pd.isna(ep) else "NA"
                opp = opp_lookup.get((name, int(gw)), "?")
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
                    injury_tag = " (INJURED)" if name in injured_set else ""
                    adj_ep = "0.00" if name in injured_set else ep_str
                    print(f"    - {name} [{pos}] ({team}) · £{price} · E[pts]={adj_ep} · vs {opp}{injury_tag}")
                else:
                    print(f"    - {name} · E[pts]={ep_str} · vs {opp}")
        if not ins and not outs:
            print("  No transfers")

        print(f"  Captain: {cap}\n")

        if xi:
            print("  Starting XI:")
            for name in sorted(xi, key=sort_key):
                ep = epts.get((name, int(gw)), None)
                ep_str = f"{ep:.2f}" if ep is not None and not pd.isna(ep) else "NA"
                opp = opp_lookup.get((name, int(gw)), "?")
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']
                    injury_tag = " (INJURED)" if name in injured_set else ""
                    adj_ep = "0.00" if name in injured_set else ep_str
                    print(f"    - {name} [{pos}] ({team}) · E[pts]={adj_ep} · vs {opp}{injury_tag}")
                else:
                    print(f"    - {name} · E[pts]={ep_str} · vs {opp}")
        print("")

        if bench:
            bench_sorted = sorted(
                bench,
                key=lambda n: epts.get((n, int(gw)), 0.0),
                reverse=True
            )
            print("  Bench (by expected points):")
            for name in bench_sorted:
                ep = epts.get((name, int(gw)), None)
                ep_str = f"{ep:.2f}" if ep is not None and not pd.isna(ep) else "NA"
                opp = opp_lookup.get((name, int(gw)), "?")
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']
                    injury_tag = " (INJURED)" if name in injured_set else ""
                    adj_ep = "0.00" if name in injured_set else ep_str
                    print(f"    - {name} [{pos}] ({team}) · E[pts]={adj_ep} · vs {opp}{injury_tag}")
                else:
                    print(f"    - {name} · E[pts]={ep_str} · vs {opp}")
        print("")
        # Update previous squad set for next GW (week-to-week diff after first GW)
        prev_set = curr_set


def full_simulation(
    gw_list,
    budget=1000,
    fixed_players=None,
    fixed_in_xi=False,
    initial_squad=None,
    initial_free_transfers=1,
    injured_players=None,
):
    ML_output = pd.read_csv("ML_model_predictions.csv")
    # Check for duplicate player IDs/names (only warn for duplicates in unique player list)
    unique_ids = ML_output['Player_Name_fbref'].unique().tolist()
    if len(unique_ids) != len(set(unique_ids)):
        print("Warning: Duplicate player IDs found in unique player list!")

    sim_engine = FPLSimulator(ML_output, n_sims=10000)
    df = ML_output[ML_output['gameweek'].isin(gw_list)].copy()

    # Build clash pairs: (attacker, defender, gw) unique
    clash_pairs_set = set()
    for gw in gw_list:
        gw_df = df[df['gameweek'] == gw]
        # Attackers vs defensive opponents
        atk_df = gw_df[gw_df['position'].isin(['FWD','MID'])]
        def_df = gw_df[gw_df['position'].isin(['DEF','GK'])]
        # Iterate
        for _, a in atk_df.iterrows():
            opp_team = a['opponent']
            # Defensive players whose team is the attacker's opponent
            clash_defs = def_df[def_df['team'] == opp_team]
            for _, d in clash_defs.iterrows():
                clash_pairs_set.add((a['Player_Name_fbref'], d['Player_Name_fbref'], gw))
    clash_pairs = list(clash_pairs_set)

    # Example optimisation with clash penalty
    result = sim_engine.optimise_squad_with_transfers(
        gw_list=gw_list,
        budget=budget,
        fixed_players=fixed_players if fixed_players is not None else [],
        fixed_in_xi=fixed_in_xi,
        initial_squad=initial_squad,
        initial_free_transfers=initial_free_transfers,
        clash_pairs=clash_pairs,
        injured_players=injured_players,
    )

    # Post-solve checks for constraint violations
    print("\n--- Constraint Diagnostics ---")
    for gw in sorted(result['squad_ids'].keys()):
        squad = result['squad_ids'][gw]
        xi = result['xi_ids'][gw]
        bench = [name for name in squad if name not in set(xi)]
        errors = []
        if len(squad) != 15:
            errors.append(f"Squad size={len(squad)} (should be 15)")
        if len(xi) != 11:
            errors.append(f"XI size={len(xi)} (should be 11)")
        if len(bench) != 4:
            errors.append(f"Bench size={len(bench)} (should be 4)")
        # Check for duplicate players in squad/XI/bench
        if len(set(squad)) != len(squad):
            errors.append("Duplicate players in squad")
        if len(set(xi)) != len(xi):
            errors.append("Duplicate players in XI")
        if len(set(bench)) != len(bench):
            errors.append("Duplicate players in bench")
        if errors:
            print(f"GW {gw}: Constraint violation(s): {', '.join(errors)}")
        else:
            print(f"GW {gw}: All constraints satisfied.")
    print("--- End Diagnostics ---\n")

    pretty_print_result(result, ML_output)


if __name__ == "__main__":
    gameweeks = [5, 6, 7, 8, 9, 10, 11]
    full_simulation(
        gw_list=gameweeks,
        budget=1000,
        fixed_players=[],
        fixed_in_xi=True,
        initial_squad=["Erling Haaland",
                       "Nick Pope",
                       "Jurriën Timber",
                       "Riccardo Calafiori",
                       "Maxence Lacroix",
                       "David Brooks",
                       "Antoine Semenyo",
                       "Bruno Borges Fernandes",
                       "Bryan Mbeumo",
                       "Lyle Foster",
                       "Chris Wood",
                       "Mads Hermansen",
                       "Declan Rice",
                       "John Stones",
                       "Dan Burn"],
        initial_free_transfers=2,
        # Example: add names whose points should be treated as 0 due to injury
        injured_players=[
            "John Stones",
        ]
    )





