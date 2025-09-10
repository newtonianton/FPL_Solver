import numpy as np
import pandas as pd
import pulp

class FPLSimulator:
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
    
    def simulate_squad_horizon(self, xi_ids_per_gw, captain_per_gw=None):
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

        for gw, xi_ids in xi_ids_per_gw.items():
            gw_points = np.zeros(self.n_sims)
            for p_id in xi_ids:
                player_row = self.players_df[(self.players_df['Player_Name_fbref'] == p_id) &
                                             (self.players_df['gameweek'] == gw)]
                gw_points += self.simulate_player_points(player_row.iloc[0])

            # Apply captain multiplier if provided
            if captain_per_gw and gw in captain_per_gw:
                cap_id = captain_per_gw[gw]
                cap_row = self.players_df[(self.players_df['Player_Name_fbref'] == cap_id) &
                                          (self.players_df['gameweek'] == gw)]
                gw_points += self.simulate_player_points(cap_row.iloc[0])
            
            total_points += gw_points

        return total_points


    def optimise_squad_with_transfers(
        self,
        gw_list,
        budget=1000,
        n_players=15,
        fixed_players=None,
        fixed_in_xi=False,
        initial_squad=None,
        free_transfers_per_gw=1,
        transfer_cost=4.0,
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


        prob = pulp.LpProblem("FPL_Squad_Horizon_Optimization", pulp.LpMaximize)

        # Objective: maximize expected total points over horizon
        prob += pulp.lpSum(epts.get((p, gw), 0.0) * (y[(p,gw)] + c[(p,gw)]) 
                           for p in players for gw in gw_list
                           ) - pulp.lpSum(transfer_cost * extra[gw] for gw in gw_list[1:]) 

        # 1) Initial squad: fix first GW composition if provided
        if initial_squad:
            missing = [p for p in initial_squad if p not in players]
            if missing:
                raise ValueError(f"Initial squad not in data: {missing}")
            gw0 = gw_list[0]
            for p in players:
                prob += x[(p, gw0)] == (1 if p in initial_squad else 0)

        # 2) Fixed players across the whole horizon (outside the per-GW loop)
        for p in fixed_players:
            for gw in gw_list:
                prob += x[(p, gw)] == 1
                if fixed_in_xi:
                    prob += y[(p, gw)] == 1

        # Constraints
        for gw in gw_list:
            # Squad size and budget per GW
            prob += pulp.lpSum(x[(p, gw)] for p in players) == n_players
            prob += pulp.lpSum(price.get(p, 0.0) * x[(p, gw)] for p in players) <= budget

            # Max 3 per team
            for t in set(team.values()):
                team_players = [p for p in players if team.get(p) == t]
                prob += pulp.lpSum(x[(p, gw)] for p in team_players) <= 3

            # Squad position requirements (fixed by rules)
            squad_min = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
            for position, count in squad_min.items():
                pos_players = [p for p in players if pos.get(p) == position]
                prob += pulp.lpSum(x[(p, gw)] for p in pos_players) == count

            # XI constraints per GW
            prob += pulp.lpSum(y[(p, gw)] for p in players) == 11
            for position, min_count in {"GK":1, "DEF":3, "MID":2, "FWD":1}.items():
                pos_players = [p for p in players if pos.get(p) == position]
                prob += pulp.lpSum(y[(p, gw)] for p in pos_players) >= min_count

            # Linking and captain per GW
            for p in players:
                prob += y[(p, gw)] <= x[(p, gw)]
                prob += c[(p, gw)] <= y[(p, gw)]
            prob += pulp.lpSum(c[(p, gw)] for p in players) == 1
        
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

            transfers_gw = pulp.lpSum(z_in[(p, gw)] for p in players)
            prob += extra[gw] >= transfers_gw - free_transfers_per_gw
            prob += extra[gw] >= 0

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        squad_ids_by_gw = {gw: [p for p in players if x[(p, gw)].value() == 1] for gw in gw_list}
        xi_ids = {gw: [p for p in players if y[(p, gw)].value() == 1] for gw in gw_list}
        captain_choices = {gw: next(p for p in players if c[(p, gw)].value() == 1) for gw in gw_list}

        # Simulate XI distribution
        xi_points = self.simulate_squad_horizon(xi_ids, captain_per_gw=captain_choices)

        transfers_by_gw = {}
        for i in range(1, len(gw_list)):
            gw = gw_list[i]
            transfers_by_gw[gw] = int(sum((z_in[(p, gw)].value() or 0) for p in players))

        return {
            "squad_ids": squad_ids_by_gw,
            "xi_ids": xi_ids,
            "captains": captain_choices,
            "mean": np.mean(xi_points),
            "std": np.std(xi_points),
            "quantiles": np.percentile(xi_points, [10, 50, 90])
        }

def pretty_print_result(result, players_df):
    """
    Print initial squad, then transfers and captain each week.
    """
    # Meta
    meta = (players_df[['Player_Name_fbref','position','team','value']]
            .drop_duplicates('Player_Name_fbref')
            .set_index('Player_Name_fbref'))
    
    # Build expected points lookup per (player, gw)
    epts = (players_df[['Player_Name_fbref','gameweek','expected_points_uncond']]
                .dropna(subset=['expected_points_uncond'])
                .assign(gameweek=lambda d: d['gameweek'].astype(int))
                .set_index(['Player_Name_fbref','gameweek'])['expected_points_uncond']
                .to_dict())

    mean = float(result['mean']); std = float(result['std'])
    q10, q50, q90 = [float(x) for x in result['quantiles']]
    print(f"Expected total points: {mean:.2f} ± {std:.2f}  (p10={q10:.2f}, median={q50:.2f}, p90={q90:.2f})\n")

    # Sorted GWs
    gw_sorted = sorted(result['squad_ids'].keys())
    if not gw_sorted:
        print("No gameweeks in result.")
        return

    # Initial squad
    gw0 = gw_sorted[0]
    initial = result['squad_ids'][gw0]
    print(f"Initial squad (GW {gw0}):")
    pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    def sort_key(name):
        pos = meta.loc[name]['position'] if name in meta.index else ''
        return (pos_order.get(pos, 9), name)

    for name in sorted(initial, key=sort_key):
        if name in meta.index:
            pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
            print(f"  - {name} [{pos}] ({team}) · £{price}")
        else:
            print(f"  - {name}")
    print("")

    # Weekly transfers and captain
    prev_set = set(initial)
    for gw in gw_sorted:
        curr = result['squad_ids'][gw]
        curr_set = set(curr)
        ins = sorted(curr_set - prev_set, key=sort_key)
        outs = sorted(prev_set - curr_set, key=sort_key)
        cap = result.get('captains', {}).get(gw, None)

        print(f"GW {gw}:")
        if ins:
            print("  Transfers in:")
            for name in ins:
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
                    print(f"    + {name} [{pos}] ({team}) · £{price}")
                else:
                    print(f"    + {name}")
        if outs:
            print("  Transfers out:")
            for name in outs:
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']; price = meta.loc[name, 'value']
                    print(f"    - {name} [{pos}] ({team}) · £{price}")
                else:
                    print(f"    - {name}")
        if not ins and not outs:
            print("  No transfers")

        print(f"  Captain: {cap}\n")

        xi = result.get('xi_ids', {}).get(gw, [])
        if xi:
            print("  Starting XI:")
            for name in sorted(xi, key=sort_key):
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']
                    print(f"    - {name} [{pos}] ({team})")
                else:
                    print(f"    - {name}")
        print("")

        bench = [name for name in curr if name not in set(xi)]
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
                if name in meta.index:
                    pos = meta.loc[name, 'position']; team = meta.loc[name, 'team']
                    print(f"    - {name} [{pos}] ({team}) · E[pts]={ep_str}")
                else:
                    print(f"    - {name} · E[pts]={ep_str}")
        print("")

        prev_set = curr_set


if __name__ == "__main__":
    ML_output = pd.read_csv("ML_model_predictions.csv")
    sim_engine = FPLSimulator(ML_output, n_sims=10000)
    fixed_players = ['Mohamed Salah']
    result = sim_engine.optimise_squad_with_transfers(gw_list=[4,5,6,7,8,9,10], budget=1000, n_players=15, fixed_players=[], fixed_in_xi=True)
    pretty_print_result(result, ML_output)








