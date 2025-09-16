import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup

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

def strip_fbref_team_url(html):
    soup = BeautifulSoup(html, 'html.parser')
    team_name = soup.a.text
    return team_name


class BaseMLModel:
    def __init__(self, df, rolling_features, future_features, pos, model_name, target='total_points'):
        self.df = df
        self.rolling_features = rolling_features
        self.future_features = future_features
        self.target = target
        self.model = None
        self.model_name = model_name
        self.pos = pos
        self.global_resid_std = None
        self.player_resid_std_map = {}
        self.beta_concentration = 8
        self.features = []
        for w in [3, 7]:
            self.features += [f'roll{w}_{feat}' for feat in self.rolling_features]
        self.features += self.future_features

    def build_features(self):
        df = self.df.copy()
        
        for feat in self.rolling_features:
            for w in [3, 7]:
                df[f'roll{w}_{feat}'] = (
                    df.groupby('FBRef_ID')[feat]
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
                )
        # Only shift features that are not known before the match
        # was_home, team_h_score, etc. are known pre-match, so use as-is


        df = df.dropna(subset=self.features)
        self.df = df

    def train_test_split_by_season(self, train_seasons, test_season):
        train = self.df[self.df['Season_fbref'].isin(train_seasons)]
        test = self.df[self.df['Season_fbref'] == test_season]
        train = train.dropna(subset=self.features)
        test = test.dropna(subset=self.features)
        if test.empty:
            print(train.tail())
            raise ValueError("Test set is empty.")
        return train, test

    def optimize_hyperparameters(self, train, model, test=None, n_trials=100):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if model == "xgboost":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBRegressor(**params, random_state=42)
                X_train, y_train = train[self.features], train[self.target]
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            
            with tqdm(total=n_trials, desc="Optimizing Hyperparameters") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)
                
                study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
            
            return study.best_params

        if self.model_name == 'neural network':
            def objective(trial):
                learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
                num_layers = trial.suggest_int('num_layers', 1, 5)
                units = trial.suggest_int('units', 32, 256, step=32)
                dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                epochs = trial.suggest_int('epochs', 10, 100)

                if test is not None:
                    X_train, y_train = train[self.features], train[self.target]
                    X_val, y_val = test[self.features], test[self.target]
                else:
                    X_train, X_val, y_train, y_val = train_test_split(train[self.features], train[self.target], test_size=0.2, random_state=42)

                model = Sequential()
                for _ in range(num_layers):
                    model.add(Dense(units, activation='relu'))
                    model.add(Dropout(dropout_rate))
                model.add(Dense(1))  # Output layer

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=0
                )

                val_loss = min(history.history['val_loss'])
                return val_loss
            
            study = optuna.create_study(direction='minimize')

            with tqdm(total=n_trials, desc="Optimizing Hyperparameters") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)

                study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
            return study.best_params

    def train(self, train, test=None, n_trials=100, best_params=None):
        X_train, y_train = train[self.features], train[self.target]
        if self.model_name == 'xgboost':
            if best_params is None:
                best_params = self.optimize_hyperparameters(train, model='xgboost', n_trials=n_trials)            
                print(f"Best parameters for {self.pos}: {best_params}")

            self.model = XGBRegressor(**best_params, random_state=42)
            self.model.fit(X_train, y_train)

        if self.model_name == 'neural network':
            if best_params is None:
                best_params = self.optimize_hyperparameters(train, model='neural network', n_trials=n_trials)
                print(f"Best parameters for {self.pos}: {best_params}")
            self.model = Sequential()
            for _ in range(best_params['num_layers']):
                self.model.add(Dense(best_params['units'], activation='relu'))
                self.model.add(Dropout(best_params['dropout_rate']))
            self.model.add(Dense(1))  # Output layer
            optimizer = Adam(learning_rate=best_params['learning_rate'])
            self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            self.model.fit(
                X_train_scaled, y_train,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                verbose=1
            )
        
        if self.model_name == "neural network":
            train_preds = self.model.predict(self.scaler.transform(X_train)).ravel()
        else:
            train_preds = self.model.predict(X_train)
        
        residuals = y_train.values - train_preds
        self.global_resid_std = float(np.std(residuals))

        resid_df = pd.DataFrame({
            'FBRef_ID': train['FBRef_ID'].values,
            'resid': residuals,
            'minutes_weightings': train.get('minutes_weightings', pd.Series([np.nan] * len(train)))
        })

        player_stats = resid_df.groupby('FBRef_ID').agg(
            resid_std=('resid', 'std'),
            n_obs=('resid', 'count'),
            avg_minutes_weighting=('minutes_weightings', 'mean')
        ).reset_index()

        player_stats['resid_std'] = player_stats['resid_std'].fillna(self.global_resid_std)
        player_stats.loc[player_stats['n_obs'] < 3, 'resid_std'] = self.global_resid_std

        self.player_resid_std_map = dict(zip(player_stats['FBRef_ID'], player_stats['resid_std']))
        self.player_residual_stats = player_stats

    def evaluate(self, test=None):
        if test is not None:
            X_test, y_test = test[self.features], test[self.target]
            if self.model_name == 'neural network':
                X_test = self.scaler.transform(X_test)
            preds = self.model.predict(X_test)
            if self.model_name == 'neural network':
                preds = preds.ravel()
            mae = mean_absolute_error(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
            # results = test[['FBRef_ID', 'Player_Name_fbref', 'Season_fbref', 'round_fbref', 'value']].copy()
            # results['actual'] = y_test
            # results['predicted'] = preds
            return mae, rmse
    
    def cross_validate(self, cv_folds=5):
        X, y = self.df[self.features], self.df[self.target]
        if self.model_name == 'xgboost':
            scores = cross_val_score(
                self.model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error'
            )
            print(f"Cross-validated MAE: {-scores.mean():.2f}")
            return -scores.mean(), scores.std()

        elif self.model_name == 'neural network':
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            maes = []
            rmses = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Train the neural network
                input_dim = len(self.features)
                nn_model = Sequential()
                nn_model.add(Dense(128, activation='relu', input_dim=input_dim))
                nn_model.add(Dropout(0.2))
                nn_model.add(Dense(64, activation='relu'))
                nn_model.add(Dropout(0.2))
                nn_model.add(Dense(1))
                nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

                nn_model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )

                # Evaluate on the validation set
                preds = nn_model.predict(X_val_scaled).ravel()
                maes.append(mean_absolute_error(y_val, preds))
                rmses.append(root_mean_squared_error(y_val, preds))

            print(f"Cross-validated MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
            print(f"Cross-validated RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
            return np.mean(maes), np.std(maes)
    
    def plot_feature_importances(self):
        if self.model_name == 'xgboost':
            # Ensure the model is trained
            if self.model is None:
                raise ValueError("Model is not trained. Train the model before plotting feature importances.")
            
            # Get feature importances
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importances for {self.pos}')
            plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
            plt.tight_layout()
            plt.show()

            return feature_importance_df
    
    def plot_residual_heteroskedasticity(self, log_scale=False):
        """
        Scatter plot residual std vs average minutes_weightings to inspect heteroskedasticity.
        Requires train() to have been called.
        """
        if not hasattr(self, 'player_residual_stats'):
            raise RuntimeError("Train the model first to compute residual statistics.")
        df_plot = self.player_residual_stats.copy()
        if df_plot['avg_minutes_weighting'].isna().all():
            print("minutes_weightings not available in training data.")
            return
        plt.figure(figsize=(7,5))
        plt.scatter(df_plot['avg_minutes_weighting'], df_plot['resid_std'], alpha=0.6)
        z = np.polyfit(df_plot['avg_minutes_weighting'].fillna(0), df_plot['resid_std'], 1)
        xp = np.linspace(0, df_plot['avg_minutes_weighting'].max(), 100)
        plt.plot(xp, z[0]*xp + z[1], color='red', linewidth=2, label='Linear trend')
        plt.xlabel('Avg minutes_weightings')
        plt.ylabel('Per-player residual std')
        if log_scale:
            plt.yscale('log')
        plt.title(f'Residual Std vs Minutes Weighting ({self.pos})\nGlobal std={self.global_resid_std:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict_next_gw(self, df, season, gw, plot=False):
        # Prepare prediction DataFrame for the specified season/gameweek
        pred_df = df[(df['Season_fbref'] == season) & (df['round_fbref'] == f'Matchweek {gw}')]
        pred_df = pred_df.dropna(subset=self.features)
        if pred_df.empty:
            print("No data available for prediction.")
            return None
        
        if self.model_name == 'neural network':
            X_pred = self.scaler.transform(pred_df[self.features])
        else:
            X_pred = pred_df[self.features]

        pred_df['predicted_points'] = self.model.predict(X_pred)
        sorted_points = pred_df[['Player_Name_fbref', 'predicted_points', 'value']].sort_values('predicted_points', ascending=False)
        print(sorted_points.head(10))

        if plot:
            top20 = sorted_points.head(40)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(top20['value'], top20['predicted_points'], c='blue')
            plt.xlabel('Player Value')
            plt.ylabel('Predicted Points')
            plt.title(f'Top 20 Predicted Points vs Value - {season} GW{gw}')
            x_min, x_max = top20['value'].min(), top20['value'].max()
            y_min, y_max = top20['predicted_points'].min(), top20['predicted_points'].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.tight_layout()

            # Add interactive hover tooltips
            cursor = mplcursors.cursor(scatter, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                name = top20.iloc[idx]['Player_Name_fbref']
                value = top20.iloc[idx]['value']
                points = top20.iloc[idx]['predicted_points']
                sel.annotation.set_text(f"{name}\nValue: {value}\nPredicted: {points:.2f}")

            plt.show()
        return sorted_points

    def predict_future_gw(self, fixtures_df, season, gw):
        self.df['gw_num'] = self.df['round_fbref'].str.extract(r'(\d+)').astype(float)
        mask = (
            (self.df['Season_fbref'] < season) |
            ((self.df['Season_fbref'] == season) & (self.df['gw_num'] < gw))
        )
        prev_stats = self.df[mask].copy()
        prev_stats = prev_stats.sort_values(['FBRef_ID', 'Season_fbref', 'gw_num'])

        last_stats = prev_stats.groupby('FBRef_ID').tail(1)
        rolling_feature_cols = [f'roll{w}_{feat}' for w in [3, 7] for feat in self.rolling_features]
        last_stats = last_stats[['team', 'Player_Name_fbref', 'Season_fbref', 'minutes_weightings'] + rolling_feature_cols].copy()
        last_stats = last_stats[last_stats['Season_fbref'] == season]

        pred_df = last_stats.merge(fixtures_df, on='team', how='left')
        pred_df['round_fbref'] = f'Matchweek {gw}'
        pred_df['was_home'] = pred_df['was_home'].astype(str).map({'True': 1, 'False': 0})

        pred_df['predicted_points'] = self.model.predict(pred_df[self.features])
        sorted_points = pred_df[['Player_Name_fbref', 'adjusted_predicted_points', 'predicted_points']].sort_values('adjusted_predicted_points', ascending=False)
        print(sorted_points.head(10))
    
    def predict_multiple_gws(self, season, gameweeks, top_n=30, start_prob_df=None):
        all_predictions = []

        if start_prob_df is None:
            start_prob_df = pd.read_csv('availability_dataset.csv')
        sp = start_prob_df.copy()
        sp['gw_num'] = sp['round_fbref'].str.extract(r'(\d+)').astype(float)
        sp = sp[sp['Season_fbref'] == season].copy()
        sp = (sp.sort_values(['FBRef_ID', 'gw_num']).drop_duplicates(subset=['FBRef_ID','gw_num'], keep='last'))
        for gw in gameweeks:
            fixtures_df = pd.read_csv(f'fixture_files/gameweek_{gw}.csv')
            self.df['gw_num'] = self.df['round_fbref'].str.extract(r'(\d+)').astype(float)
            mask = (
                (self.df['Season_fbref'] < season) |
                ((self.df['Season_fbref'] == season) & (self.df['gw_num'] < gw))
            )
            prev_stats = self.df[mask].copy()
            prev_stats = prev_stats.sort_values(['FBRef_ID', 'Season_fbref', 'gw_num'])

            last_stats = prev_stats.groupby('FBRef_ID').tail(1)
            rolling_feature_cols = [f'roll{w}_{feat}' for w in [3, 7] for feat in self.rolling_features]
            last_stats = last_stats[['team', 'FBRef_ID', 'Player_Name_fbref', 'Season_fbref', 'value', 'minutes_weightings'] + rolling_feature_cols].copy()
            last_stats = last_stats[last_stats['Season_fbref'] == season]

            last_stats['team'] = last_stats['team'].apply(strip_fbref_team_url)
            last_stats['team'] = last_stats['team'].replace(teams_dict)

            pred_df = last_stats.merge(fixtures_df, on='team', how='left')
            pred_df['was_home'] = pred_df['was_home'].astype(str).map({'True': 1, 'False': 0})
            X = pred_df[self.features]
            if self.model_name == 'neural network':
                X = self.scaler.transform(X)
            pred_df['predicted_points'] = self.model.predict(X)

            sp_prev = (sp[sp['gw_num'] < gw]
                       .sort_values(['FBRef_ID', 'gw_num'])
                       .groupby('FBRef_ID', as_index=False)
                       .tail(1)[['FBRef_ID','start_prob']])
            pred_df = pred_df.merge(sp_prev, on='FBRef_ID', how='left')
            dist_df = self.assign_distributions(pred_df, start_prob_df=None, hist_start_df=start_prob_df, tune_beta=True)
            dist_df['gameweek'] = gw
            all_predictions.append(dist_df[['Player_Name_fbref', 'team', 'opponent', 'FBRef_ID','gameweek',
                                    'expected_points_uncond', 'mean_points_cond', 'start_prob',
                                    'std_points_cond', 'resid_std_scaled', 'alpha_start', 'beta_start', 'value']]
            )
        
        combined = pd.concat(all_predictions)
        avg = (combined.groupby(['FBRef_ID','Player_Name_fbref'])['expected_points_uncond']
               .mean().reset_index()).rename(columns={'expected_points_uncond': 'avg_expected_points'})
        top_players = avg.sort_values('avg_expected_points', ascending=False).head(top_n)
        # Aggregate std columns by player (mean across gameweeks)
        std_agg = (combined.groupby(['FBRef_ID', 'Player_Name_fbref'])[['std_points_cond', 'resid_std_scaled']]
                .mean()
                .reset_index())

        # Merge aggregated stds with top_players
        merged = top_players.merge(std_agg, on=['FBRef_ID', 'Player_Name_fbref'], how='left')

        print(f"Top {top_n} players over gameweeks {gameweeks}:")
        print(merged[['Player_Name_fbref', 'avg_expected_points', 'resid_std_scaled']]) # 'std_points_cond' is the same when start_prob=1.0
        return top_players, combined
    
    def get_player_resid_std(self, fbref_id):
        return max(1e-3, self.player_resid_std_map.get(fbref_id, self.global_resid_std))

    def tune_beta_concentration_logloss(self, hist_df, prob_col='start_prob', actual_col='started_flag', candidates=(4,6,8,10,12,16)):
        """
        Tune the concentration parameter k for the Beta prior on start probability.

        Each player’s start probability is modeled as:
            p ~ Beta(1 + k*p, 1 + k*(1-p))
        where p is the predicted start probability from the model.

        The mean of this Beta is always p (independent of k), but the variance depends on k.
        Larger k → narrower distribution (more confidence), smaller k → wider distribution (more uncertainty).

        We select k by matching the Beta variance to the empirical noise in historical predictions:
        - Compute average Beta variance for each candidate k
        - Compute average squared error between predicted probabilities and actual starts
        - Pick k that minimizes the mismatch between these two variances
        """
        hist = hist_df[[prob_col, actual_col]].dropna()
        if hist.empty:
            # Default to 8 if no history is available
            self.beta_concentration = 8
            return 8
        p = hist[prob_col].clip(1e-6, 1-1e-6).values
        y = hist[actual_col].values
        best_k, best_score = None, float('inf')
        for k in candidates:
            # Beta parameters given mean p and concentration k
            alpha = 1 + p * k
            beta = 1 + (1 - p) * k
            # Theoretical variance of start probability under Beta(α, β)
            beta_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            # Empirical squared error between predicted probability and actual outcome
            se = (y - p)**2
            # Score = absolute mismatch between model variance and empirical variance
            mismatch = abs(beta_var.mean() - se.mean())
            if mismatch < best_score:
                best_score = mismatch
                best_k = k
        self.beta_concentration = best_k if best_k is not None else 8
        return self.beta_concentration

    def assign_distributions(self, pred_df, start_prob_df=None, hist_start_df=None,tune_beta=False):
        '''
        - Merges in the unconditional start probability (start_prob) from start_prob_df.
        - Uses the model's predicted points as the conditional mean (mean_points_cond), i.e., E[points | played].
        - Uses the per-player residual std as the conditional std (std_points_cond), i.e., std[points | played].
        - Computes the unconditional expected points: expected_points_uncond = start_prob * mean_points_cond.
        - Computes the unconditional standard deviation (resid_std_scaled) using the law of total variance:
                Var[Y] = p * σ_c^2 + p * (1-p) * μ_c^2,
            where p = start_prob, μ_c = mean_points_cond, σ_c = std_points_cond.
        - Optionally tunes the Beta concentration parameter k (if tune_beta=True and hist_start_df provided)
            to match the empirical variance of start events.
        - Computes Beta distribution parameters for start probability uncertainty:
                alpha_start = 1 + k * start_prob
                beta_start  = 1 + k * (1 - start_prob)
            and stores the value of k as beta_concentration_k.'''
        df = pred_df.copy()

        if start_prob_df is not None:
            sp = start_prob_df[['FBRef_ID', 'start_prob']].drop_duplicates()
            df = df.merge(sp, on='FBRef_ID', how='left', suffixes=('', '_sp'))
        df['start_prob'] = df['start_prob'].fillna(0.5).clip(0, 1)
        df['mean_points_cond'] = df['predicted_points']
        df['std_points_cond'] = [self.get_player_resid_std(fbref_id) for fbref_id in df['FBRef_ID']]
        df['std_points_cond'] = df['std_points_cond'].clip(1e-3, None)
        df['expected_points_uncond'] = df['mean_points_cond'] * df['start_prob']

        # Unconditional variance via mixture (law of total variance)
        # Var[Y] = p * σ_c^2 + p * (1-p) * μ_c^2
        p = df['start_prob'].values
        mu_c = df['mean_points_cond'].values
        sigma_c = df['std_points_cond'].values
        var_uncond = p * sigma_c**2 + p * (1 - p) * mu_c**2
        df['resid_std_scaled'] = np.sqrt(var_uncond).clip(1e-3, None)

        if tune_beta:
            availability_dataset = hist_start_df[['FBRef_ID', 'start_prob', 'started_flag']].dropna()
            self.tune_beta_concentration_logloss(hist_df=availability_dataset)
        k = getattr(self, 'beta_concentration', 8)
        df['alpha_start'] = 1 + k * df.get('start_prob', 0.5)
        df['beta_start'] = 1 + k * (1 - df.get('start_prob', 0.5))
        df['beta_concentration_k'] = k

        return df


class PlayerPipeline:
    def __init__(self, df, rolling_features, future_features, pos, model_name='xgboost'):
        self.df = df
        self.rolling_features = rolling_features
        self.future_features = future_features
        self.pos = pos
        self.model_name = model_name
        self.model = BaseMLModel(
            df=self.df,
            rolling_features=self.rolling_features,
            future_features=self.future_features,
            pos=self.pos,
            model_name=self.model_name
        )

    def build_features(self):
        self.model.build_features()

    def train_test_split(self, train_seasons, test_season):
        self.train, self.test = self.model.train_test_split_by_season(train_seasons, test_season)

    def train_model(self, n_trials=100, best_params=None):
        self.model.train(train=self.train, test=self.test, n_trials=n_trials, best_params=best_params)

    def evaluate(self, test=None, cv_folds=5):
        if test is not None:
            return self.model.evaluate(test=test)
        else:
            return self.model.cross_validate(cv_folds=cv_folds)

    def predict_next_gw(self, season, gw, plot=False):
        return self.model.predict_next_gw(self.model.df, season, gw, plot)

    def predict_future_gw(self, fixtures_df, season, gw):
        return self.model.predict_future_gw(fixtures_df, season, gw)

    def predict_multiple_gws(self, season, gameweeks, start_prob_df=None, top_n=10):
        return self.model.predict_multiple_gws(season, gameweeks=gameweeks, start_prob_df=start_prob_df, top_n=top_n)

class GoalkeeperPipeline(PlayerPipeline):
    def __init__(self, df, model_name='xgboost'):
        rolling_features = [
            'total_points', 'saves', 'bps', '60_played',
            'expected_goals_conceded', 'goals_conceded', 'clean_sheets', 'minutes_weightings'
        ]
        future_features = ['was_home', 'team_strength', 'opponent_strength']
        super().__init__(df, rolling_features, future_features, pos='GK', model_name=model_name)


class DefenderPipeline(PlayerPipeline):
    def __init__(self, df, model_name='xgboost'):
        rolling_features = [
            'total_points', 'bps',
            'goals_conceded', 'clean_sheets', '60_played',
            'assists_fpl', 'npxg', 'expected_goals_conceded', 'minutes_weightings'
        ]
        future_features = ['was_home', 'team_strength', 'opponent_strength']
        super().__init__(df, rolling_features, future_features, pos='DEF', model_name=model_name)


class MidfielderPipeline(PlayerPipeline):
    def __init__(self, df, model_name='xgboost'):
        rolling_features = [
            'total_points', 'bps', '60_played',
            'clean_sheets', 'assists_fpl', 'npxg', 'sca', 'gca',
            'goals', 'shots', 'expected_goals', 'expected_assists', 'minutes_weightings'
        ]
        future_features = ['was_home', 'team_strength', 'opponent_strength']
        super().__init__(df, rolling_features, future_features, pos='MID', model_name=model_name)


class ForwardPipeline(PlayerPipeline):
    def __init__(self, df, model_name='xgboost'):
        rolling_features = [
            'total_points', 'bps', '60_played',
            'assists_fpl', 'npxg', 'sca', 'gca',
            'goals', 'shots', 'expected_goals', 'expected_assists', 'minutes_weightings'
        ]
        future_features = ['was_home', 'team_strength', 'opponent_strength']
        super().__init__(df, rolling_features, future_features, pos='FWD', model_name=model_name)


def add_team_data(fixture_file, team_data_file):
    fixtures_df = pd.read_csv(fixture_file)
    team_df = pd.read_csv(team_data_file)

    for index, row in fixtures_df.iterrows():
        team = row['team']
        opponent = row['opponent']
        was_home_raw = row['was_home']
        if isinstance(was_home_raw, str):
            was_home = was_home_raw.strip().lower() in ('true', '1', 'yes')
        else:
            was_home = bool(was_home_raw)
        team_strength_col = 'strength_overall_home' if was_home else 'strength_overall_away'
        opp_strength_col = 'strength_overall_away' if was_home else 'strength_overall_home'

        team_strength = team_df.loc[team_df['name'] == team, team_strength_col].values[0]
        opp_strength = team_df.loc[team_df['name'] == opponent, opp_strength_col].values[0]

        fixtures_df.at[index, 'team_strength'] = team_strength
        fixtures_df.at[index, 'opponent_strength'] = opp_strength

    return fixtures_df

def process_all_fixtures(fixtures_folder, team_data_file):
    for fixture_file in os.listdir(fixtures_folder):
        if fixture_file.endswith('.csv'):
            fixture_path = os.path.join(fixtures_folder, fixture_file)
            updated_fixtures = add_team_data(fixture_path, team_data_file)
            updated_fixtures.to_csv(fixture_path, index=False)


def ML_predictions(gameweeks=None):
    # fixtures_folder = "fixture_files"
    # team_data_file = "Fantasy-Premier-League/data/2025-26/teams.csv"
    # process_all_fixtures(fixtures_folder, team_data_file)
    
    # Goalkeeper pipeline
    gk_df = pd.read_csv("merged_gk_stats.csv")
    gk_pipeline = GoalkeeperPipeline(gk_df, model_name='xgboost')
    gk_pipeline.build_features()
    gk_pipeline.train_test_split(train_seasons=['2023-2024'], test_season='2024-2025')
    gk_pipeline.train_model(best_params={'n_estimators': 60, 'max_depth': 6, 'learning_rate': 0.010643654007341458, 'subsample': 0.707939659787393, 'colsample_bytree': 0.659667184367691})
    
    # Defender pipeline
    def_df = pd.read_csv("merged_def_stats.csv")
    def_pipeline = DefenderPipeline(def_df, model_name='xgboost')
    def_pipeline.build_features()
    def_pipeline.train_test_split(train_seasons=['2023-2024'], test_season='2024-2025') 
    def_pipeline.train_model(best_params={'n_estimators': 123, 'max_depth': 7, 'learning_rate': 0.03284182903635011, 'subsample': 0.9842574373873748, 'colsample_bytree': 0.6021656349105039})

    # Midfielder pipeline
    mid_df = pd.read_csv("merged_mid_stats.csv")
    mid_pipeline = MidfielderPipeline(mid_df, model_name='xgboost')
    mid_pipeline.build_features()
    mid_pipeline.train_test_split(train_seasons=['2023-2024'], test_season='2024-2025')
    mid_pipeline.train_model(best_params={'n_estimators': 123, 'max_depth': 7, 'learning_rate': 0.03284182903635011, 'subsample': 0.9842574373873748, 'colsample_bytree': 0.6021656349105039})

    # Forward pipeline
    fwd_df = pd.read_csv("merged_fwd_stats.csv")
    fwd_pipeline = ForwardPipeline(fwd_df, model_name='xgboost')
    fwd_pipeline.build_features()
    fwd_pipeline.train_test_split(train_seasons=['2023-2024'], test_season='2024-2025')
    fwd_pipeline.train_model(best_params={'n_estimators': 60, 'max_depth': 3, 'learning_rate': 0.023469835157219225, 'subsample': 0.7081955706547465, 'colsample_bytree': 0.8680614630492703})

    output = []
    for pos, combined in [('FWD', fwd_pipeline), ('GK', gk_pipeline), ('DEF', def_pipeline), ('MID', mid_pipeline)]:
        top_players, all_preds = combined.predict_multiple_gws(season='2025-2026', gameweeks=gameweeks, start_prob_df=None, top_n=30)
        all_preds['position'] = pos
        output.append(all_preds)

    output_df = pd.concat(output)
    output_df.to_csv("ML_model_predictions.csv", index=False)