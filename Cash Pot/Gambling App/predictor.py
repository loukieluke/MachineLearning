import pandas as pd
import random
import numpy as np
from collections import Counter, defaultdict
from ml_trainer import LotteryMLTrainer, RandomForestPredictor

class LotteryPredictor:
    def __init__(self, db_manager, auto_train_startup=True):
        self.db = db_manager
        self.ml_trainer = LotteryMLTrainer()
        self.rf_predictor = RandomForestPredictor()
        self.last_prediction_method = None
        self.last_prediction_confidence = None
        self.update_data(train_models=auto_train_startup)
        # Try to load pre-trained models
        self.ml_loaded = self.ml_trainer.load_models()

    def update_data(self, train_models=True):
        """Update internal data from database and optionally retrain models."""
        self.df = self.db.get_all_draws()

        if not self.df.empty:
            self.number_frequency = Counter(self.df['numbers'])
            self._analyze_patterns()

            # Train ML models if we have enough data (chronological order)
            numbers_list = self._numbers_chronological()
            if train_models and len(numbers_list) >= 20:
                # Train Random Forest
                rf_trained = self.rf_predictor.train(numbers_list) # Store the result

                self.rf_trained = rf_trained # Track if RF is trained

                # Train LSTM if we have significant new data
                if len(numbers_list) >= 50 and len(numbers_list) % 10 == 0: # Retrain periodically
                    print("Retraining LSTM model with new data...")
                    self.ml_trainer.train_models(self.df)
                    self.ml_loaded = True
            elif train_models:
                self.rf_trained = False # Not enough data for RF training
            else:
                self.rf_trained = hasattr(self.rf_predictor, 'models')
        else:
            self.number_frequency = Counter()
            self.patterns = defaultdict(Counter)
            self.last_draws = []
            self.rf_trained = False # No data, so RF not trained

    def _numbers_chronological(self):
        """
        Draws from the DB are newest-first (ORDER BY date DESC).
        All sequence models and transition counts need oldest → newest.
        """
        if self.df is None or self.df.empty:
            return []
        return [int(x) for x in self.df['numbers'].iloc[::-1].tolist()]

    def _analyze_patterns(self):
        """Analyze patterns in the historical data"""
        self.patterns = defaultdict(Counter)
        numbers_chron = self._numbers_chronological()
        self.last_draws = numbers_chron[-20:] if numbers_chron else []

        if len(numbers_chron) < 2:
            return

        for i in range(1, len(numbers_chron)):
            prev_num = numbers_chron[i - 1]
            current_num = numbers_chron[i]
            self.patterns[prev_num][current_num] += 1

    def predict(self, method='ml_lstm', count=5, target_draw_at=None):
        """Generate predictions based on selected method. target_draw_at = datetime string for which draw this is for."""
        if self.df.empty:
            return self._random_prediction(count)

        method_to_use = self._resolve_method(method)
        self.last_prediction_method = method_to_use
        self.last_prediction_confidence = None

        numbers = []

        if method_to_use == 'ml_lstm' and self.ml_loaded:
            lstm_probs = self.ml_trainer.predict_next_probabilities(self.last_draws)
            if lstm_probs is not None:
                numbers = self._topk_from_probability_vector(lstm_probs, count)
                self.last_prediction_confidence = self._confidence_from_probability_vector(lstm_probs, count)
            else:
                numbers = None
        elif method_to_use == 'ml_rf' and hasattr(self, 'rf_trained') and self.rf_trained:
            numbers = self.rf_predictor.predict(self.last_draws)
        elif method_to_use == 'weighted':
            numbers = self._weighted_prediction(count)
        elif method_to_use == 'pattern':
            numbers = self._pattern_based_prediction(count)
        elif method_to_use == 'hot':
            numbers = self._hot_numbers_prediction(count)
        elif method_to_use == 'ensemble':
            numbers, ensemble_probs = self._ensemble_prediction(count, return_probs=True)
            self.last_prediction_confidence = self._confidence_from_probability_vector(ensemble_probs, count)
        else:
            numbers = self._random_prediction(count)

        if numbers is None or len(numbers) == 0:
            numbers = self._random_prediction(count)

        # Ensure we have exactly 'count' numbers
        if len(numbers) > count:
            numbers = numbers[:count]
        elif len(numbers) < count:
            numbers.extend(self._random_prediction(count - len(numbers)))    

        # Save prediction to database with target draw datetime for linking when actual is added
        self.db.save_prediction(
            numbers,
            method_to_use,
            target_draw_at=target_draw_at,
            confidence=self.last_prediction_confidence,
            requested_method=method,
        )
        return sorted([int(num) for num in numbers])

    def _resolve_method(self, requested_method):
        """Resolve requested method. 'auto' selects strongest available method from recent scores."""
        if requested_method != 'auto':
            return requested_method

        # Governance-aware preference: if latest promoted benchmark says ensemble beats LSTM,
        # bias auto selection toward ensemble.
        promoted = self.db.get_latest_promoted_benchmark()
        if promoted and isinstance(promoted.get('summary_json'), dict):
            summary = promoted.get('summary_json', {})
            ens = summary.get('ensemble', {})
            lstm = summary.get('ml_lstm', {})
            ens_mean = ens.get('mean_hit_rate')
            lstm_mean = lstm.get('mean_hit_rate')
            if (
                ens_mean is not None
                and (lstm_mean is None or float(ens_mean) >= float(lstm_mean))
                and self.ml_loaded
            ):
                return 'ensemble'

        recent_scores = self._get_recent_method_scores(window=300, min_samples=5)
        priors = {
            'ensemble': 0.20,
            'ml_lstm': 0.19,
            'ml_rf': 0.15,
            'weighted': 0.16,
            'pattern': 0.14,
            'hot': 0.14,
            'random': 0.10
        }

        candidates = []
        if self.ml_loaded:
            candidates.append('ml_lstm')
        if hasattr(self, 'rf_trained') and self.rf_trained:
            candidates.append('ml_rf')
        candidates.extend(['ensemble', 'weighted', 'pattern', 'hot', 'random'])

        best_method = 'weighted'
        best_score = -1.0
        for m in candidates:
            score = recent_scores.get(m, priors.get(m, 0.1))
            if score > best_score:
                best_score = score
                best_method = m
        return best_method

    def get_preferred_auto_method(self):
        """Public helper for UI/status APIs."""
        return self._resolve_method('auto')
    
    def _ensemble_prediction(self, count, return_probs=False):
        """Combine methods by blending per-number probabilities with LSTM floor weighting."""
        method_scores = self._get_recent_method_scores(window=200, min_samples=3)
        method_probs = {}

        if self.ml_loaded:
            lstm_probs = self.ml_trainer.predict_next_probabilities(self.last_draws)
            if lstm_probs is not None and len(lstm_probs) == 36:
                method_probs['ml_lstm'] = np.array(lstm_probs, dtype=float)

        if hasattr(self, 'rf_trained') and self.rf_trained:
            rf_preds = self.rf_predictor.predict(self.last_draws)
            if rf_preds:
                method_probs['ml_rf'] = self._prediction_list_to_prob_vector(rf_preds)

        weighted_preds = self._weighted_prediction(count)
        hot_preds = self._hot_numbers_prediction(count)
        pattern_preds = self._pattern_based_prediction(count)
        method_probs['weighted'] = self._prediction_list_to_prob_vector(weighted_preds)
        method_probs['hot'] = self._prediction_list_to_prob_vector(hot_preds)
        method_probs['pattern'] = self._prediction_list_to_prob_vector(pattern_preds)

        combined = self._blend_method_probabilities(method_probs, method_scores)
        if combined is None:
            result = self._random_prediction(count)
            return (result, None) if return_probs else result
        result = self._topk_from_probability_vector(combined, count)
        return (result, combined) if return_probs else result

    def _prediction_list_to_prob_vector(self, preds):
        """Convert ranked prediction list into a 36-length pseudo-probability vector."""
        vec = np.zeros(36, dtype=float)
        if not preds:
            return vec
        clean = [int(n) for n in preds if 1 <= int(n) <= 36]
        if not clean:
            return vec
        # Higher rank gets more mass.
        total_rank_weight = sum(range(1, len(clean) + 1))
        for i, n in enumerate(clean):
            rank_weight = (len(clean) - i) / total_rank_weight
            vec[n - 1] += rank_weight
        s = vec.sum()
        return (vec / s) if s > 0 else vec

    def _blend_method_probabilities(self, method_probs, method_scores):
        """Blend method probability vectors with score-based weights and LSTM floor."""
        if not method_probs:
            return None

        has_lstm = 'ml_lstm' in method_probs
        min_non_lstm_score = 0.135
        weights = {}

        for method_name in method_probs:
            score = method_scores.get(method_name, 0.25)
            if has_lstm and method_name != 'ml_lstm' and score < min_non_lstm_score:
                continue

            if method_name == 'ml_lstm':
                mult = 1.8
            elif method_name == 'ml_rf':
                mult = 0.9
            else:
                mult = 0.75
            weights[method_name] = max(0.01, score * mult)

        if not weights:
            return None

        if has_lstm and 'ml_lstm' in weights:
            non_lstm_total = sum(w for m, w in weights.items() if m != 'ml_lstm')
            # Keep LSTM dominant if available.
            weights['ml_lstm'] = max(weights['ml_lstm'], 1.2 * non_lstm_total, 0.8)

        total_w = sum(weights.values())
        if total_w <= 0:
            return None

        combined = np.zeros(36, dtype=float)
        for method_name, w in weights.items():
            vec = np.array(method_probs[method_name], dtype=float)
            if vec.shape[0] != 36:
                continue
            s = vec.sum()
            if s > 0:
                combined += (w / total_w) * (vec / s)
        s = combined.sum()
        return (combined / s) if s > 0 else None

    def _topk_from_probability_vector(self, probs, count):
        """Return top-k unique numbers from a 36-length probability vector."""
        if probs is None or len(probs) != 36:
            return self._random_prediction(count)
        ranked_idx = np.argsort(np.array(probs, dtype=float))[::-1]
        out = []
        for idx in ranked_idx:
            num = int(idx) + 1
            if 1 <= num <= 36 and num not in out:
                out.append(num)
            if len(out) >= count:
                break
        if len(out) < count:
            # deterministic fill from lowest not yet used numbers
            for num in range(1, 37):
                if num not in out:
                    out.append(num)
                if len(out) >= count:
                    break
        return sorted(out[:count])

    def _confidence_from_probability_vector(self, probs, count):
        """
        Confidence score in [0, 1] based on probability concentration on selected top-k numbers.
        Higher means model is more certain.
        """
        if probs is None or len(probs) != 36:
            return None
        p = np.array(probs, dtype=float)
        s = p.sum()
        if s <= 0:
            return None
        p = p / s
        topk = np.sort(p)[-max(1, count):]
        mass = float(topk.sum())
        # normalize entropy of top-k distribution
        q = topk / np.clip(topk.sum(), 1e-8, None)
        entropy = float(-np.sum(q * np.log(np.clip(q, 1e-8, 1.0))))
        max_entropy = np.log(len(topk))
        sharpness = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
        conf = 0.7 * mass + 0.3 * sharpness
        return max(0.0, min(1.0, conf))

    def _get_recent_method_scores(self, window=200, min_samples=3, alpha=1.0, beta=3.0):
        """
        Return smoothed recent hit-rate scores for each method.
        Uses linked prediction history and a Beta prior to avoid over-weighting small samples.
        """
        history = self.db.get_prediction_history(limit=window)
        if history is None or history.empty:
            return {}

        stats = defaultdict(lambda: {'hits': 0, 'total': 0})
        for _, row in history.iterrows():
            actual = row.get('actual_number')
            method = row.get('method')
            pred_str = row.get('predicted_numbers')
            if pd.isna(actual) or not method or not pred_str:
                continue
            try:
                preds = {int(x.strip()) for x in str(pred_str).split(',') if x.strip()}
                stats[method]['total'] += 1
                if int(actual) in preds:
                    stats[method]['hits'] += 1
            except (TypeError, ValueError):
                continue

        scores = {}
        for method, s in stats.items():
            total = s['total']
            if total < min_samples:
                continue
            # Smoothed posterior mean: (hits + alpha) / (total + alpha + beta)
            scores[method] = (s['hits'] + alpha) / (total + alpha + beta)
        return scores

    def _random_prediction(self, count):
        """Generate completely random numbers"""
        return sorted(random.sample(range(1, 37), count))
    
    def _weighted_prediction(self, count):
        """Generate numbers weighted by frequency"""
        total = sum(self.number_frequency.values())
        if total == 0:
            return self._random_prediction(count)
        
        weights = [self.number_frequency.get(i, 0.1) / total for i in range(1, 37)]
        numbers = random.choices(range(1, 37), weights=weights, k=count*2)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for num in numbers:
            if num not in seen and len(result) < count:
                seen.add(num)
                result.append(num)

        # if we don't have enough unique numbers, fill with random
        while len(result) < count:
            num = random.randint(1, 36)
            if num not in seen:
                seen.add(num)
                result.append(num)        

        return sorted(result)

    def _pattern_based_prediction(self, count):
        """Generate numbers based on pattern analysis"""
        if not self.last_draws:
            return self._random_prediction(count)

        last_number = self.last_draws[-1]
        candidates = []

        if last_number in self.patterns and self.patterns[last_number]:
            # Get numbers that frequently follow the last drawn number
            for num, freq in self.patterns[last_number].most_common(5):
                candidates.append(num)

        # Add some hot numbers
        hot_numbers = [num for num, freq in self.number_frequency.most_common(5)]
        candidates.extend(hot_numbers)

        # Remove duplicates and ensure we have enough numbers
        candidates = list(set(candidates))
        while len(candidates) < count:
            num = random.randint(1, 36)
            if num not in candidates:
                candidates.append(num)

        return sorted(candidates[:count])
    
    def _hot_numbers_prediction(self, count):
        """Generate prediction based on most frequently drawn numbers"""

        if not self.number_frequency:
            return self._random_prediction(count)
        
        hot_numbers = [num for num, freq in self.number_frequency.most_common(10)]
        result = hot_numbers[:count]

        while len(result) < count:
            num = random.randint(1, 36)
            if num not in result:
                result.append(num)

        return sorted(result)

    def get_stats(self):
        """Get statistics about the data and models"""
        if self.df.empty:
            return {
                'total_draws': 0,
                'date_range': 'No data',
                'most_frequent': [],
                'recent_draws': [],
                'ml_model_loaded': False,
                'rf_model_trained': False
            }
        
        return {
                'total_draws': len(self.df),
                'date_range': f"{self.df['date'].min()} to {self.df['date'].max()}",
                'most_frequent': self.number_frequency.most_common(5),
                'recent_draws': self.last_draws[-5:] if self.last_draws else [],
                'ml_model_loaded': self.ml_loaded,
                'rf_model_trained': hasattr(self.rf_predictor, 'models')
        }

    def backtest_walk_forward(
        self,
        count=5,
        start_size=60,
        methods=None,
        mode='quick',
        rf_retrain_every=25,
        seed=None,
        max_evals=None
    ):
        """
        Walk-forward backtest for lightweight methods.
        Predict draw t using history [0..t-1], then score against draw t.
        """
        if methods is None:
            if mode == 'full':
                methods = ['weighted', 'pattern', 'hot', 'random', 'ensemble', 'ml_lstm', 'ml_rf']
            else:
                methods = ['weighted', 'pattern', 'hot', 'random', 'ensemble']

        draws = self.db.get_all_draws()
        if draws is None or draws.empty or len(draws) <= start_size:
            return {
                'evaluated_draws': 0,
                'start_size': start_size,
                'count': count,
                'methods': {}
            }

        # get_all_draws returns DESC by date; reverse for chronological walk-forward
        numbers = [int(x) for x in draws['numbers'].tolist()][::-1]
        total_possible = max(0, len(numbers) - start_size)
        if max_evals is not None:
            try:
                max_evals = int(max_evals)
            except (TypeError, ValueError):
                max_evals = None
        if max_evals is not None and max_evals > 0:
            eval_count = min(total_possible, max_evals)
        else:
            eval_count = total_possible
        start_t = len(numbers) - eval_count
        if start_t < start_size:
            start_t = start_size

        rng = random.Random(seed) if seed is not None else random

        method_hits = defaultdict(int)
        method_total = defaultdict(int)
        recent_outcomes = defaultdict(list)  # rolling hits used by ensemble in backtest

        def random_pick(k):
            return sorted(rng.sample(range(1, 37), k))

        def weighted_pick(history, k):
            freq = Counter(history)
            total = sum(freq.values())
            if total == 0:
                return random_pick(k)
            weights = [freq.get(i, 0.1) / total for i in range(1, 37)]
            picks = rng.choices(range(1, 37), weights=weights, k=k * 3)
            result = []
            seen = set()
            for n in picks:
                if n not in seen:
                    seen.add(n)
                    result.append(n)
                if len(result) >= k:
                    break
            while len(result) < k:
                n = rng.randint(1, 36)
                if n not in seen:
                    seen.add(n)
                    result.append(n)
            return sorted(result)

        def hot_pick(history, k):
            freq = Counter(history)
            hot = [n for n, _ in freq.most_common(10)]
            result = hot[:k]
            while len(result) < k:
                n = rng.randint(1, 36)
                if n not in result:
                    result.append(n)
            return sorted(result)

        def pattern_pick(history, k):
            if not history:
                return random_pick(k)
            pat = defaultdict(Counter)
            for i in range(1, len(history)):
                pat[history[i - 1]][history[i]] += 1
            candidates = []
            last_num = history[-1]
            for n, _ in pat[last_num].most_common(6):
                candidates.append(n)
            for n, _ in Counter(history).most_common(6):
                candidates.append(n)
            dedup = []
            seen = set()
            for n in candidates:
                if n not in seen:
                    seen.add(n)
                    dedup.append(n)
                if len(dedup) >= k:
                    break
            while len(dedup) < k:
                n = rng.randint(1, 36)
                if n not in seen:
                    seen.add(n)
                    dedup.append(n)
            return sorted(dedup[:k])

        # Optional backtest RF model (trained periodically on historical window only).
        rf_bt = RandomForestPredictor() if 'ml_rf' in methods else None
        rf_bt_ready = False

        for t in range(start_t, len(numbers)):
            history = numbers[:t]
            actual = numbers[t]

            candidates = {}
            if 'weighted' in methods:
                candidates['weighted'] = weighted_pick(history, count)
            if 'hot' in methods:
                candidates['hot'] = hot_pick(history, count)
            if 'pattern' in methods:
                candidates['pattern'] = pattern_pick(history, count)
            if 'random' in methods:
                candidates['random'] = random_pick(count)
            if 'ml_lstm' in methods and self.ml_loaded:
                lstm_preds = self.ml_trainer.predict_next_numbers(history, count=count)
                if lstm_preds:
                    candidates['ml_lstm'] = lstm_preds
            if 'ml_rf' in methods and rf_bt is not None:
                # Re-fit periodically so each prediction uses only past data.
                if (not rf_bt_ready) or ((t - start_size) % max(1, int(rf_retrain_every)) == 0):
                    rf_bt_ready = rf_bt.train(history)
                if rf_bt_ready:
                    rf_preds = rf_bt.predict(history)
                    if rf_preds:
                        candidates['ml_rf'] = rf_preds
            if 'ensemble' in methods:
                bt_method_probs = {}
                bt_scores = {}
                for m in ('weighted', 'hot', 'pattern', 'ml_lstm', 'ml_rf'):
                    if m not in candidates:
                        continue
                    bt_method_probs[m] = self._prediction_list_to_prob_vector(candidates[m])
                    outcomes = recent_outcomes[m][-50:]
                    hits = sum(outcomes)
                    total = len(outcomes)
                    bt_scores[m] = (hits + 1.0) / (total + 4.0) if total > 0 else 0.25

                combined = self._blend_method_probabilities(bt_method_probs, bt_scores)
                if combined is not None:
                    candidates['ensemble'] = self._topk_from_probability_vector(combined, count)
                else:
                    candidates['ensemble'] = random_pick(count)

            for method, preds in candidates.items():
                method_total[method] += 1
                hit = int(actual in set(preds))
                method_hits[method] += hit
                recent_outcomes[method].append(hit)

        out = {}
        for m in methods:
            total = method_total.get(m, 0)
            hits = method_hits.get(m, 0)
            out[m] = {
                'hits': hits,
                'total': total,
                'hit_rate': (hits / total) if total else None
            }

        return {
            'evaluated_draws': max(0, len(numbers) - start_t),
            'start_size': start_size,
            'count': count,
            'mode': mode,
            'seed': seed,
            'max_evals': max_evals,
            'rf_retrain_every': rf_retrain_every if 'ml_rf' in methods else None,
            'notes': {
                'ml_lstm_requires_loaded_model': bool(self.ml_loaded),
                'ml_rf_backtest_retrained_periodically': 'ml_rf' in methods
            },
            'methods': out
        }

    def backtest_multi_seed(
        self,
        seeds,
        count=5,
        start_size=60,
        mode='quick',
        rf_retrain_every=25,
        max_evals=None
    ):
        """
        Run multiple seeded backtests and aggregate hit-rate statistics.
        Returns per-seed runs + mean/std/min/max hit rate per method.
        """
        clean_seeds = []
        for s in seeds or []:
            try:
                clean_seeds.append(int(s))
            except (TypeError, ValueError):
                continue
        if not clean_seeds:
            clean_seeds = [42, 43, 44]

        runs = []
        for s in clean_seeds:
            runs.append(
                self.backtest_walk_forward(
                    count=count,
                    start_size=start_size,
                    mode=mode,
                    rf_retrain_every=rf_retrain_every,
                    seed=s,
                    max_evals=max_evals,
                )
            )

        method_rates = defaultdict(list)
        method_hits = defaultdict(int)
        method_total = defaultdict(int)
        for run in runs:
            for method, stats in run.get('methods', {}).items():
                hr = stats.get('hit_rate')
                if hr is not None:
                    method_rates[method].append(float(hr))
                method_hits[method] += int(stats.get('hits', 0))
                method_total[method] += int(stats.get('total', 0))

        summary = {}
        for method, rates in method_rates.items():
            arr = np.array(rates, dtype=float)
            summary[method] = {
                'mean_hit_rate': float(arr.mean()) if arr.size else None,
                'std_hit_rate': float(arr.std()) if arr.size else None,
                'min_hit_rate': float(arr.min()) if arr.size else None,
                'max_hit_rate': float(arr.max()) if arr.size else None,
                'runs': int(arr.size),
                'combined_hits': method_hits.get(method, 0),
                'combined_total': method_total.get(method, 0),
            }

        return {
            'seeds': clean_seeds,
            'count': count,
            'start_size': start_size,
            'mode': mode,
            'rf_retrain_every': rf_retrain_every if mode == 'full' else None,
            'max_evals': max_evals,
            'evaluated_draws_per_run': runs[0].get('evaluated_draws', 0) if runs else 0,
            'summary': summary,
            'runs': runs,
        }
    
    def force_retrain(self):
        """Force retraining of ML models"""
        if not self.df.empty:
            success = self.ml_trainer.train_models(self.df, force_retrain=True)
            if success:
                self.ml_loaded = True

                # Also retrain Random Forest
                numbers_list = self._numbers_chronological()
                if len(numbers_list) >= 20:
                    rf_success = self.rf_predictor.train(numbers_list)
                    self.rf_trained = rf_success
                    return success or rf_success
        return False



            
                
