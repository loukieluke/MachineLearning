import sqlite3
import pandas as pd
from datetime import datetime
import os
import json

class DatabaseManager:
    def __init__(self, db_path='lottery_data.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Main draws table
        c.execute('''CREATE TABLE IF NOT EXISTS draws
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    game TEXT NOT NULL,
                    name TEXT NOT NULL,
                    numbers INTEGER NOT NULL,
                    event TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Predictions history table
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_numbers TEXT NOT NULL,
                    method TEXT NOT NULL,
                    target_draw_at TEXT,
                    confidence REAL,
                    actual_draw_id INTEGER REFERENCES draws(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Benchmark snapshots for model governance (promotion gate)
        c.execute('''CREATE TABLE IF NOT EXISTS benchmarks
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mode TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    start_size INTEGER NOT NULL,
                    max_evals INTEGER,
                    rf_retrain_every INTEGER,
                    seeds_json TEXT,
                    summary_json TEXT NOT NULL,
                    decision TEXT,
                    reason TEXT,
                    ensemble_mean REAL,
                    ensemble_std REAL,
                    lstm_mean REAL)''')

        # Add new columns if they don't exist (for existing databases)
        for col, spec in [
            ('target_draw_at', 'TEXT'),
            ('confidence', 'REAL'),
            ('actual_draw_id', 'INTEGER REFERENCES draws(id)'),
            ('requested_method', 'TEXT'),
        ]:
            try:
                c.execute(f'ALTER TABLE predictions ADD COLUMN {col} {spec}')
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.commit()
        conn.close()

    def add_draw(self, draw_data):
        """Add a single draw to the database. Returns new draw id if added, None if duplicate."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Clean the event ID - remove '#' if present
        event_id = draw_data['event'].strip()
        if event_id.startswith('#'):
            event_id = event_id[1:]  # Remove the '#' character

        # Check if draw already exists (based on date, game, and event)
        c.execute('''SELECT id FROM draws WHERE date = ? AND game = ? AND event = ?''',
                  (draw_data['date'], draw_data['game'], event_id))

        if c.fetchone() is None:
            c.execute('''INSERT INTO draws (date, game, name, numbers, event)
                      VALUES (?, ?, ?, ?, ?)''',
                      (draw_data['date'], draw_data['game'], draw_data['name'],
                       draw_data['numbers'], event_id))
            draw_id = c.lastrowid
            conn.commit()
            # Link any predictions that were for this draw
            from schedule import target_draw_at_from_date_and_game
            target_at = target_draw_at_from_date_and_game(draw_data['date'], draw_data['game'])
            if target_at:
                self._link_predictions_to_draw(c, draw_id, target_at)
                conn.commit()
            conn.close()
            return draw_id
        conn.close()
        return None

    def add_draws_from_csv(self, csv_data):
        """Add multiple draws from CSV data. Uses 'game' column for linking predictions if present."""
        added_count = 0
        for draw_data in csv_data:
            # Clean event ID before adding
            if 'event' in draw_data and draw_data['event']:
                event_id = draw_data['event'].strip()
                if event_id.startswith('#'):
                    event_id = event_id[1:]
                draw_data['event'] = event_id

            if self.add_draw(draw_data) is not None:
                added_count += 1
        return added_count

    def get_all_draws(self):
        """Get all draws ordered by date"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('SELECT * FROM draws ORDER BY date DESC', conn)
        conn.close()
        return df

    def get_draws_page(self, page=1, per_page=50):
        """Get paginated draws and total count."""
        try:
            page = max(1, int(page))
            per_page = max(1, int(per_page))
        except (TypeError, ValueError):
            page, per_page = 1, 50

        offset = (page - 1) * per_page
        conn = sqlite3.connect(self.db_path)
        count_row = pd.read_sql('SELECT COUNT(*) AS total FROM draws', conn)
        total = int(count_row.iloc[0]['total']) if not count_row.empty else 0
        df = pd.read_sql(
            'SELECT * FROM draws ORDER BY date DESC LIMIT ? OFFSET ?',
            conn,
            params=(per_page, offset),
        )
        conn.close()
        return df, total

    def get_draws_count(self):
        """Get total number of draws"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM draws')
        count = c.fetchone()[0]
        conn.close()
        return count

    def update_draw(self, draw_id, numbers):
        """Update the drawn number for an existing draw. Returns True if a row was updated."""
        try:
            draw_id = int(draw_id)
            numbers = int(numbers)
            if not (1 <= numbers <= 36):
                return False
        except (TypeError, ValueError):
            return False
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE draws SET numbers = ? WHERE id = ?', (numbers, draw_id))
        updated = c.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def save_prediction(self, numbers, method, target_draw_at=None, confidence=None, requested_method=None):
        """Save a prediction to the database with optional target draw datetime.
        method = resolved engine; requested_method = dropdown choice (e.g. 'auto' vs resolved 'ensemble').
        created_at is stored in Jamaica time (America/Jamaica)."""
        from schedule import JAMAICA
        created_at = datetime.now(JAMAICA).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        numbers_str = ','.join(map(str, numbers))
        c.execute(
            '''INSERT INTO predictions (predicted_numbers, method, target_draw_at, confidence, created_at, requested_method)
                      VALUES (?, ?, ?, ?, ?, ?)''',
            (numbers_str, method, target_draw_at, confidence, created_at, requested_method),
        )
        conn.commit()
        conn.close()

    def get_recent_predictions(self, limit=10):
        """Get recent predictions with actual draw number when linked."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('''
            SELECT p.id, p.predicted_numbers, p.method, p.target_draw_at, p.actual_draw_id,
                   p.confidence, p.created_at, d.numbers AS actual_number
            FROM predictions p
            LEFT JOIN draws d ON p.actual_draw_id = d.id
            ORDER BY p.created_at DESC LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df

    def get_prediction_dashboard_rows(self):
        """
        One row per prediction method for the home dashboard: latest saved prediction for each
        method key, plus a dedicated row for Auto (requested_method='auto').
        Rows without any saved prediction yet show empty placeholders.
        """
        method_labels = {
            'ml_lstm': 'ML: LSTM Neural Network',
            'ml_rf': 'ML: Random Forest',
            'ensemble': 'Ensemble (All Methods)',
            'weighted': 'Weighted (Frequency-based)',
            'pattern': 'Pattern-based',
            'hot': 'Hot Numbers',
            'random': 'Random',
        }

        def _clean_scalar(val):
            if val is None:
                return None
            if isinstance(val, float) and pd.isna(val):
                return None
            return val

        conn = sqlite3.connect(self.db_path)
        # (row_key, display_label, sql_extra, params) — row_key 'auto' uses requested_method
        specs = [
            ('auto', 'Auto (Best Available)', 'p.requested_method = ?', ('auto',)),
            ('ml_lstm', 'ML: LSTM Neural Network', 'p.method = ?', ('ml_lstm',)),
            ('ml_rf', 'ML: Random Forest', 'p.method = ?', ('ml_rf',)),
            ('ensemble', 'Ensemble (All Methods)', 'p.method = ?', ('ensemble',)),
            ('weighted', 'Weighted (Frequency-based)', 'p.method = ?', ('weighted',)),
            ('pattern', 'Pattern-based', 'p.method = ?', ('pattern',)),
            ('hot', 'Hot Numbers', 'p.method = ?', ('hot',)),
            ('random', 'Random', 'p.method = ?', ('random',)),
        ]
        rows = []
        for row_key, label, where_sql, params in specs:
            df = pd.read_sql(
                f'''
                SELECT p.id, p.predicted_numbers, p.method, p.target_draw_at, p.actual_draw_id,
                       p.confidence, p.created_at, d.numbers AS actual_number
                FROM predictions p
                LEFT JOIN draws d ON p.actual_draw_id = d.id
                WHERE {where_sql}
                ORDER BY p.id DESC
                LIMIT 1
                ''',
                conn,
                params=params,
            )
            if df is None or df.empty:
                rows.append(
                    {
                        'row_key': row_key,
                        'method_label': label,
                        'has_data': False,
                        'created_at': None,
                        'target_draw_at': None,
                        'resolved_method': None,
                        'confidence': None,
                        'predicted_numbers': None,
                        'actual_number': None,
                    }
                )
            else:
                r = df.iloc[0]
                resolved = _clean_scalar(r.get('method'))
                conf = _clean_scalar(r.get('confidence'))
                actual = r.get('actual_number')
                if actual is not None and isinstance(actual, float) and pd.isna(actual):
                    actual = None
                rows.append(
                    {
                        'row_key': row_key,
                        'method_label': label,
                        'has_data': True,
                        'created_at': r.get('created_at'),
                        'target_draw_at': r.get('target_draw_at'),
                        'resolved_method': resolved,
                        'resolved_label': method_labels.get(resolved, resolved),
                        'confidence': conf,
                        'predicted_numbers': r.get('predicted_numbers'),
                        'actual_number': actual,
                    }
                )
        conn.close()
        return rows

    def get_prediction_history(self, limit=200):
        """Return recent predictions joined to their actual draws (if linked). Used for History page."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            '''
            SELECT
                p.id,
                p.method,
                p.requested_method,
                p.predicted_numbers,
                p.target_draw_at,
                p.confidence,
                p.created_at,
                p.actual_draw_id,
                d.id AS draw_id,
                d.date AS draw_date,
                d.game AS draw_game,
                d.numbers AS actual_number
            FROM predictions p
            LEFT JOIN draws d ON p.actual_draw_id = d.id
            ORDER BY p.created_at DESC
            LIMIT ?
            ''',
            conn,
            params=(limit,),
        )
        conn.close()
        return df

    def get_prediction_history_page(self, page=1, per_page=100):
        """Get paginated prediction history and total count."""
        try:
            page = max(1, int(page))
            per_page = max(1, int(per_page))
        except (TypeError, ValueError):
            page, per_page = 1, 100

        offset = (page - 1) * per_page
        conn = sqlite3.connect(self.db_path)
        count_row = pd.read_sql('SELECT COUNT(*) AS total FROM predictions', conn)
        total = int(count_row.iloc[0]['total']) if not count_row.empty else 0
        df = pd.read_sql(
            '''
            SELECT
                p.id,
                p.method,
                p.requested_method,
                p.predicted_numbers,
                p.target_draw_at,
                p.confidence,
                p.created_at,
                p.actual_draw_id,
                d.id AS draw_id,
                d.date AS draw_date,
                d.game AS draw_game,
                d.numbers AS actual_number
            FROM predictions p
            LEFT JOIN draws d ON p.actual_draw_id = d.id
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
            ''',
            conn,
            params=(per_page, offset),
        )
        conn.close()
        return df, total

    def get_confidence_bucket_accuracy(self, min_samples=5):
        """Return hit-rate by confidence bucket (low/medium/high)."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            '''
            SELECT p.predicted_numbers, p.confidence, d.numbers AS actual_number
            FROM predictions p
            JOIN draws d ON p.actual_draw_id = d.id
            WHERE p.confidence IS NOT NULL
            ''',
            conn
        )
        conn.close()
        if df.empty:
            return {}

        buckets = {
            'low': {'hits': 0, 'total': 0},
            'medium': {'hits': 0, 'total': 0},
            'high': {'hits': 0, 'total': 0},
        }

        def bucket_name(conf):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                return None
            if c < 0.45:
                return 'low'
            if c < 0.65:
                return 'medium'
            return 'high'

        for _, row in df.iterrows():
            b = bucket_name(row.get('confidence'))
            actual = row.get('actual_number')
            pred_str = row.get('predicted_numbers')
            if b is None or pd.isna(actual) or not pred_str:
                continue
            try:
                preds = {int(x.strip()) for x in str(pred_str).split(',') if x.strip()}
                buckets[b]['total'] += 1
                if int(actual) in preds:
                    buckets[b]['hits'] += 1
            except (TypeError, ValueError):
                continue

        out = {}
        for name, s in buckets.items():
            if s['total'] >= min_samples:
                out[name] = {
                    'hits': s['hits'],
                    'total': s['total'],
                    'hit_rate': (s['hits'] / s['total']) if s['total'] else None
                }
        return out

    def _link_predictions_to_draw(self, cursor, draw_id, target_draw_at_str):
        """Link predictions that were for this draw (target_draw_at match) to actual_draw_id."""
        target_draw_at_str = (target_draw_at_str or "").strip()
        if not target_draw_at_str:
            return
        # Match exact or trimmed (in case of stored whitespace)
        cursor.execute('''
            UPDATE predictions SET actual_draw_id = ?
            WHERE TRIM(COALESCE(target_draw_at, '')) = ? AND (actual_draw_id IS NULL OR actual_draw_id = 0)
        ''', (draw_id, target_draw_at_str))

    def relink_predictions_to_draws(self):
        """Re-run linking: for every draw, link any prediction whose target_draw_at matches that draw's date+time."""
        from schedule import target_draw_at_from_date_and_game
        draws_df = self.get_all_draws()
        if draws_df.empty:
            return 0
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        total_linked = 0
        for _, row in draws_df.iterrows():
            target_at = target_draw_at_from_date_and_game(str(row['date']), str(row['game']))
            if not target_at:
                continue
            target_at = target_at.strip()
            c.execute('''
                UPDATE predictions SET actual_draw_id = ?
                WHERE TRIM(COALESCE(target_draw_at, '')) = ?
            ''', (row['id'], target_at))
            total_linked += c.rowcount
        conn.commit()
        conn.close()
        return total_linked

    def get_method_accuracy(self, min_samples=5):
        """Return hit rate per method (actual number in predicted list). Dict method -> (hits, total)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT p.method, p.predicted_numbers, d.numbers
            FROM predictions p
            JOIN draws d ON p.actual_draw_id = d.id
            WHERE p.predicted_numbers IS NOT NULL AND p.predicted_numbers != ''
        ''')
        rows = c.fetchall()
        conn.close()
        from collections import defaultdict
        correct = defaultdict(int)
        total = defaultdict(int)
        for method, pred_str, actual in rows:
            if actual is None:
                continue
            total[method] += 1
            pred_set = set(int(x.strip()) for x in str(pred_str).split(',') if x.strip())
            if int(actual) in pred_set:
                correct[method] += 1
        return {m: (correct[m], total[m]) for m in total if total[m] >= min_samples}

    def get_prediction_accuracy_summary(self, min_samples=1):
        """
        Overall and per-method prediction accuracy.

        Returns a dict:
        {
          "total_predictions": int,
          "total_hits": int,
          "overall_hit_rate": float or None,
          "per_method": {
              method_name: {"hits": int, "total": int, "hit_rate": float or None}
          }
        }
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT p.method, p.predicted_numbers, d.numbers
            FROM predictions p
            JOIN draws d ON p.actual_draw_id = d.id
            WHERE p.predicted_numbers IS NOT NULL AND p.predicted_numbers != ''
        ''')
        rows = c.fetchall()
        conn.close()

        from collections import defaultdict
        correct = defaultdict(int)
        total = defaultdict(int)

        for method, pred_str, actual in rows:
            if actual is None:
                continue
            try:
                total[method] += 1
                pred_set = set(
                    int(x.strip())
                    for x in str(pred_str).split(',')
                    if x.strip()
                )
                if int(actual) in pred_set:
                    correct[method] += 1
            except (ValueError, TypeError):
                # Skip malformed prediction strings gracefully
                continue

        total_predictions = sum(total.values())
        total_hits = sum(correct.values())
        overall_hit_rate = (total_hits / total_predictions) if total_predictions else None

        per_method = {}
        for m in total:
            if total[m] >= min_samples:
                hit_rate = (correct[m] / total[m]) if total[m] else None
                per_method[m] = {
                    "hits": correct[m],
                    "total": total[m],
                    "hit_rate": hit_rate,
                }

        return {
            "total_predictions": total_predictions,
            "total_hits": total_hits,
            "overall_hit_rate": overall_hit_rate,
            "per_method": per_method,
        }

    def find_duplicate_events(self):
        """Find duplicate event IDs in the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''SELECT event, COUNT(*) as count
                  FROM draws
                  GROUP BY event
                  HAVING COUNT(*) > 1''')
        
        duplicates = c.fetchall()
        conn.close()
        return duplicates
    
    def cleanup_duplicates(self):
        """Remove duplicate entries, keeping the ealiest one"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Keep the earliest entry for each event
        c.execute('''
                  DELETE FROM draws
                  WHERE id NOT IN (
                  SELECT MIN(id)
                  FROM draws
                  GROUP BY event
                  )
                  ''')
        
        deleted_count = c.rowcount
        conn.commit()
        conn.close()
        return deleted_count

    def delete_draw(self, draw_id):
        """Delete a single draw by id. Returns True if a row was deleted."""
        try:
            draw_id = int(draw_id)
        except (TypeError, ValueError):
            return False
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM draws WHERE id = ?', (draw_id,))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        return deleted    
    
    def cleanup_hash_duplicates(self):
        """Remove entries with '#' in event ID and keep the clean versions"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Find entries with '#' in event ID
        c.execute('''SELECT id, event FROM draws WHERE event LIKE '#%' ''')
        hash_entries = c.fetchall()

        deleted_count = 0

        for entry_id, event_with_hash in hash_entries:
            clean_event = event_with_hash[1:] # Remove the '#' character

            # Check if clean version exists
            c.execute('''SELECT id FROM draws WHERE event = ?''', (clean_event,))
            clean_entry = c.fetchone()

            if clean_entry:
                # CLean version exists, delete the hash version
                c.execute('''DELETE FROM draws WHERE id = ?''', (entry_id,))
                deleted_count += 1
                print(f"DEBUG: Deleted duplicate #{clean_event} (ID: {entry_id})")
            else:
                # No clean version exists, update the hash version to clean version
                c.execute('''UPDATE draws SET event = ? WHERE id = ?''', (clean_event, entry_id))

        conn.commit()
        conn.close()
        return deleted_count

    def save_benchmark_snapshot(
        self,
        mode,
        count,
        start_size,
        max_evals,
        rf_retrain_every,
        seeds,
        summary,
        decision=None,
        reason=None
    ):
        """Persist a benchmark result and optional promotion decision."""
        ensemble = summary.get('ensemble', {}) if isinstance(summary, dict) else {}
        lstm = summary.get('ml_lstm', {}) if isinstance(summary, dict) else {}
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''
            INSERT INTO benchmarks (
                mode, count, start_size, max_evals, rf_retrain_every,
                seeds_json, summary_json, decision, reason,
                ensemble_mean, ensemble_std, lstm_mean
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                str(mode),
                int(count),
                int(start_size),
                int(max_evals) if max_evals is not None else None,
                int(rf_retrain_every) if rf_retrain_every is not None else None,
                json.dumps(list(seeds or [])),
                json.dumps(summary or {}),
                decision,
                reason,
                ensemble.get('mean_hit_rate'),
                ensemble.get('std_hit_rate'),
                lstm.get('mean_hit_rate'),
            )
        )
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_latest_benchmark(self):
        """Return latest benchmark snapshot row as dict, or None."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM benchmarks ORDER BY id DESC LIMIT 1')
        row = c.fetchone()
        conn.close()
        if row is None:
            return None
        result = dict(row)
        for field in ('seeds_json', 'summary_json'):
            try:
                result[field] = json.loads(result[field]) if result.get(field) else None
            except json.JSONDecodeError:
                result[field] = None
        return result

    def get_latest_promoted_benchmark(self):
        """Return latest promoted benchmark snapshot row as dict, or None."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM benchmarks WHERE decision = 'promote' ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row is None:
            return None
        result = dict(row)
        for field in ('seeds_json', 'summary_json'):
            try:
                result[field] = json.loads(result[field]) if result.get(field) else None
            except json.JSONDecodeError:
                result[field] = None
        return result

    def get_recent_benchmarks(self, limit=10):
        """Return recent benchmark snapshots (newest first)."""
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 10
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM benchmarks ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        out = []
        for row in rows:
            item = dict(row)
            for field in ('seeds_json', 'summary_json'):
                try:
                    item[field] = json.loads(item[field]) if item.get(field) else None
                except json.JSONDecodeError:
                    item[field] = None
            out.append(item)
        return out

    def get_latest_matching_benchmark(self, mode, count, start_size, max_evals, rf_retrain_every):
        """
        Return latest benchmark matching a profile/config so comparisons stay apples-to-apples.
        Matching keys: mode, count, start_size, max_evals, rf_retrain_every.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            '''
            SELECT * FROM benchmarks
            WHERE mode = ?
              AND count = ?
              AND start_size = ?
              AND COALESCE(max_evals, -1) = COALESCE(?, -1)
              AND COALESCE(rf_retrain_every, -1) = COALESCE(?, -1)
            ORDER BY id DESC
            LIMIT 1
            ''',
            (str(mode), int(count), int(start_size), max_evals, rf_retrain_every),
        )
        row = c.fetchone()
        conn.close()
        if row is None:
            return None
        result = dict(row)
        for field in ('seeds_json', 'summary_json'):
            try:
                result[field] = json.loads(result[field]) if result.get(field) else None
            except json.JSONDecodeError:
                result[field] = None
        return result

    def get_recent_method_hit_rates(self, window=200):
        """
        Compute hit-rate per method over the most recent linked predictions.
        Returns dict: method -> {hits, total, hit_rate}.
        """
        try:
            window = max(1, int(window))
        except (TypeError, ValueError):
            window = 200

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            '''
            SELECT p.method, p.predicted_numbers, d.numbers AS actual_number, p.created_at
            FROM predictions p
            JOIN draws d ON p.actual_draw_id = d.id
            WHERE p.predicted_numbers IS NOT NULL AND p.predicted_numbers != ''
            ORDER BY p.created_at DESC
            LIMIT ?
            ''',
            conn,
            params=(window,),
        )
        conn.close()

        if df.empty:
            return {}

        stats = {}
        for _, row in df.iterrows():
            method = row.get('method')
            if not method:
                continue
            if method not in stats:
                stats[method] = {'hits': 0, 'total': 0, 'hit_rate': None}
            try:
                preds = {int(x.strip()) for x in str(row.get('predicted_numbers', '')).split(',') if x.strip()}
                actual = int(row.get('actual_number'))
            except (TypeError, ValueError):
                continue
            stats[method]['total'] += 1
            if actual in preds:
                stats[method]['hits'] += 1

        for method, s in stats.items():
            s['hit_rate'] = (s['hits'] / s['total']) if s['total'] else None
        return stats