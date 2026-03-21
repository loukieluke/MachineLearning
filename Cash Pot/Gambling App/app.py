from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import io
import csv
import sqlite3
import threading
import json
import os
from models import DatabaseManager
from predictor import LotteryPredictor
from schedule import get_next_draw_datetime, format_target_draw_at

app = Flask(__name__)
app.secret_key = 'your-secret-key-here' # change this in production

# Initialize database and predictor
db = DatabaseManager()
startup_training_enabled = os.environ.get('ENABLE_STARTUP_TRAINING', '1') == '1'
startup_training_reason = 'enabled via environment'
# In debug mode with reloader, only run startup training in the reloader child process.
if os.environ.get('WERKZEUG_RUN_MAIN') is not None and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    startup_training_enabled = False
    startup_training_reason = 'disabled in reloader parent process'
elif not startup_training_enabled:
    startup_training_reason = 'disabled via ENABLE_STARTUP_TRAINING=0'
predictor = LotteryPredictor(db, auto_train_startup=startup_training_enabled)

BENCHMARK_PROFILES = {
    'quick': {
        'mode': 'full',
        'rf_retrain_every': 150,
        'max_evals': 200,
        'seeds': [42, 43, 44],
        'min_gain': 0.004,
        'max_std_increase': 0.012,
    },
    'standard': {
        'mode': 'full',
        'rf_retrain_every': 100,
        'max_evals': 400,
        'seeds': [40, 41, 42, 43, 44],
        'min_gain': 0.003,
        'max_std_increase': 0.010,
    },
    'deep': {
        'mode': 'full',
        'rf_retrain_every': 80,
        'max_evals': 800,
        'seeds': [40, 41, 42, 43, 44, 45, 46],
        'min_gain': 0.002,
        'max_std_increase': 0.008,
    },
}

# Training status (for UI feedback when retrain runs in background)
_training_lock = threading.Lock()
_training_in_progress = False
_last_training_result = None   # 'success' | 'error' | 'insufficient_data'
_last_training_message = None  # optional detail message

@app.route('/')
def index():
    stats = predictor.get_stats()
    recent_predictions = db.get_recent_predictions(5)
    accuracy_summary = db.get_prediction_accuracy_summary()
    benchmark_baseline = db.get_latest_benchmark()
    return render_template(
        'index.html',
        stats=stats,
        recent_predictions=recent_predictions,
        accuracy_summary=accuracy_summary,
        benchmark_baseline=benchmark_baseline,
        startup_training_enabled=startup_training_enabled,
        startup_training_reason=startup_training_reason
    )

ALLOWED_COUNTS = (3, 5, 7, 10)

@app.route('/predict', methods=['POST'])
def predict_numbers():
    method = request.form.get('method', 'auto')
    try:
        count = int(request.form.get('count', 5))
    except (TypeError, ValueError):
        count = 5
    if count not in ALLOWED_COUNTS:
        count = 5

    target_draw_at = format_target_draw_at(get_next_draw_datetime())

    try:
        numbers = predictor.predict(method, count, target_draw_at=target_draw_at)
        flash('Prediction generated successfully!', 'success')
        return jsonify({
            'success': True,
            'numbers': [int(num) for num in numbers],
            'method': predictor.last_prediction_method or method,
            'confidence': predictor.last_prediction_confidence,
            'count': count,
            'target_draw_at': target_draw_at
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Read CSV file
                stream = io.StringIO(file.stream.read().decode('UTF8'), newline=None)
                csv_input = csv.DictReader(stream)

                # Convert to a list of dictionaries
                data = [row for row in csv_input if row['numbers'].strip()]

                # Add to database
                added_count = db.add_draws_from_csv(data)

                # Update predictor with the new data
                predictor.update_data()

                flash(f'Successfully added {added_count} new draws!', 'success')
                return redirect(url_for('index'))
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/add_single', methods=['POST'])
def add_single_draw():
    try:
        draw_data = {
            'date': request.form['date'],
            'game': request.form['game'],
            'name': request.form['name'],
            'numbers': int(request.form['numbers']),
            'event': request.form['event'] 
        }

        draw_id = db.add_draw(draw_data)
        if draw_id is not None:
            predictor.update_data()
            flash('Draw added successfully! Predictions for this draw have been linked.', 'success')
        else:
            flash('Draw already exists in database', 'warning')

    except Exception as e:
        flash(f'Error adding draw: {str(e)}', 'error')

    return redirect(url_for('history'))


@app.route('/edit_draw', methods=['POST'])
def edit_draw():
    """Update the drawn number for an existing draw."""
    try:
        draw_id = request.form.get('draw_id', type=int)
        numbers = request.form.get('numbers', type=int)
        if draw_id is None or numbers is None or not (1 <= numbers <= 36):
            flash('Invalid draw or number. Number must be between 1 and 36.', 'error')
            return redirect(url_for('history'))
        if db.update_draw(draw_id, numbers):
            predictor.update_data()
            flash('Draw updated successfully.', 'success')
        else:
            flash('Draw not found or could not be updated.', 'error')
    except Exception as e:
        flash(f'Error updating draw: {str(e)}', 'error')
    return redirect(url_for('history'))

@app.route('/delete_draw', methods=['POST'])
def delete_draw():
    """Delete a draw."""
    try:
        draw_id = request.form.get('draw_id', type=int)
        if draw_id is None:
            flash('Invalid draw selected.', 'error')
            return redirect(url_for('history'))
        if db.delete_draw(draw_id):
            predictor.update_data()
            flash('Draw removed successfully.', 'success')
        else:
            flash('Draw not found or could not be removed.', 'error')
    except Exception as e:
        flash(f'Error removing draw: {str(e)}', 'error')
    return redirect(url_for('history'))

@app.route('/history')
def history():
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=50, type=int)
    if per_page not in (25, 50, 100):
        per_page = 50
    if page < 1:
        page = 1
    draws, total_draws = db.get_draws_page(page=page, per_page=per_page)
    total_pages = max(1, (total_draws + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
        draws, total_draws = db.get_draws_page(page=page, per_page=per_page)
    accuracy_summary = db.get_prediction_accuracy_summary()
    return render_template(
        'history.html',
        draws=draws,
        total_draws=total_draws,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        accuracy_summary=accuracy_summary,
    )


@app.route('/relink_predictions', methods=['POST'])
def relink_predictions():
    """Re-link predictions to draws by date+game so Actual numbers show correctly."""
    try:
        linked = db.relink_predictions_to_draws()
        predictor.update_data()
        flash(f'Re-linked {linked} prediction(s) to draws. Refresh the page to see updates.', 'success')
    except Exception as e:
        flash(f'Error re-linking: {str(e)}', 'error')
    return redirect(url_for('prediction_history_page'))


@app.route('/prediction-history')
def prediction_history_page():
    """Dedicated page for Predictions vs Actual Draws table."""
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=100, type=int)
    if per_page not in (50, 100, 200):
        per_page = 100
    if page < 1:
        page = 1
    prediction_history, total_predictions = db.get_prediction_history_page(page=page, per_page=per_page)
    total_pages = max(1, (total_predictions + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
        prediction_history, total_predictions = db.get_prediction_history_page(page=page, per_page=per_page)
    accuracy_summary = db.get_prediction_accuracy_summary()
    confidence_bucket_accuracy = db.get_confidence_bucket_accuracy(min_samples=5)

    if not prediction_history.empty:
        hit_list = []
        conf_band_list = []
        for _, row in prediction_history.iterrows():
            actual = row.get('actual_number')
            if pd.isna(actual):
                hit_list.append(False)
            else:
                try:
                    preds = {
                        int(x.strip())
                        for x in str(row['predicted_numbers']).split(',')
                        if x.strip()
                    }
                    hit_list.append(int(actual) in preds)
                except (ValueError, TypeError):
                    hit_list.append(False)

            try:
                conf = float(row.get('confidence')) if row.get('confidence') is not None else None
                if conf is None:
                    conf_band_list.append(None)
                elif conf < 0.45:
                    conf_band_list.append('low')
                elif conf < 0.65:
                    conf_band_list.append('medium')
                else:
                    conf_band_list.append('high')
            except (ValueError, TypeError):
                conf_band_list.append(None)
        prediction_history = prediction_history.copy()
        prediction_history['hit'] = hit_list
        prediction_history['confidence_band'] = conf_band_list

    return render_template(
        'prediction_history.html',
        prediction_history=prediction_history if not prediction_history.empty else None,
        total_predictions=total_predictions,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        accuracy_summary=accuracy_summary,
        confidence_bucket_accuracy=confidence_bucket_accuracy,
    )

@app.route('/api/stats')
def api_stats():
    stats = predictor.get_stats()
    return jsonify(stats)

@app.route('/api/draws')
def api_draws():
    draws = db.get_all_draws()
    return jsonify(draws.to_dict('records'))

def _run_training():
    """Background thread: run force_retrain and update training status."""
    global _training_in_progress, _last_training_result, _last_training_message
    try:
        success = predictor.force_retrain()
        with _training_lock:
            _training_in_progress = False
            _last_training_result = 'success' if success else 'insufficient_data'
            _last_training_message = None if success else 'Not enough data to retrain models'
    except Exception as e:
        with _training_lock:
            _training_in_progress = False
            _last_training_result = 'error'
            _last_training_message = str(e)


@app.route('/retrain', methods=['POST'])
def retrain_models():
    """Start ML model retraining in the background."""
    global _training_in_progress, _last_training_result, _last_training_message
    with _training_lock:
        if _training_in_progress:
            flash('Training already in progress. Please wait.', 'info')
            return redirect(url_for('index'))
        _training_in_progress = True
        _last_training_result = None
        _last_training_message = None
    thread = threading.Thread(target=_run_training)
    thread.daemon = True
    thread.start()
    flash('Training started in the background. This may take a few minutes.', 'info')
    return redirect(url_for('index'))


@app.route('/api/training_status')
def api_training_status():
    """Return whether training is in progress and last result (for UI polling)."""
    with _training_lock:
        return jsonify({
            'training': _training_in_progress,
            'last_result': _last_training_result,
            'last_message': _last_training_message or ''
        })

@app.route('/api/model_status')
def api_model_status():
    """Get ML model status"""
    stats = predictor.get_stats()
    preferred_auto_method = predictor.get_preferred_auto_method()
    return jsonify({
        'ml_loaded': stats.get('ml_model_loaded', False),
        'rf_trained': stats.get('rf_model_trained', False),
        'total_draws': stats['total_draws'],
        'min_draws_for_ml': 20,
        'preferred_auto_method': preferred_auto_method
    })

@app.route('/api/backtest')
def api_backtest():
    """Walk-forward backtest for quick accuracy checks."""
    try:
        count = request.args.get('count', default=5, type=int)
        start_size = request.args.get('start_size', default=60, type=int)
        mode = request.args.get('mode', default='quick', type=str)
        rf_retrain_every = request.args.get('rf_retrain_every', default=25, type=int)
        seed = request.args.get('seed', default=None, type=int)
        max_evals = request.args.get('max_evals', default=None, type=int)
        if count not in (3, 5, 7, 10):
            count = 5
        if start_size < 20:
            start_size = 20
        if mode not in ('quick', 'full'):
            mode = 'quick'
        if rf_retrain_every < 1:
            rf_retrain_every = 25
        result = predictor.backtest_walk_forward(
            count=count,
            start_size=start_size,
            mode=mode,
            rf_retrain_every=rf_retrain_every,
            seed=seed,
            max_evals=max_evals
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest_multi')
def api_backtest_multi():
    """Run deterministic multi-seed backtests and return aggregate stats."""
    try:
        count = request.args.get('count', default=5, type=int)
        start_size = request.args.get('start_size', default=60, type=int)
        mode = request.args.get('mode', default='quick', type=str)
        rf_retrain_every = request.args.get('rf_retrain_every', default=25, type=int)
        max_evals = request.args.get('max_evals', default=None, type=int)
        seeds_str = request.args.get('seeds', default='42,43,44', type=str)
        profile = request.args.get('profile', default=None, type=str)

        if count not in (3, 5, 7, 10):
            count = 5
        if start_size < 20:
            start_size = 20
        if mode not in ('quick', 'full'):
            mode = 'quick'
        if rf_retrain_every < 1:
            rf_retrain_every = 25

        if profile in BENCHMARK_PROFILES:
            p = BENCHMARK_PROFILES[profile]
            mode = p['mode']
            rf_retrain_every = p['rf_retrain_every']
            max_evals = p['max_evals']
            seeds = list(p['seeds'])
        else:
            seeds = []
            for s in (seeds_str or '').split(','):
                s = s.strip()
                if not s:
                    continue
                try:
                    seeds.append(int(s))
                except ValueError:
                    continue

        result = predictor.backtest_multi_seed(
            seeds=seeds,
            count=count,
            start_size=start_size,
            mode=mode,
            rf_retrain_every=rf_retrain_every,
            max_evals=max_evals,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/benchmark/baseline')
def api_benchmark_baseline():
    """Return latest stored benchmark snapshot."""
    latest = db.get_latest_benchmark()
    return jsonify({'latest': latest})

@app.route('/api/benchmark/evaluate')
def api_benchmark_evaluate():
    """
    Run multi-seed benchmark and return promotion decision.
    Decision logic compares against latest stored baseline.
    """
    try:
        count = request.args.get('count', default=5, type=int)
        start_size = request.args.get('start_size', default=60, type=int)
        mode = request.args.get('mode', default='full', type=str)
        rf_retrain_every = request.args.get('rf_retrain_every', default=100, type=int)
        max_evals = request.args.get('max_evals', default=300, type=int)
        seeds_str = request.args.get('seeds', default='42,43,44', type=str)
        profile = request.args.get('profile', default=None, type=str)
        min_gain = request.args.get('min_gain', default=0.005, type=float)  # +0.5%
        max_std_increase = request.args.get('max_std_increase', default=0.01, type=float)
        persist = request.args.get('persist', default=1, type=int)

        if count not in (3, 5, 7, 10):
            count = 5
        if start_size < 20:
            start_size = 20
        if mode not in ('quick', 'full'):
            mode = 'full'
        if rf_retrain_every < 1:
            rf_retrain_every = 100
        if max_evals is not None and max_evals < 50:
            max_evals = 50

        if profile in BENCHMARK_PROFILES:
            p = BENCHMARK_PROFILES[profile]
            mode = p['mode']
            rf_retrain_every = p['rf_retrain_every']
            max_evals = p['max_evals']
            seeds = list(p['seeds'])
            if request.args.get('min_gain') is None:
                min_gain = p.get('min_gain', min_gain)
            if request.args.get('max_std_increase') is None:
                max_std_increase = p.get('max_std_increase', max_std_increase)
        else:
            seeds = []
            for s in (seeds_str or '').split(','):
                s = s.strip()
                if not s:
                    continue
                try:
                    seeds.append(int(s))
                except ValueError:
                    continue
        if not seeds:
            seeds = [42, 43, 44]

        result = predictor.backtest_multi_seed(
            seeds=seeds,
            count=count,
            start_size=start_size,
            mode=mode,
            rf_retrain_every=rf_retrain_every,
            max_evals=max_evals,
        )
        summary = result.get('summary', {})
        ensemble = summary.get('ensemble', {})
        current_mean = ensemble.get('mean_hit_rate')
        current_std = ensemble.get('std_hit_rate')

        latest = db.get_latest_matching_benchmark(
            mode=mode,
            count=count,
            start_size=start_size,
            max_evals=max_evals,
            rf_retrain_every=rf_retrain_every,
        )
        decision = 'hold'
        reason = 'Insufficient benchmark summary data.'
        baseline_mean = None
        baseline_std = None

        if current_mean is not None and current_std is not None:
            if latest is None or latest.get('ensemble_mean') is None:
                decision = 'promote'
                reason = 'No matching-profile baseline found; promoting first baseline for this profile/config.'
            else:
                baseline_mean = float(latest.get('ensemble_mean'))
                baseline_std = float(latest.get('ensemble_std') or 0.0)
                gain = current_mean - baseline_mean
                std_delta = current_std - baseline_std
                if gain >= min_gain and std_delta <= max_std_increase:
                    decision = 'promote'
                    reason = (
                        f'Ensemble mean improved by {gain:.4f} '
                        f'with acceptable std change ({std_delta:.4f}).'
                    )
                else:
                    decision = 'hold'
                    reason = (
                        f'Gain {gain:.4f} (needed {min_gain:.4f}), '
                        f'std delta {std_delta:.4f} (limit {max_std_increase:.4f}).'
                    )

        snapshot_id = None
        if persist:
            snapshot_id = db.save_benchmark_snapshot(
                mode=mode,
                count=count,
                start_size=start_size,
                max_evals=max_evals,
                rf_retrain_every=rf_retrain_every,
                seeds=seeds,
                summary=summary,
                decision=decision,
                reason=reason,
            )

        return jsonify({
            'decision': decision,
            'reason': reason,
            'profile': profile if profile in BENCHMARK_PROFILES else None,
            'baseline': {
                'ensemble_mean': baseline_mean,
                'ensemble_std': baseline_std,
                'latest_snapshot_id': latest.get('id') if latest else None,
                'matching_profile': bool(latest is not None),
            },
            'current': {
                'ensemble_mean': current_mean,
                'ensemble_std': current_std,
            },
            'snapshot_id': snapshot_id,
            'result': result,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/benchmark/history')
def api_benchmark_history():
    """Return recent benchmark snapshots for auditing trend over time."""
    limit = request.args.get('limit', default=10, type=int)
    rows = db.get_recent_benchmarks(limit=limit)
    return jsonify({'items': rows})

@app.route('/api/monitor/health')
def api_monitor_health():
    """
    Live monitoring endpoint:
    compares recent linked prediction hit-rates with latest promoted benchmark.
    """
    window = request.args.get('window', default=200, type=int)
    tolerance_warn = request.args.get('tolerance_warn', default=0.015, type=float)
    tolerance_critical = request.args.get('tolerance_critical', default=0.03, type=float)
    if window < 20:
        window = 20

    recent = db.get_recent_method_hit_rates(window=window)
    promoted = db.get_latest_promoted_benchmark()

    baseline_ensemble = None
    baseline_lstm = None
    if promoted and isinstance(promoted.get('summary_json'), dict):
        baseline_ensemble = promoted['summary_json'].get('ensemble', {}).get('mean_hit_rate')
        baseline_lstm = promoted['summary_json'].get('ml_lstm', {}).get('mean_hit_rate')

    current_ensemble = recent.get('ensemble', {}).get('hit_rate')
    current_lstm = recent.get('ml_lstm', {}).get('hit_rate')

    status = 'healthy'
    reason = 'No promoted baseline yet; monitoring recent rates only.'

    if baseline_ensemble is not None and current_ensemble is not None:
        gap = float(baseline_ensemble) - float(current_ensemble)
        if gap >= tolerance_critical:
            status = 'critical'
            reason = f'Ensemble recent hit-rate is down by {gap:.4f} vs promoted baseline.'
        elif gap >= tolerance_warn:
            status = 'warning'
            reason = f'Ensemble recent hit-rate is down by {gap:.4f} vs promoted baseline.'
        else:
            status = 'healthy'
            reason = f'Ensemble recent hit-rate is within tolerance (gap {gap:.4f}).'
    elif baseline_ensemble is not None:
        status = 'warning'
        reason = 'No recent linked ensemble predictions found for comparison.'

    return jsonify({
        'status': status,
        'reason': reason,
        'window': window,
        'baseline': {
            'snapshot_id': promoted.get('id') if promoted else None,
            'ensemble_mean': baseline_ensemble,
            'ml_lstm_mean': baseline_lstm,
        },
        'current_recent': {
            'ensemble_hit_rate': current_ensemble,
            'ml_lstm_hit_rate': current_lstm,
            'per_method': recent,
        },
        'thresholds': {
            'tolerance_warn': tolerance_warn,
            'tolerance_critical': tolerance_critical,
        },
    })

@app.route('/cleanup', methods=['POST'])
def cleanup_duplicates():
    """Clean up duplicate event IDs"""
    try:
        deleted_count = db.cleanup_duplicates()
        predictor.update_data() # Refresh data after cleanup
        flash(f'Cleaned up {deleted_count} duplicate entries!', 'success')
    except Exception as e:
        flash(f'Error cleaning duplicates: {str(e)}', 'error')

    return redirect(url_for('history'))

@app.route('/cleanup_hash', methods=['POST'])
def cleanup_hash_duplicates():
    """Clean up entries with '#' in the event IDs"""
    try:
        deleted_count = db.cleanup_hash_duplicates()
        predictor.update_data() # Refresh data after cleanup
        if deleted_count > 0:
            flash(f'Cleaned up {deleted_count} duplicate entries with "#"!', 'success')
        else:
            flash('No duplicate entries with "#" found.', 'info')
    except Exception as e:
        flash(f'Error cleaning hash duplicates: {str(e)}', 'error')

    return redirect(url_for('history'))

@app.route('/debug_hash')
def debug_hash():
    """Debug route to check for hash entries"""
    conn = sqlite3.connect('lottery_data.db')
    c = conn.cursor()
    c.execute('''SELECT id, date, game, event FROM draws WHERE event LIKE '#%' ''')
    hash_entries = c.fetchall()
    conn.close()

    return jsonify({
        'hash_entries_count': len(hash_entries),
        'hash_entries': hash_entries
    })


if __name__ == '__main__':
    app.run(debug=True)    
    


