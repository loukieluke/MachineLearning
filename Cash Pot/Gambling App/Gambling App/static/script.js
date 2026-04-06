// Main application JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Prediction form handling
    const predictionForm = document.getElementById('prediction-form');
    const predictionResult = document.getElementById('prediction-result');
    const predictionNumbers = document.querySelector('.prediction-numbers');
    const predictionMethod = document.getElementById('prediction-method');
    const generateAllButton = document.getElementById('generate-all-btn');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(predictionForm);
            const submitButton = predictionForm.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            
            // Show loading state
            submitButton.innerHTML = '<span class="loading"></span> Generating...';
            submitButton.disabled = true;
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayPrediction(data.numbers, data.method, data.target_draw_at, data.confidence);
                    showNotification('Prediction generated successfully!', 'success');
                } else {
                    showNotification('Error generating prediction: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('Network error occurred', 'error');
            })
            .finally(() => {
                // Restore button
                submitButton.textContent = originalText;
                submitButton.disabled = false;
            });
        });

        if (generateAllButton) {
            generateAllButton.addEventListener('click', function() {
                const formData = new FormData(predictionForm);
                const submitButton = predictionForm.querySelector('button[type="submit"]');
                const submitOriginalText = submitButton.textContent;
                const allOriginalText = generateAllButton.textContent;

                submitButton.disabled = true;
                generateAllButton.disabled = true;
                generateAllButton.innerHTML = '<span class="loading"></span> Generating all...';

                fetch('/predict_all', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        const errorText = (data.errors || []).map(e => `${e.method}: ${e.error}`).join('; ');
                        showNotification('Error generating all predictions' + (errorText ? `: ${errorText}` : ''), 'error');
                        return;
                    }

                    const generated = Array.isArray(data.generated) ? data.generated : [];
                    const autoResult = generated.find(item => item.requested_method === 'auto') || generated[0];
                    if (autoResult && Array.isArray(autoResult.numbers)) {
                        displayPrediction(
                            autoResult.numbers,
                            autoResult.resolved_method || autoResult.requested_method || 'auto',
                            data.target_draw_at,
                            autoResult.confidence
                        );
                    }

                    const errorCount = Array.isArray(data.errors) ? data.errors.length : 0;
                    const message = errorCount > 0
                        ? `Generated ${generated.length} methods (${errorCount} failed).`
                        : `Generated all ${generated.length} prediction methods.`;
                    showNotification(message, errorCount > 0 ? 'error' : 'success');

                    // Refresh quickly so "Recent Predictions" table updates.
                    setTimeout(() => window.location.reload(), 500);
                })
                .catch(error => {
                    console.error('Error:', error);
                    showNotification('Network error occurred', 'error');
                })
                .finally(() => {
                    submitButton.disabled = false;
                    generateAllButton.disabled = false;
                    submitButton.textContent = submitOriginalText;
                    generateAllButton.textContent = allOriginalText;
                });
            });
        }
    }
    
    function displayPrediction(numbers, method, targetDrawAt, confidence) {
        // Clear previous results
        predictionNumbers.innerHTML = '';
        
        // Create number balls
        numbers.forEach(number => {
            const ball = document.createElement('div');
            ball.className = 'number-ball';
            ball.textContent = number;
            ball.style.animationDelay = (Math.random() * 0.5) + 's';
            predictionNumbers.appendChild(ball);
        });
        
        // Update method text
        const methodNames = {
            'auto': 'Auto (Best Available)',
            'ml_lstm': 'ML: LSTM Neural Network',
            'ml_rf': 'ML: Random Forest',
            'ensemble': 'Ensemble (All Methods)',
            'weighted': 'Weighted Frequency Method',
            'pattern': 'Pattern Analysis Method',
            'hot': 'Hot Numbers Method',
            'random': 'Random Selection'
        };
        
        predictionMethod.textContent = `Generated using: ${methodNames[method] || method}`;

        const confidenceEl = document.getElementById('prediction-confidence');
        if (confidenceEl) {
            if (typeof confidence === 'number') {
                const pct = Math.round(confidence * 100);
                confidenceEl.textContent = `Model confidence: ${pct}%`;
            } else {
                confidenceEl.textContent = '';
            }
        }
        
        const targetEl = document.getElementById('prediction-target-draw');
        if (targetEl) {
            targetEl.textContent = targetDrawAt ? `For Cash Pot draw: ${targetDrawAt}` : '';
        }
        
        // Show result section
        predictionResult.style.display = 'block';
        
        // Smooth scroll to results
        predictionResult.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }
    
    function showNotification(message, type) {
        // Remove any existing JS notification so only one shows at a time
        document.querySelectorAll('.js-notification').forEach(el => el.remove());

        const alert = document.createElement('div');
        alert.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show js-notification`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alert, container.firstChild);
        }

        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
    
    // Auto-refresh stats every 30 seconds
    function refreshStats() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                updateStatsDisplay(data);
            })
            .catch(error => console.error('Error refreshing stats:', error));
    }
    
    function updateStatsDisplay(stats) {
        // Update stats in the UI
        const totalDrawsElement = document.querySelector('strong:contains("Total Draws")');
        if (totalDrawsElement) {
            totalDrawsElement.parentElement.innerHTML = `<strong>Total Draws:</strong> ${stats.total_draws}`;
        }
        
        const dateRangeElement = document.querySelector('strong:contains("Date Range")');
        if (dateRangeElement) {
            dateRangeElement.parentElement.innerHTML = `<strong>Date Range:</strong> ${stats.date_range}`;
        }
    }
    
    // Load model status function (LSTM/RF badges). Optional prefixHtml is shown above the badges.
    function loadModelStatus(prefixHtml) {
        fetch('/api/model_status')
            .then(response => response.json())
            .then(data => {
                const statusElement = document.getElementById('model-status');
                if (!statusElement) return;
                let statusHtml = '';
                const autoNames = {
                    auto: 'Auto',
                    ml_lstm: 'ML: LSTM',
                    ml_rf: 'ML: Random Forest',
                    ensemble: 'Ensemble',
                    weighted: 'Weighted',
                    pattern: 'Pattern',
                    hot: 'Hot',
                    random: 'Random'
                };
                
                if (data.ml_loaded) {
                    statusHtml += '<span class="badge bg-success">LSTM: Loaded</span> ';
                } else {
                    const need = Math.max(0, (data.min_draws_for_ml || 20) - data.total_draws);
                    statusHtml += need > 0
                        ? `<span class="badge bg-warning">LSTM: Need ${need} more draws</span> `
                        : '<span class="badge bg-warning">LSTM: Train via button</span> ';
                }
                
                if (data.rf_trained) {
                    statusHtml += '<span class="badge bg-success">Random Forest: Trained</span>';
                } else {
                    statusHtml += '<span class="badge bg-warning">Random Forest: Training needed</span>';
                }
                if (data.preferred_auto_method) {
                    const label = autoNames[data.preferred_auto_method] || data.preferred_auto_method;
                    statusHtml += ` <span class="badge bg-dark">Auto prefers: ${label}</span>`;
                }
                
                const fullHtml = (prefixHtml ? prefixHtml + '<br>' : '') + `<small>${statusHtml}</small>`;
                statusElement.innerHTML = fullHtml;
            })
            .catch(error => {
                console.error('Error loading model status:', error);
            });
    }

    // Poll training status and update UI (training in progress / finished)
    function pollTrainingStatus() {
        fetch('/api/training_status')
            .then(response => response.json())
            .then(data => {
                const statusElement = document.getElementById('model-status');
                const retrainBtn = document.getElementById('retrain-btn');
                if (!statusElement) return;

                if (data.training) {
                    if (retrainBtn) {
                        retrainBtn.disabled = true;
                        retrainBtn.textContent = 'Training…';
                    }
                    statusElement.innerHTML = '<small><span class="spinner-border spinner-border-sm me-1" role="status"></span> Training in progress… This may take a few minutes.</small>';
                    statusElement.className = 'alert alert-info';
                } else {
                    if (retrainBtn) {
                        retrainBtn.disabled = false;
                        retrainBtn.textContent = 'Force Retrain ML Models';
                    }
                    let prefixHtml = '';
                    if (data.last_result === 'success') {
                        prefixHtml = '<span class="badge bg-success">Last training: finished successfully</span>';
                        statusElement.className = 'alert alert-success';
                    } else if (data.last_result === 'insufficient_data') {
                        prefixHtml = '<span class="badge bg-warning">Last training: not enough data</span>';
                        statusElement.className = 'alert alert-warning';
                    } else if (data.last_result === 'error') {
                        prefixHtml = '<span class="badge bg-danger">Last training: error</span> ' + (data.last_message ? `<small>${escapeHtml(data.last_message)}</small>` : '');
                        statusElement.className = 'alert alert-danger';
                    } else {
                        statusElement.className = 'alert alert-info';
                    }
                    loadModelStatus(prefixHtml);
                }
            })
            .catch(error => console.error('Error polling training status:', error));
    }

    // Poll live monitoring health and show status badge
    function pollMonitorHealth() {
        fetch('/api/monitor/health')
            .then(response => response.json())
            .then(data => {
                const el = document.getElementById('monitor-health-status');
                if (!el) return;
                const status = (data.status || 'unknown').toLowerCase();
                const reason = data.reason || 'No details available.';
                const baseline = data.baseline || {};
                const current = data.current_recent || {};
                const currentEnsemble = current.ensemble_hit_rate;
                const baselineEnsemble = baseline.ensemble_mean;

                let badge = 'secondary';
                if (status === 'healthy') badge = 'success';
                else if (status === 'warning') badge = 'warning';
                else if (status === 'critical') badge = 'danger';

                const currentPct = (typeof currentEnsemble === 'number')
                    ? `${Math.round(currentEnsemble * 1000) / 10}%`
                    : '—';
                const baselinePct = (typeof baselineEnsemble === 'number')
                    ? `${Math.round(baselineEnsemble * 1000) / 10}%`
                    : '—';
                const checkedAt = new Date().toLocaleTimeString();

                el.className = `alert alert-${badge} py-2 mb-2`;
                el.innerHTML = `<small><strong>Monitoring:</strong> ${status.toUpperCase()} | Recent ensemble: ${currentPct} | Baseline: ${baselinePct}<br>${escapeHtml(reason)}<br><span class="text-muted">Last checked: ${checkedAt}</span></small>`;
            })
            .catch(error => {
                const el = document.getElementById('monitor-health-status');
                if (!el) return;
                const checkedAt = new Date().toLocaleTimeString();
                el.className = 'alert alert-secondary py-2 mb-2';
                el.innerHTML = `<small><strong>Monitoring:</strong> unavailable<br><span class="text-muted">Last checked: ${checkedAt}</span></small>`;
                console.error('Error polling monitor health:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Start auto-refresh if on index page
    if (window.location.pathname === '/' || window.location.pathname === '') {
        setInterval(refreshStats, 30000);
        loadModelStatus();
        pollTrainingStatus();
        setInterval(pollTrainingStatus, 2500);
        pollMonitorHealth();
        setInterval(pollMonitorHealth, 10000);
    }
    
    // CSV file validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileName = file.name.toLowerCase();
                if (!fileName.endsWith('.csv')) {
                    showNotification('Please select a CSV file', 'error');
                    e.target.value = '';
                }
            }
        });
    }
    
    // Add some visual effects
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.transition = 'transform 0.2s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});