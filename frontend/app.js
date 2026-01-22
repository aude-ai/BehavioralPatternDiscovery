// API base URL
const API_BASE = window.location.origin;

// State
let currentEngineerId = null;
let statusCheckInterval = null;
let engineerSummaryData = [];  // Store engineer data for sorting
let currentSortColumn = 'activity_count';  // Default sort column
let currentSortDirection = 'desc';  // Default sort direction

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    startStatusPolling();
});

function initializeApp() {
    updateStatus();
    refreshLogs();
}

function setupEventListeners() {
    // Button listeners
    document.getElementById('btn-preprocess').addEventListener('click', preprocessData);
    document.getElementById('btn-normalization').addEventListener('click', applyNormalization);
    document.getElementById('btn-train-vae').addEventListener('click', trainVAE);
    // Pattern identification - new order: batch_score -> shap_analysis -> identify_patterns
    document.getElementById('btn-batch-score').addEventListener('click', batchScore);
    document.getElementById('btn-shap-analysis').addEventListener('click', runSHAPAnalysis);
    document.getElementById('btn-identify').addEventListener('click', identifyPatterns);
    document.getElementById('btn-population-viewer').addEventListener('click', openPopulationViewer);
    document.getElementById('btn-refresh-engineers').addEventListener('click', refreshEngineers);
    document.getElementById('btn-score').addEventListener('click', scoreEngineer);
    document.getElementById('btn-report').addEventListener('click', generateReport);
    document.getElementById('btn-view-report').addEventListener('click', viewReport);
    document.getElementById('btn-refresh-logs').addEventListener('click', refreshLogs);
    document.getElementById('btn-clear-logs').addEventListener('click', clearLogs);
    document.getElementById('btn-download-report').addEventListener('click', downloadReport);
    document.getElementById('engineer-select').addEventListener('change', onEngineerSelected);

    // MongoDB and engineer management listeners
    document.getElementById('btn-fetch-mongodb').addEventListener('click', fetchMongoDB);
    document.getElementById('btn-refresh-engineer-summary').addEventListener('click', refreshEngineerSummary);

    // NDJSON loader listener
    document.getElementById('btn-fetch-ndjson').addEventListener('click', fetchNDJSON);

    // Train from existing VAE listeners
    document.getElementById('btn-train-from-existing').addEventListener('click', showExistingModelInput);
    document.getElementById('btn-start-from-existing').addEventListener('click', trainFromExistingVAE);
    document.getElementById('btn-cancel-from-existing').addEventListener('click', hideExistingModelInput);

    // Stop training button
    document.getElementById('btn-stop-training').addEventListener('click', stopTraining);

    // Tab switching
    document.querySelectorAll('.loader-tab').forEach(tab => {
        tab.addEventListener('click', (e) => switchLoaderTab(e.target.dataset.tab));
    });

    // Synthetic profile listeners
    document.getElementById('btn-generate-synthetic').addEventListener('click', generateSyntheticProfiles);
    document.getElementById('btn-refresh-synthetic-summary').addEventListener('click', refreshSyntheticSummary);
    document.getElementById('btn-add-synthetic').addEventListener('click', addSyntheticToActivities);
    document.getElementById('btn-set-train').addEventListener('click', () => setEngineerSplit('train'));
    document.getElementById('btn-set-validation').addEventListener('click', () => setEngineerSplit('validation'));
    document.getElementById('btn-remove-selected').addEventListener('click', removeSelectedEngineers);
    document.getElementById('btn-merge-engineers').addEventListener('click', mergeSelectedEngineers);
    document.getElementById('select-all-engineers').addEventListener('change', toggleSelectAllEngineers);

    // Sortable table headers
    document.querySelectorAll('#engineer-summary-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sortEngineerTable(th.dataset.sort));
    });
}

function startStatusPolling() {
    updateStatus();
    statusCheckInterval = setInterval(updateStatus, 5000); // Every 5 seconds
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        const status = data.status;
        const canRun = data.can_run;

        // Enable/disable buttons based on file existence
        document.getElementById('btn-preprocess').disabled = !canRun.preprocess;
        document.getElementById('btn-normalization').disabled = !canRun.train_vae;  // Enabled when preprocessed data exists
        document.getElementById('btn-train-vae').disabled = !canRun.train_vae;
        document.getElementById('btn-train-from-existing').disabled = !canRun.train_vae;

        // Pattern identification buttons - NEW dependency chain
        document.getElementById('btn-batch-score').disabled = !canRun.batch_score;
        document.getElementById('btn-shap-analysis').disabled = !canRun.shap_analysis;
        document.getElementById('btn-identify').disabled = !canRun.identify_patterns;
        // Population viewer works with or without pattern names (uses placeholders if not named)
        document.getElementById('btn-population-viewer').disabled = !canRun.population_viewer;

        // Evaluation buttons
        document.getElementById('btn-refresh-engineers').disabled = !status.data_collected;
        document.getElementById('engineer-select').disabled = !status.data_collected;
        document.getElementById('btn-score').disabled = !canRun.score_engineer || !currentEngineerId;

        // Auto-refresh engineers list if data was just collected
        if (status.data_collected && document.getElementById('engineer-select').options.length === 1) {
            refreshEngineers();
        }

    } catch (error) {
        console.error('Failed to update status:', error);
    }
}

async function batchScore() {
    const btn = document.getElementById('btn-batch-score');
    const statusEl = document.getElementById('batch-status');

    btn.disabled = true;
    showStatus(statusEl, 'Starting batch scoring and message assignment...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/batch_score`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Batch scoring failed');
        }

        showStatus(statusEl, 'Batch scoring started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'batch_scoring') {
                clearInterval(pollInterval);
                if (status.status.batch_scored) {
                    showStatus(statusEl, 'Batch scoring and message assignment complete!', 'success');
                } else if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function refreshEngineers() {
    const select = document.getElementById('engineer-select');

    try {
        const response = await fetch(`${API_BASE}/api/engineers`);
        if (!response.ok) throw new Error('Failed to fetch engineers');

        const data = await response.json();

        // Clear and repopulate
        select.innerHTML = '<option value="">Select an engineer...</option>';
        data.engineers.forEach(engineer => {
            const option = document.createElement('option');
            option.value = engineer.engineer_id;
            option.textContent = engineer.engineer_id;
            select.appendChild(option);
        });

    } catch (error) {
        console.error('Failed to refresh engineers:', error);
    }
}

async function onEngineerSelected(event) {
    currentEngineerId = event.target.value;

    // Clear previous results
    document.getElementById('report-container').style.display = 'none';
    document.getElementById('scores-container').style.display = 'none';
    document.getElementById('eval-status').innerHTML = '';

    // Check if score file exists for this engineer
    await updateButtonStates();
}

async function updateButtonStates() {
    // Update general status
    await updateStatus();

    // If an engineer is selected, check if their score file exists
    if (currentEngineerId) {
        try {
            const response = await fetch(`${API_BASE}/api/check_score_exists?engineer_id=${currentEngineerId}`);
            const data = await response.json();

            // Generate Report button ONLY depends on individual score file existence
            const reportBtn = document.getElementById('btn-report');
            reportBtn.disabled = !data.exists;

            // View Report button depends on report file existence
            const viewReportBtn = document.getElementById('btn-view-report');
            viewReportBtn.disabled = !data.report_exists;

            // Download Report button depends on report file existence
            const downloadReportBtn = document.getElementById('btn-download-report');
            downloadReportBtn.disabled = !data.report_exists;

            // Score button needs batch scores to be available (for percentile calculations)
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());
            const scoreBtn = document.getElementById('btn-score');
            scoreBtn.disabled = !status.can_run.score_engineer;

        } catch (error) {
            console.error('Failed to check score existence:', error);
        }
    }
}

async function scoreEngineer() {
    if (!currentEngineerId) return;

    const btn = document.getElementById('btn-score');
    const statusEl = document.getElementById('eval-status');

    btn.disabled = true;
    showStatus(statusEl, 'Scoring engineer...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/score_engineer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engineer_id: currentEngineerId })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Scoring failed');
        }

        const scores = await response.json();

        // Display scores
        document.getElementById('scores-content').textContent = JSON.stringify(scores, null, 2);
        document.getElementById('scores-container').style.display = 'block';

        showStatus(statusEl, 'Scoring complete!', 'success');
        btn.disabled = false;

        // Update button states - Generate Report should now be enabled
        await updateButtonStates();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function generateReport() {
    if (!currentEngineerId) return;

    const btn = document.getElementById('btn-report');
    const statusEl = document.getElementById('eval-status');

    btn.disabled = true;
    showStatus(statusEl, 'Generating report...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/generate_report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engineer_id: currentEngineerId })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Report generation failed');
        }

        const data = await response.json();

        // Display report
        document.getElementById('report-content').textContent = data.report;
        document.getElementById('report-container').style.display = 'block';

        showStatus(statusEl, 'Report generated!', 'success');

        // Update button states to enable View Report and Download Report buttons
        await updateButtonStates();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

function viewReport() {
    // Get current engineer
    const engineerSelect = document.getElementById('engineer-select');
    const engineerId = engineerSelect.value;

    if (!engineerId) {
        alert('Please select an engineer first');
        return;
    }

    // Open report viewer in new tab (uses route, not static file)
    const viewerUrl = `/report_viewer?engineer_id=${encodeURIComponent(engineerId)}`;
    window.open(viewerUrl, '_blank');
}

function openPopulationViewer() {
    // Open population viewer in new tab (uses route, not static file)
    window.open('/population_viewer', '_blank');
}

async function refreshLogs() {
    try {
        const response = await fetch(`${API_BASE}/api/logs`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        const logsContainer = document.getElementById('logs-container');
        logsContainer.innerHTML = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            lines.forEach(line => {
                if (line.startsWith('data:')) {
                    const logLine = line.substring(5).trim();
                    if (logLine && logLine !== 'No logs yet') {
                        appendLog(logLine);
                    }
                }
            });
        }

        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;

    } catch (error) {
        console.error('Failed to fetch logs:', error);
    }
}

function appendLog(text) {
    const logsContainer = document.getElementById('logs-container');
    const logLine = document.createElement('div');
    logLine.className = 'log-line';

    // Add color based on log level
    if (text.includes('ERROR')) {
        logLine.classList.add('log-error');
    } else if (text.includes('WARNING')) {
        logLine.classList.add('log-warning');
    } else if (text.includes('INFO')) {
        logLine.classList.add('log-info');
    }

    logLine.textContent = text;
    logsContainer.appendChild(logLine);
}

function clearLogs() {
    document.getElementById('logs-container').innerHTML = '';
}

function downloadReport() {
    if (!currentEngineerId) return;

    const filename = `report_${currentEngineerId}.txt`;
    window.location.href = `${API_BASE}/api/download/reporting/${filename}`;
}

function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status-message ${type}`;
    element.style.display = 'block';
}

async function preprocessData() {
    const btn = document.getElementById('btn-preprocess');
    const statusEl = document.getElementById('preprocess-status');

    btn.disabled = true;
    showStatus(statusEl, 'Starting preprocessing...', 'info');

    try {
        // Get optional max activities override
        const maxActivitiesInput = document.getElementById('max-activities-per-engineer');
        const maxActivities = maxActivitiesInput.value ? parseInt(maxActivitiesInput.value) : null;

        // Build URL with query param if value provided
        let url = `${API_BASE}/api/preprocess`;
        if (maxActivities !== null) {
            url += `?max_activities_per_engineer=${maxActivities}`;
        }

        const response = await fetch(url, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Preprocessing failed');
        }

        showStatus(statusEl, 'Preprocessing started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'preprocessing') {
                clearInterval(pollInterval);
                showStatus(statusEl, 'Preprocessing complete!', 'success');
                btn.disabled = false;
            } else if (status.status.last_error) {
                clearInterval(pollInterval);
                showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function applyNormalization() {
    const btn = document.getElementById('btn-normalization');
    const statusEl = document.getElementById('preprocess-status');

    btn.disabled = true;
    showStatus(statusEl, 'Starting normalization...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/normalization`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Normalization failed');
        }

        showStatus(statusEl, 'Normalization started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'normalization') {
                clearInterval(pollInterval);
                if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'Normalization complete! Original files backed up with -base suffix.', 'success');
                }
                btn.disabled = false;
            }
        }, 2000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function trainVAE() {
    const btn = document.getElementById('btn-train-vae');
    const stopBtn = document.getElementById('btn-stop-training');
    const statusEl = document.getElementById('train-status');

    btn.disabled = true;
    stopBtn.style.display = 'inline-block';
    showStatus(statusEl, 'Starting VAE training (this may take 20+ minutes)...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/train_vae`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'VAE training failed');
        }

        showStatus(statusEl, 'VAE training started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'training_vae') {
                clearInterval(pollInterval);
                stopBtn.style.display = 'none';
                if (status.status.models_trained) {
                    showStatus(statusEl, 'VAE training complete!', 'success');
                } else if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'VAE training stopped.', 'info');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
        stopBtn.style.display = 'none';
    }
}

function showExistingModelInput() {
    document.getElementById('existing-model-input').style.display = 'block';
    document.getElementById('btn-train-from-existing').disabled = true;
}

function hideExistingModelInput() {
    document.getElementById('existing-model-input').style.display = 'none';
    document.getElementById('btn-train-from-existing').disabled = false;
    document.getElementById('existing-model-path').value = '';
}

async function trainFromExistingVAE() {
    const modelPath = document.getElementById('existing-model-path').value.trim();
    if (!modelPath) {
        alert('Please enter the path to an existing VAE checkpoint');
        return;
    }

    const btn = document.getElementById('btn-start-from-existing');
    const stopBtn = document.getElementById('btn-stop-training');
    const statusEl = document.getElementById('train-status');

    btn.disabled = true;
    stopBtn.style.display = 'inline-block';
    hideExistingModelInput();
    showStatus(statusEl, `Starting VAE training from existing model: ${modelPath}...`, 'info');

    try {
        const response = await fetch(`${API_BASE}/api/train_vae_from_existing`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'VAE training from existing model failed');
        }

        showStatus(statusEl, 'VAE training from existing model started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'training_vae_from_existing') {
                clearInterval(pollInterval);
                stopBtn.style.display = 'none';
                if (status.status.models_trained) {
                    showStatus(statusEl, 'VAE training from existing model complete!', 'success');
                } else if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'VAE training stopped.', 'info');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
        stopBtn.style.display = 'none';
    }
}

async function stopTraining() {
    const stopBtn = document.getElementById('btn-stop-training');
    const statusEl = document.getElementById('train-status');

    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping...';

    try {
        const response = await fetch(`${API_BASE}/api/train/stop`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to stop training');
        }

        showStatus(statusEl, 'Stop requested. Training will stop after current epoch...', 'info');

    } catch (error) {
        showStatus(statusEl, `Error stopping: ${error.message}`, 'error');
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Training';
    }
}

async function runSHAPAnalysis() {
    const btn = document.getElementById('btn-shap-analysis');
    const statusEl = document.getElementById('shap-status');

    btn.disabled = true;
    showStatus(statusEl, 'Starting SHAP analysis...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/shap_analysis`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'SHAP analysis failed');
        }

        showStatus(statusEl, 'SHAP analysis started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'shap_analysis') {
                clearInterval(pollInterval);
                if (status.status.patterns_interpreted) {
                    showStatus(statusEl, 'SHAP analysis complete!', 'success');
                } else if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function identifyPatterns() {
    const btn = document.getElementById('btn-identify');
    const statusEl = document.getElementById('identify-status');

    btn.disabled = true;
    showStatus(statusEl, 'Starting pattern naming with LLM...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/identify_patterns`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Pattern naming failed');
        }

        showStatus(statusEl, 'Pattern naming started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            await updateStatus();
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'pattern_naming') {
                clearInterval(pollInterval);
                if (status.status.patterns_identified) {
                    showStatus(statusEl, 'Pattern naming complete!', 'success');
                } else if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

// MongoDB and Engineer Management Functions

async function fetchMongoDB() {
    const btn = document.getElementById('btn-fetch-mongodb');
    const statusEl = document.getElementById('mongodb-status');

    const database = document.getElementById('mongodb-database').value.trim() || null;
    const maxEngineers = document.getElementById('mongodb-max-engineers').value || null;
    const maxActivities = document.getElementById('mongodb-max-activities').value || null;
    const append = document.getElementById('mongodb-append').checked;

    btn.disabled = true;
    showStatus(statusEl, `Fetching data from MongoDB (${database})...`, 'info');

    try {
        const requestBody = {
            database: database,
            append: append,
            max_engineers: maxEngineers ? parseInt(maxEngineers) : null,
            max_activities_per_engineer: maxActivities ? parseInt(maxActivities) : null
        };

        const response = await fetch(`${API_BASE}/api/fetch_mongodb`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'MongoDB fetch failed');
        }

        showStatus(statusEl, 'MongoDB fetch started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'fetching_mongodb') {
                clearInterval(pollInterval);
                if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'MongoDB fetch complete! Refresh summary to see engineers.', 'success');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

function switchLoaderTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.loader-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
        content.classList.remove('active');
    });

    const activeContent = document.getElementById(`tab-${tabName}`);
    if (activeContent) {
        activeContent.style.display = 'block';
        activeContent.classList.add('active');
    }
}

async function fetchNDJSON() {
    const btn = document.getElementById('btn-fetch-ndjson');
    const statusEl = document.getElementById('ndjson-status');

    const basePath = document.getElementById('ndjson-path').value.trim();
    const dateStart = document.getElementById('ndjson-date-start').value;
    const dateEnd = document.getElementById('ndjson-date-end').value;

    if (!basePath) {
        showStatus(statusEl, 'Please enter a path to the data folder', 'error');
        return;
    }

    btn.disabled = true;
    showStatus(statusEl, `Loading data from ${basePath}...`, 'info');

    try {
        const requestBody = {
            base_path: basePath
        };

        // Only include date_range if both dates are provided
        if (dateStart && dateEnd) {
            requestBody.date_range = {
                start: dateStart,
                end: dateEnd
            };
        }

        const response = await fetch(`${API_BASE}/api/fetch_ndjson`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'NDJSON fetch failed');
        }

        showStatus(statusEl, 'NDJSON loading started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'fetching_ndjson') {
                clearInterval(pollInterval);
                if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'NDJSON loading complete! Refresh summary to see engineers.', 'success');
                }
                btn.disabled = false;
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function refreshEngineerSummary() {
    const statusEl = document.getElementById('engineer-summary-status');
    const container = document.getElementById('engineer-summary-container');

    showStatus(statusEl, 'Loading engineer summary...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/engineers`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch engineer summary');
        }

        const data = await response.json();

        // Update counts
        document.getElementById('train-count').textContent = data.train_count;
        document.getElementById('validation-count').textContent = data.validation_count;
        document.getElementById('bot-count').textContent = data.bot_count;
        document.getElementById('total-engineers').textContent = data.total_engineers;

        // Store data for sorting
        engineerSummaryData = data.engineers;

        // Sort and render
        sortEngineerTable(currentSortColumn, false);

        container.style.display = 'block';
        statusEl.style.display = 'none';

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    }
}

function sortEngineerTable(column, toggleDirection = true) {
    // Toggle direction if clicking the same column
    if (toggleDirection) {
        if (column === currentSortColumn) {
            currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            currentSortColumn = column;
            // Default direction based on column type
            currentSortDirection = (column === 'activity_count' || column === 'is_bot' || column === 'is_internal') ? 'desc' : 'asc';
        }
    }

    // Sort the data
    engineerSummaryData.sort((a, b) => {
        let aVal, bVal;

        switch (column) {
            case 'engineer_id':
                aVal = a.engineer_id.toLowerCase();
                bVal = b.engineer_id.toLowerCase();
                break;
            case 'activity_count':
                aVal = a.activity_count;
                bVal = b.activity_count;
                break;
            case 'split':
                aVal = a.split;
                bVal = b.split;
                break;
            case 'sources':
                aVal = a.sources.join(', ').toLowerCase();
                bVal = b.sources.join(', ').toLowerCase();
                break;
            case 'is_bot':
                aVal = a.is_bot ? 1 : 0;
                bVal = b.is_bot ? 1 : 0;
                break;
            case 'is_internal':
                aVal = a.is_internal ? 1 : 0;
                bVal = b.is_internal ? 1 : 0;
                break;
            case 'projects':
                aVal = (a.projects || '').toLowerCase();
                bVal = (b.projects || '').toLowerCase();
                break;
            default:
                return 0;
        }

        if (aVal < bVal) return currentSortDirection === 'asc' ? -1 : 1;
        if (aVal > bVal) return currentSortDirection === 'asc' ? 1 : -1;
        return 0;
    });

    renderEngineerTable();
    updateSortIndicators();
}

function renderEngineerTable() {
    const tbody = document.getElementById('engineer-summary-body');
    tbody.innerHTML = '';

    engineerSummaryData.forEach(eng => {
        const row = document.createElement('tr');
        const botIcon = eng.is_bot ? '<span title="Bot" style="font-size: 1.2em;">&#129302;</span>' : '<span title="Human" style="color: #888;">&#128100;</span>';
        const internalIcon = eng.is_internal === undefined ? '<span style="color: #888;">-</span>' :
            (eng.is_internal ? '<span title="Internal" style="color: #28a745;">&#10004;</span>' : '<span title="External" style="color: #dc3545;">&#10006;</span>');
        const projectBadges = (eng.projects || '').split(',')
            .filter(p => p)
            .map(p => `<span style="display: inline-block; padding: 2px 6px; margin: 1px; background: #e0e0e0; border-radius: 4px; font-size: 0.8em;">${p}</span>`)
            .join('');

        row.innerHTML = `
            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                <input type="checkbox" class="engineer-checkbox" value="${eng.engineer_id}">
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">${eng.engineer_id}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">${eng.activity_count}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                <span style="color: ${eng.split === 'train' ? '#28a745' : '#007bff'};">${eng.split}</span>
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">${eng.sources.join(', ')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">${botIcon}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">${internalIcon}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">${projectBadges}</td>
        `;
        tbody.appendChild(row);
    });

    // Add change listeners to checkboxes
    document.querySelectorAll('.engineer-checkbox').forEach(cb => {
        cb.addEventListener('change', updateActionButtons);
    });
}

function updateSortIndicators() {
    // Clear all indicators
    ['engineer_id', 'activity_count', 'split', 'sources', 'is_bot', 'is_internal', 'projects'].forEach(col => {
        const indicator = document.getElementById(`sort-indicator-${col}`);
        if (indicator) {
            indicator.textContent = '';
        }
    });

    // Set current indicator
    const currentIndicator = document.getElementById(`sort-indicator-${currentSortColumn}`);
    if (currentIndicator) {
        currentIndicator.textContent = currentSortDirection === 'asc' ? '▲' : '▼';
    }
}

function toggleSelectAllEngineers(event) {
    const checked = event.target.checked;
    document.querySelectorAll('.engineer-checkbox').forEach(cb => {
        cb.checked = checked;
    });
    updateActionButtons();
}

function updateActionButtons() {
    const selectedCount = document.querySelectorAll('.engineer-checkbox:checked').length;
    const hasSelection = selectedCount > 0;

    document.getElementById('btn-set-train').disabled = !hasSelection;
    document.getElementById('btn-set-validation').disabled = !hasSelection;
    document.getElementById('btn-remove-selected').disabled = !hasSelection;
    document.getElementById('btn-merge-engineers').disabled = !hasSelection;
}

function getSelectedEngineerIds() {
    const checkboxes = document.querySelectorAll('.engineer-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

async function setEngineerSplit(split) {
    const engineerIds = getSelectedEngineerIds();
    if (engineerIds.length === 0) return;

    const statusEl = document.getElementById('action-status');

    try {
        const response = await fetch(`${API_BASE}/api/engineers/set_split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engineer_ids: engineerIds, split: split })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `Failed to set split to ${split}`);
        }

        const data = await response.json();
        showStatus(statusEl, `${data.message} (${data.rows_updated} rows)`, 'success');

        // Refresh the summary
        await refreshEngineerSummary();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    }
}

async function removeSelectedEngineers() {
    const engineerIds = getSelectedEngineerIds();
    if (engineerIds.length === 0) return;

    const confirmed = confirm(`Are you sure you want to remove ${engineerIds.length} engineer(s)?\n\nThis will delete all their activities from the CSV.`);
    if (!confirmed) return;

    const statusEl = document.getElementById('action-status');

    try {
        const response = await fetch(`${API_BASE}/api/engineers/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engineer_ids: engineerIds })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to remove engineers');
        }

        const data = await response.json();
        showStatus(statusEl, `${data.message} (${data.rows_deleted} rows)`, 'success');

        // Refresh the summary
        await refreshEngineerSummary();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    }
}

async function mergeSelectedEngineers() {
    const sourceIds = getSelectedEngineerIds();
    if (sourceIds.length === 0) return;

    const targetId = document.getElementById('merge-target-id').value.trim();
    if (!targetId) {
        alert('Please enter a target ID to merge into');
        return;
    }

    const confirmed = confirm(`Merge ${sourceIds.length} engineer(s) into "${targetId}"?\n\nSelected IDs: ${sourceIds.join(', ')}`);
    if (!confirmed) return;

    const statusEl = document.getElementById('action-status');

    try {
        const response = await fetch(`${API_BASE}/api/engineers/merge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_ids: sourceIds, target_id: targetId })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to merge engineers');
        }

        const data = await response.json();
        showStatus(statusEl, `${data.message} (${data.rows_updated} rows)`, 'success');

        // Clear the target input
        document.getElementById('merge-target-id').value = '';

        // Refresh the summary
        await refreshEngineerSummary();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    }
}

// Synthetic Profile Functions

async function generateSyntheticProfiles() {
    const btn = document.getElementById('btn-generate-synthetic');
    const statusEl = document.getElementById('synthetic-status');

    const copiesPerProfile = parseInt(document.getElementById('synthetic-copies').value) || 3;

    btn.disabled = true;
    showStatus(statusEl, 'Starting synthetic profile generation...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/synthetic/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ copies_per_profile: copiesPerProfile })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Synthetic generation failed');
        }

        showStatus(statusEl, 'Synthetic generation started. Check logs for progress.', 'info');

        const pollInterval = setInterval(async () => {
            const status = await fetch(`${API_BASE}/api/status`).then(r => r.json());

            if (status.status.current_task !== 'generating_synthetic') {
                clearInterval(pollInterval);
                if (status.status.last_error) {
                    showStatus(statusEl, `Error: ${status.status.last_error}`, 'error');
                } else {
                    showStatus(statusEl, 'Synthetic generation complete! Refresh summary to see profiles.', 'success');
                    await refreshSyntheticSummary();
                }
                btn.disabled = false;
            } else {
                const progress = status.status.synthetic_generation_progress || 0;
                const total = status.status.synthetic_generation_total || 0;
                if (total > 0) {
                    showStatus(statusEl, `Generating... ${progress}/${total} profiles`, 'info');
                }
            }
        }, 3000);

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
        btn.disabled = false;
    }
}

async function refreshSyntheticSummary() {
    const statusEl = document.getElementById('synthetic-status');
    const container = document.getElementById('synthetic-summary-container');
    const addBtn = document.getElementById('btn-add-synthetic');

    showStatus(statusEl, 'Loading synthetic profile summary...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/synthetic/summary`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch synthetic summary');
        }

        const data = await response.json();

        // Update counts
        document.getElementById('synthetic-template-count').textContent = data.template_count;
        document.getElementById('synthetic-profile-count').textContent = data.total_profiles;

        // Update details
        const detailsEl = document.getElementById('synthetic-details');
        if (data.profile_details.length > 0) {
            detailsEl.innerHTML = data.profile_details.map(d =>
                `<div><strong>${d.template_name}:</strong> ${d.profile_count} profiles</div>`
            ).join('');
            addBtn.disabled = false;
        } else {
            detailsEl.innerHTML = '<div style="color: #666;">No synthetic profiles generated yet.</div>';
            addBtn.disabled = true;
        }

        container.style.display = 'block';
        statusEl.style.display = 'none';

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    }
}

async function addSyntheticToActivities() {
    const btn = document.getElementById('btn-add-synthetic');
    const statusEl = document.getElementById('synthetic-status');
    const split = document.getElementById('synthetic-split').value;

    const confirmed = confirm(`Add all synthetic profiles to activities.csv as "${split}" data?\n\nThis will convert JSON profiles to CSV rows and merge them with existing activities.`);
    if (!confirmed) return;

    btn.disabled = true;
    showStatus(statusEl, 'Adding synthetic profiles to activities...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/synthetic/add_to_activities`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ split: split })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to add synthetic profiles');
        }

        const data = await response.json();

        let message = `Added ${data.profiles_added} profiles (${data.activities_added} activities)`;
        if (data.errors && data.errors.length > 0) {
            message += ` with ${data.errors.length} errors`;
        }

        showStatus(statusEl, message, 'success');

        // Refresh engineer summary to show new engineers
        await refreshEngineerSummary();

    } catch (error) {
        showStatus(statusEl, `Error: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}
