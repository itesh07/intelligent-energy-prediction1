/**
 * WATTWISE — Intelligent Energy Consumption Prediction System
 * Frontend Application & Telemetry Engine
 * Coordinates Chart.js visualizations, REST API interactions,
 * real-time inference, and CSV ingestion workflows.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ------------------------------------------------------------------------
  // State & Chart Registry
  // ------------------------------------------------------------------------
  const chartInstances = {};
  let currentUploadedFile = null;

  // Resolves backend API URL (supports local single-origin, Vercel proxy, or decoupled Render URL)
  const API_BASE = (
    document.querySelector('meta[name="api-base-url"]')?.getAttribute('content') ||
    window.WATTWISE_API_URL ||
    localStorage.getItem('WATTWISE_API_URL') ||
    ''
  ).replace(/\/+$/, '');

  // ------------------------------------------------------------------------
  // DOM Element Selectors
  // ------------------------------------------------------------------------
  const systemStatusPill = document.getElementById('systemStatusPill');
  const systemStatusText = document.getElementById('systemStatusText');
  const activeDatasetLabel = document.getElementById('activeDatasetLabel');
  const datasetBadge = document.getElementById('datasetBadge');
  const noticeTitle = document.getElementById('noticeTitle');
  const noticeDesc = document.getElementById('noticeDesc');
  const btnResetDemo = document.getElementById('btnResetDemo');

  // Dashboard Metric Cards
  const metricCurrentKwh = document.getElementById('metricCurrentKwh');
  const metricPredictedKwh = document.getElementById('metricPredictedKwh');
  const metricAvgDailyKwh = document.getElementById('metricAvgDailyKwh');
  const metricCostInr = document.getElementById('metricCostInr');
  const selectedModelBadge = document.getElementById('selectedModelBadge');

  // Prediction Form Elements
  const predictionForm = document.getElementById('predictionForm');
  const inputTemp = document.getElementById('inputTemp');
  const tempDisplay = document.getElementById('tempDisplay');
  const inputHumidity = document.getElementById('inputHumidity');
  const humidityDisplay = document.getElementById('humidityDisplay');
  const inputOccupancy = document.getElementById('inputOccupancy');
  const occupancyDisplay = document.getElementById('occupancyDisplay');
  const inputHour = document.getElementById('inputHour');
  const hourDisplay = document.getElementById('hourDisplay');
  const inputDayOfWeek = document.getElementById('inputDayOfWeek');
  const inputPrevEnergy = document.getElementById('inputPrevEnergy');
  const btnPredict = document.getElementById('btnPredict');

  // Prediction Views
  const predViews = {
    empty: document.getElementById('predEmptyState'),
    loading: document.getElementById('predLoadingState'),
    result: document.getElementById('predResultState'),
    error: document.getElementById('predErrorState'),
  };
  const reportState = document.getElementById('reportState');

  // Prediction Result Display Fields
  const resultLoadClass = document.getElementById('resultLoadClass');
  const resultTimestamp = document.getElementById('resultTimestamp');
  const resultKwhValue = document.getElementById('resultKwhValue');
  const resultDeltaBadge = document.getElementById('resultDeltaBadge');
  const resultCostInr = document.getElementById('resultCostInr');
  const resultModelName = document.getElementById('resultModelName');
  const resultModelRmse = document.getElementById('resultModelRmse');
  const valBaseLoad = document.getElementById('valBaseLoad');
  const valHvacLoad = document.getElementById('valHvacLoad');
  const valPlugLoad = document.getElementById('valPlugLoad');
  const barBaseLoad = document.getElementById('barBaseLoad');
  const barHvacLoad = document.getElementById('barHvacLoad');
  const barPlugLoad = document.getElementById('barPlugLoad');

  // Analytics Chart Tabs
  const chartTabPills = document.querySelectorAll('#chartTabPills .pill-btn');
  const chartCards = {
    hourly: document.getElementById('chartCardHourly'),
    daily: document.getElementById('chartCardDaily'),
    weekly: document.getElementById('chartCardWeekly'),
    actualVsPred: document.getElementById('chartCardActualVsPred'),
    distribution: document.getElementById('chartCardDistribution'),
  };

  // Upload Elements
  const csvFileInput = document.getElementById('csvFileInput');
  const csvDropzone = document.getElementById('csvDropzone');
  const csvSelectedFile = document.getElementById('csvSelectedFile');
  const csvFileName = document.getElementById('csvFileName');
  const csvFileSize = document.getElementById('csvFileSize');
  const csvRemoveFile = document.getElementById('csvRemoveFile');
  const btnUploadCsv = document.getElementById('btnUploadCsv');
  const uploadStateBadge = document.getElementById('uploadStateBadge');
  const uploadViews = {
    empty: document.getElementById('uploadEmptyState'),
    loading: document.getElementById('uploadLoadingState'),
    preview: document.getElementById('uploadPreviewState'),
    error: document.getElementById('uploadErrorState'),
  };
  const btnRetrainModel = document.getElementById('btnRetrainModel');
  const btnRestoreDemoFromPreview = document.getElementById('btnRestoreDemoFromPreview');

  // ------------------------------------------------------------------------
  // Helper & Formatting Utilities
  // ------------------------------------------------------------------------
  function setPredView(viewName) {
    Object.entries(predViews).forEach(([name, el]) => {
      if (el) el.classList.toggle('hidden', name !== viewName);
    });
  }

  function setUploadView(viewName) {
    Object.entries(uploadViews).forEach(([name, el]) => {
      if (el) el.classList.toggle('hidden', name !== viewName);
    });
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function formatHour(h) {
    const hour = parseInt(h, 10);
    const suffix = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 === 0 ? 12 : hour % 12;
    return `${String(hour).padStart(2, '0')}:00 (${displayHour} ${suffix})`;
  }

  // ------------------------------------------------------------------------
  // Sliders & Preset Sync
  // ------------------------------------------------------------------------
  if (inputTemp) {
    inputTemp.addEventListener('input', () => {
      tempDisplay.textContent = `${parseFloat(inputTemp.value).toFixed(1)} °C`;
    });
  }

  if (inputHumidity) {
    inputHumidity.addEventListener('input', () => {
      humidityDisplay.textContent = `${inputHumidity.value} %`;
    });
  }

  if (inputOccupancy) {
    inputOccupancy.addEventListener('input', () => {
      occupancyDisplay.textContent = `${inputOccupancy.value} occupants`;
    });
  }

  if (inputHour) {
    inputHour.addEventListener('input', () => {
      hourDisplay.textContent = formatHour(inputHour.value);
    });
  }

  document.querySelectorAll('.preset-chips .chip-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const field = btn.getAttribute('data-field');
      const val = btn.getAttribute('data-val');
      const siblings = btn.parentElement.querySelectorAll('.chip-btn');
      siblings.forEach((s) => s.classList.remove('active'));
      btn.classList.add('active');

      if (field === 'temp') {
        inputTemp.value = val;
        tempDisplay.textContent = `${parseFloat(val).toFixed(1)} °C`;
      } else if (field === 'occ') {
        inputOccupancy.value = val;
        occupancyDisplay.textContent = `${val} occupants`;
      }
    });
  });

  // ------------------------------------------------------------------------
  // Chart.js Setup & Theming
  // ------------------------------------------------------------------------
  Chart.defaults.color = '#8e9bb1';
  Chart.defaults.font.family = "'DM Sans', sans-serif";
  Chart.defaults.borderColor = 'rgba(190, 203, 234, 0.08)';

  function getGradient(ctx, colorStart, colorEnd) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 350);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
  }

  // 1. Hourly Chart
  function renderHourlyChart(hourlyData) {
    const ctx = document.getElementById('hourlyChart');
    if (!ctx) return;
    if (chartInstances.hourly) chartInstances.hourly.destroy();

    const canvasCtx = ctx.getContext('2d');
    const energyGradient = getGradient(canvasCtx, 'rgba(94, 229, 223, 0.35)', 'rgba(94, 229, 223, 0.01)');

    chartInstances.hourly = new Chart(ctx, {
      type: 'line',
      data: {
        labels: hourlyData.labels,
        datasets: [
          {
            label: 'Energy Consumption (kWh)',
            data: hourlyData.values,
            borderColor: '#5ee5df',
            backgroundColor: energyGradient,
            borderWidth: 2.2,
            pointRadius: 2.5,
            pointBackgroundColor: '#5ee5df',
            pointHoverRadius: 5,
            fill: true,
            tension: 0.35,
            yAxisID: 'yEnergy',
          },
          {
            label: 'Temperature (°C)',
            data: hourlyData.temperatures,
            borderColor: '#f8b449',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: false,
            tension: 0.35,
            yAxisID: 'yTemp',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12, usePointStyle: true, padding: 16 },
          },
          tooltip: {
            backgroundColor: 'rgba(15, 21, 35, 0.95)',
            borderColor: 'rgba(94, 229, 223, 0.3)',
            borderWidth: 1,
            titleColor: '#f3f5fb',
            bodyColor: '#cbd5e5',
            padding: 10,
          },
        },
        scales: {
          yEnergy: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Energy (kWh)', color: '#5ee5df' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
          },
          yTemp: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Ambient Temp (°C)', color: '#f8b449' },
            grid: { drawOnChartArea: false },
          },
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 12, font: { size: 9 } },
          },
        },
      },
    });
  }

  // 2. Daily Chart
  function renderDailyChart(dailyData) {
    const ctx = document.getElementById('dailyChart');
    if (!ctx) return;
    if (chartInstances.daily) chartInstances.daily.destroy();

    const canvasCtx = ctx.getContext('2d');
    const barGradient = getGradient(canvasCtx, '#5ee5df', '#378d8a');

    chartInstances.daily = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: dailyData.labels,
        datasets: [
          {
            label: 'Daily Total Consumption (kWh)',
            data: dailyData.values,
            backgroundColor: barGradient,
            borderRadius: 6,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 21, 35, 0.95)',
            borderColor: 'rgba(94, 229, 223, 0.3)',
            borderWidth: 1,
          },
        },
        scales: {
          y: {
            title: { display: true, text: 'Total Daily kWh' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // 3. Weekly Chart
  function renderWeeklyChart(weeklyData) {
    const ctx = document.getElementById('weeklyChart');
    if (!ctx) return;
    if (chartInstances.weekly) chartInstances.weekly.destroy();

    const canvasCtx = ctx.getContext('2d');
    const barGradient = getGradient(canvasCtx, '#9e8bff', '#5b4bc7');

    chartInstances.weekly = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: weeklyData.labels,
        datasets: [
          {
            label: 'Mean Consumption (kWh)',
            data: weeklyData.values,
            backgroundColor: barGradient,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 21, 35, 0.95)',
            borderColor: 'rgba(158, 139, 255, 0.3)',
            borderWidth: 1,
          },
        },
        scales: {
          y: {
            title: { display: true, text: 'Average Hourly kWh' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // 4. Actual vs Predicted Chart (Chronological Test Partition)
  function renderActualVsPredChart(data) {
    const ctx = document.getElementById('actualVsPredChart');
    if (!ctx) return;
    if (chartInstances.actualVsPred) chartInstances.actualVsPred.destroy();

    chartInstances.actualVsPred = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.timestamps,
        datasets: [
          {
            label: 'Actual Measured (kWh)',
            data: data.actual,
            borderColor: '#7a88a3',
            borderWidth: 1.8,
            pointRadius: 1,
            fill: false,
            tension: 0.25,
          },
          {
            label: 'ML Model Forecast (kWh)',
            data: data.predicted,
            borderColor: '#5ee5df',
            borderWidth: 2.2,
            pointRadius: 2,
            pointBackgroundColor: '#5ee5df',
            fill: false,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { usePointStyle: true } },
          tooltip: {
            backgroundColor: 'rgba(15, 21, 35, 0.95)',
            borderColor: 'rgba(94, 229, 223, 0.3)',
            borderWidth: 1,
          },
        },
        scales: {
          y: {
            title: { display: true, text: 'Energy (kWh)' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
          },
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 10, font: { size: 9 } },
          },
        },
      },
    });
  }

  // 5. Distribution Chart
  function renderDistributionChart(distData) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;
    if (chartInstances.distribution) chartInstances.distribution.destroy();

    chartInstances.distribution = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: distData.labels,
        datasets: [
          {
            label: 'Frequency (Hours)',
            data: distData.counts,
            backgroundColor: 'rgba(94, 229, 223, 0.45)',
            borderColor: '#5ee5df',
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            title: { display: true, text: 'Observations' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
          },
          x: {
            title: { display: true, text: 'Energy Range (kWh)' },
            grid: { display: false },
          },
        },
      },
    });
  }

  // 6. Model Comparison Chart (MAE vs RMSE)
  function renderModelComparisonChart(allMetrics) {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) return;
    if (chartInstances.comparison) chartInstances.comparison.destroy();

    const modelNames = Object.keys(allMetrics);
    const maes = modelNames.map((m) => allMetrics[m].mae);
    const rmses = modelNames.map((m) => allMetrics[m].rmse);

    chartInstances.comparison = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: modelNames.map((n) => n.replace(' Regressor', '')),
        datasets: [
          {
            label: 'MAE (kWh)',
            data: maes,
            backgroundColor: 'rgba(94, 229, 223, 0.6)',
            borderColor: '#5ee5df',
            borderWidth: 1,
            borderRadius: 5,
          },
          {
            label: 'RMSE (kWh)',
            data: rmses,
            backgroundColor: 'rgba(158, 139, 255, 0.6)',
            borderColor: '#9e8bff',
            borderWidth: 1,
            borderRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10, usePointStyle: true } },
          tooltip: {
            backgroundColor: 'rgba(15, 21, 35, 0.95)',
            borderColor: 'rgba(94, 229, 223, 0.3)',
            borderWidth: 1,
          },
        },
        scales: {
          y: {
            title: { display: true, text: 'Error in kWh (Lower is better)' },
            grid: { color: 'rgba(190, 203, 234, 0.06)' },
            beginAtZero: true,
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // ------------------------------------------------------------------------
  // Chart Tabs Navigation
  // ------------------------------------------------------------------------
  chartTabPills.forEach((btn) => {
    btn.addEventListener('click', () => {
      chartTabPills.forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');

      const targetView = btn.getAttribute('data-view');
      Object.entries(chartCards).forEach(([name, card]) => {
        if (card) card.classList.toggle('active', name === targetView);
      });
    });
  });

  // ------------------------------------------------------------------------
  // Data Fetching: Analytics Telemetry & Model Benchmarks
  // ------------------------------------------------------------------------
  async function loadAnalytics() {
    try {
      const res = await fetch(`${API_BASE}/api/analytics`);
      if (!res.ok) throw new Error('Analytics API failed');
      const data = await res.json();

      // Update 4 Metric Cards
      const m = data.metrics;
      if (metricCurrentKwh) metricCurrentKwh.textContent = m.current_energy_kwh.toFixed(1);
      if (metricPredictedKwh) metricPredictedKwh.textContent = m.predicted_next_kwh.toFixed(1);
      if (metricAvgDailyKwh) metricAvgDailyKwh.textContent = m.average_daily_kwh.toFixed(1);
      if (metricCostInr) metricCostInr.textContent = m.estimated_daily_cost_inr.toLocaleString();

      // Prepopulate Previous Energy input with current kWh
      if (inputPrevEnergy) inputPrevEnergy.value = m.current_energy_kwh.toFixed(1);

      // Active dataset label
      if (activeDatasetLabel) {
        const isSynthetic = m.is_synthetic;
        activeDatasetLabel.textContent = isSynthetic
          ? `Active Dataset: ${m.dataset_name} (Synthetic Simulation Demo)`
          : `Active Dataset: ${m.dataset_name} (${m.total_records.toLocaleString()} records)`;
      }

      if (noticeTitle && noticeDesc) {
        if (m.is_synthetic) {
          noticeTitle.textContent = 'Synthetic Demo Dataset Active';
          noticeDesc.textContent =
            'Currently utilizing the 90-day smart building physics simulation benchmark. You can upload custom building sensor CSV telemetry anytime.';
        } else {
          noticeTitle.textContent = `Custom Dataset Active: ${m.dataset_name}`;
          noticeDesc.textContent = `Loaded ${m.total_records.toLocaleString()} valid chronological records for machine learning training and analytics.`;
        }
      }

      // Render Charts
      if (data.hourly) renderHourlyChart(data.hourly);
      if (data.daily) renderDailyChart(data.daily);
      if (data.weekly) renderWeeklyChart(data.weekly);
      if (data.actual_vs_predicted) renderActualVsPredChart(data.actual_vs_predicted);
      if (data.distribution) renderDistributionChart(data.distribution);
    } catch (err) {
      console.error('Error fetching analytics:', err);
    }
  }

  async function loadModelMetrics() {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (!res.ok) return;
      const data = await res.json();

      if (selectedModelBadge) {
        selectedModelBadge.textContent = data.best_model_name;
      }

      // Populate Model Comparison Table
      const tbody = document.getElementById('modelComparisonBody');
      if (tbody && data.all_metrics) {
        tbody.innerHTML = '';
        Object.entries(data.all_metrics).forEach(([name, metrics]) => {
          const isBest = name === data.best_model_name;
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="model-name-cell">${name}</td>
            <td class="metric-num">${metrics.mae.toFixed(3)}</td>
            <td class="metric-num">${metrics.rmse.toFixed(3)}</td>
            <td class="metric-num">${metrics.r2_score.toFixed(4)}</td>
            <td class="metric-num">${metrics.training_time_seconds.toFixed(2)}s</td>
            <td>
              <span class="status-badge ${isBest ? 'best' : ''}">
                ${isBest ? '★ SELECTED BEST' : 'BENCHMARKED'}
              </span>
            </td>
          `;
          tbody.appendChild(tr);
        });
      }

      // Render Comparison Chart
      if (data.all_metrics) {
        renderModelComparisonChart(data.all_metrics);
      }
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        if (systemStatusText) {
          systemStatusText.textContent = data.model_ready ? 'AI MODEL ONLINE' : 'TRAINING REQUIRED';
        }
      }
    } catch (e) {
      if (systemStatusText) systemStatusText.textContent = 'OFFLINE';
    }
  }

  // ------------------------------------------------------------------------
  // Prediction Form Submission Handler
  // ------------------------------------------------------------------------
  if (predictionForm) {
    predictionForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const payload = {
        temperature: parseFloat(inputTemp.value),
        humidity: parseFloat(inputHumidity.value),
        occupancy: parseInt(inputOccupancy.value, 10),
        hour: parseInt(inputHour.value, 10),
        day_of_week: parseInt(inputDayOfWeek.value, 10),
        prev_energy_kwh: parseFloat(inputPrevEnergy.value),
      };

      if (isNaN(payload.prev_energy_kwh) || payload.prev_energy_kwh < 0) {
        alert('Please enter a valid previous consumption value.');
        return;
      }

      setPredView('loading');
      if (reportState) {
        reportState.innerHTML = '<i></i> COMPUTING';
        reportState.classList.add('active');
      }
      btnPredict.disabled = true;

      try {
        const response = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Prediction request failed.');
        }

        // Render Prediction Result
        resultKwhValue.textContent = data.predicted_kwh.toFixed(1);
        resultTimestamp.textContent = data.timestamp;

        // Load class & badge
        resultLoadClass.textContent = data.load_category.toUpperCase();
        resultLoadClass.className = `result-label ${data.load_class}`;

        // Delta
        const diffSign = data.difference_kwh >= 0 ? '+' : '';
        resultDeltaBadge.textContent = `${diffSign}${data.difference_kwh} kWh (${diffSign}${data.percentage_change}%) vs previous hour`;
        resultDeltaBadge.style.color = data.difference_kwh > 0 ? 'var(--cyan)' : 'var(--violet)';

        // Costs & Metrics
        resultCostInr.textContent = `₹${data.estimated_cost_inr.toLocaleString()}`;
        resultModelName.textContent = data.selected_model;
        resultModelRmse.textContent = `${data.model_metrics.rmse.toFixed(3)} kWh (MAE: ${data.model_metrics.mae.toFixed(3)})`;

        // Decomposition bars
        const totalEstimated =
          data.breakdown.base_building_load +
          data.breakdown.hvac_thermal_load +
          data.breakdown.occupancy_plug_load;

        valBaseLoad.textContent = `${data.breakdown.base_building_load} kWh`;
        valHvacLoad.textContent = `${data.breakdown.hvac_thermal_load} kWh`;
        valPlugLoad.textContent = `${data.breakdown.occupancy_plug_load} kWh`;

        const basePct = Math.round((data.breakdown.base_building_load / totalEstimated) * 100);
        const hvacPct = Math.round((data.breakdown.hvac_thermal_load / totalEstimated) * 100);
        const plugPct = Math.max(5, 100 - basePct - hvacPct);

        barBaseLoad.style.width = `${basePct}%`;
        barHvacLoad.style.width = `${hvacPct}%`;
        barPlugLoad.style.width = `${plugPct}%`;

        setPredView('result');
        if (reportState) {
          reportState.innerHTML = '<i></i> INFERENCE COMPLETE';
          reportState.classList.add('active');
        }
      } catch (err) {
        document.getElementById('predErrorMessage').textContent = err.message;
        setPredView('error');
        if (reportState) {
          reportState.innerHTML = '<i></i> ERROR';
          reportState.classList.remove('active');
        }
      } finally {
        btnPredict.disabled = false;
      }
    });
  }

  // ------------------------------------------------------------------------
  // CSV Dataset Upload & Drag-and-Drop
  // ------------------------------------------------------------------------
  function chooseCsv(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Only CSV (.csv) files are supported.');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      alert('File exceeds 25 MB limit.');
      return;
    }

    currentUploadedFile = file;
    csvFileName.textContent = file.name;
    csvFileSize.textContent = formatBytes(file.size);
    csvSelectedFile.classList.remove('hidden');
    csvDropzone.classList.add('hidden');
    btnUploadCsv.disabled = false;
  }

  function clearCsv() {
    currentUploadedFile = null;
    csvFileInput.value = '';
    csvSelectedFile.classList.add('hidden');
    csvDropzone.classList.remove('hidden');
    btnUploadCsv.disabled = true;
  }

  if (csvFileInput) {
    csvFileInput.addEventListener('change', () => chooseCsv(csvFileInput.files[0]));
  }
  if (csvRemoveFile) {
    csvRemoveFile.addEventListener('click', clearCsv);
  }

  if (csvDropzone) {
    ['dragenter', 'dragover'].forEach((eventName) => {
      csvDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        csvDropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
      csvDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        csvDropzone.classList.remove('dragover');
      });
    });
    csvDropzone.addEventListener('drop', (e) => {
      chooseCsv(e.dataTransfer.files[0]);
    });
  }

  // Upload Button Trigger
  if (btnUploadCsv) {
    btnUploadCsv.addEventListener('click', async () => {
      if (!currentUploadedFile) return;

      const formData = new FormData();
      formData.append('file', currentUploadedFile);

      btnUploadCsv.disabled = true;
      setUploadView('loading');
      document.getElementById('uploadLoadingTitle').textContent = 'Uploading & Validating…';
      document.getElementById('uploadLoadingMsg').textContent = 'Parsing timestamps and verifying column schema.';

      try {
        const response = await fetch(`${API_BASE}/api/upload`, {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
          const err = new Error(data.error || 'Upload failed');
          err.details = data.details || '';
          throw err;
        }

        // Render Preview Table & Statistics
        document.getElementById('previewRows').textContent = data.summary.total_records.toLocaleString();
        document.getElementById('previewMean').textContent = `${data.summary.mean_energy_kwh} kWh`;
        document.getElementById('previewMinMax').textContent = `${data.summary.min_energy_kwh} - ${data.summary.max_energy_kwh} kWh`;

        const headRow = document.getElementById('previewTableHead');
        headRow.innerHTML = '';
        data.columns.forEach((col) => {
          const th = document.createElement('th');
          th.textContent = col;
          headRow.appendChild(th);
        });

        const body = document.getElementById('previewTableBody');
        body.innerHTML = '';
        data.preview.forEach((row) => {
          const tr = document.createElement('tr');
          data.columns.forEach((col) => {
            const td = document.createElement('td');
            td.textContent = row[col] !== undefined ? row[col] : '—';
            tr.appendChild(td);
          });
          body.appendChild(tr);
        });

        setUploadView('preview');
        uploadStateBadge.innerHTML = '<i></i> VALIDATED';
        uploadStateBadge.classList.add('active');

        // Refresh analytics with newly active dataset preview
        await loadAnalytics();
      } catch (err) {
        document.getElementById('uploadErrorMessage').textContent = err.message;
        const detailsEl = document.getElementById('uploadErrorDetails');
        if (detailsEl) {
          detailsEl.textContent = err.details || '';
        }
        setUploadView('error');
        uploadStateBadge.innerHTML = '<i></i> ERROR';
        uploadStateBadge.classList.remove('active');
      } finally {
        btnUploadCsv.disabled = false;
      }
    });
  }

  // Retrain Trigger
  if (btnRetrainModel) {
    btnRetrainModel.addEventListener('click', async () => {
      btnRetrainModel.disabled = true;
      setUploadView('loading');
      document.getElementById('uploadLoadingTitle').textContent = 'Training ML Models…';
      document.getElementById('uploadLoadingMsg').textContent =
        'Fitting Linear Regression, Random Forest, and Gradient Boosting on chronological splits…';

      try {
        const res = await fetch(`${API_BASE}/api/train`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Training failed');

        alert(`ML Models successfully trained!\nSelected Best Model: ${data.best_model_name}\nTraining Time: ${data.training_time_seconds}s`);

        setUploadView('preview');
        await loadAnalytics();
        await loadModelMetrics();
        await checkHealth();
      } catch (err) {
        alert(`Training failed: ${err.message}`);
        setUploadView('preview');
      } finally {
        btnRetrainModel.disabled = false;
      }
    });
  }

  // Reset Demo Handlers
  async function handleResetDemo() {
    if (!confirm('Reset back to the default synthetic smart building demo dataset?')) return;
    try {
      const res = await fetch(`${API_BASE}/api/reset-demo`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Reset failed');
      alert('Reset to demo dataset completed.');
      clearCsv();
      setUploadView('empty');
      await loadAnalytics();
      await loadModelMetrics();
      await checkHealth();
    } catch (e) {
      alert(`Reset failed: ${e.message}`);
    }
  }

  if (btnResetDemo) btnResetDemo.addEventListener('click', handleResetDemo);
  if (btnRestoreDemoFromPreview) btnRestoreDemoFromPreview.addEventListener('click', handleResetDemo);

  // ------------------------------------------------------------------------
  // Initialize Application
  // ------------------------------------------------------------------------
  checkHealth();
  loadAnalytics();
  loadModelMetrics();
});
