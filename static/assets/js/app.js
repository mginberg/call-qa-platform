document.addEventListener('DOMContentLoaded', () => {

class CallQADashboard {
    constructor() {
        this.convosoConfig = {
            endpoints: {
                callLogs: '/log/retrieve',
                leadSearch: '/leads/search',
                recordings: '/leads/get-recordings'
            }
        };
        
        this.data = {
            calls: [],
            filteredCalls: [],
            agents: [],
            campaigns: [],
            selectedCalls: new Set()
        };

        this.filters = {
            campaign: '',
            agent: '',
            status: '',
            callType: ''
        };

        this.dashboardData = {
            agent: {
                name: 'Alex Wilson',
                initials: 'AW',
                role: 'Senior Call Agent',
                email: 'alex.wilson@company.com',
                phone: '3269856696',
                engagementScore: 65
            },
            metrics: {
                qaScore: 85.2,
                conversionRate: 23.7,
                totalCalls: 847,
                avgDuration: 4.2,
                transferSuccess: 78,
                flaggedCalls: 23,
                performanceRank: 3
            },
            chartData: {
                callVolume: {
                    labels: ['02/2024', '03/2024', '04/2024', '05/2024', '06/2024'],
                    data: [650, 780, 920, 850, 760]
                },
                dispositions: {
                    labels: ['SALE', 'DROP', 'AM', 'NI', 'CB'],
                    data: [32, 45, 23, 8, 12],
                    colors: ['#2ECC71', '#E74C3C', '#F39C12', '#95A5A6', '#3498DB']
                }
            }
        };

        this.charts = {};
        this.init();
    }

    init() {
        this.setupTabNavigation();
        this.setupCharts();
        this.bindEvents();
        this.setupCallLogHandlers();
        this.startDataRefresh();
        console.log('Call QA Dashboard initialized successfully');
    }

    setupCallLogHandlers() {
        const fetchBtn = document.getElementById('fetch-calls-btn');
        if (fetchBtn) {
            fetchBtn.addEventListener('click', () => this.fetchCallLogs());
        }

        const processBtn = document.getElementById('process-selected-btn');
        if (processBtn) {
            processBtn.addEventListener('click', () => this.processSelectedCalls());
        }

        const selectAllCheckbox = document.getElementById('select-all-calls');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => this.toggleSelectAll(e.target.checked));
        }

        ['campaign-filter', 'agent-filter', 'status-filter', 'calltype-filter'].forEach(filterId => {
            const filterEl = document.getElementById(filterId);
            if (filterEl) {
                filterEl.addEventListener('change', () => this.applyFilters());
            }
        });

        const closeModal = document.getElementById('close-modal');
        if (closeModal) {
            closeModal.addEventListener('click', () => this.closeProcessingModal());
        }
    }

    setupTabNavigation() {
        const navItems = document.querySelectorAll('.sidebar-nav-item');
        const tabContents = document.querySelectorAll('.tab-content');

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const targetTab = item.dataset.tab;

                navItems.forEach(i => i.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                item.classList.add('active');
                const targetContent = document.getElementById(`${targetTab}-tab`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    }

    async fetchCallLogs() {
        const fetchBtn = document.getElementById('fetch-calls-btn');
        const originalText = fetchBtn.innerHTML;
        
        try {
            fetchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            fetchBtn.disabled = true;

            const campaignFilter = document.getElementById('campaign-filter')?.value || '';
            const agentFilter = document.getElementById('agent-filter')?.value || '';
            const statusFilter = document.getElementById('status-filter')?.value || '';
            const calltypeFilter = document.getElementById('calltype-filter')?.value || '';

            const response = await this.makeConvosoRequest('/log/retrieve', {
                limit: 100,
                start_time: this.getDateString(-30),
                end_time: this.getDateString(0),
                campaign_filter: campaignFilter,
                agent_filter: agentFilter,
                status_filter: statusFilter,
                calltype_filter: calltypeFilter
            });

            if (response && response.calls) {
                this.data.calls = response.calls;
                this.data.filteredCalls = [...this.data.calls];
                
                if (response.filters) {
                    this.updateFilterDropdowns(response.filters);
                }
                
                this.renderCallsTable();
            }
        } catch (error) {
            console.error('Error fetching call logs:', error);
            alert('Error fetching call logs: ' + error.message);
        } finally {
            fetchBtn.innerHTML = originalText;
            fetchBtn.disabled = false;
        }
    }

    updateFilterDropdowns(filters) {
        const campaignFilter = document.getElementById('campaign-filter');
        const agentFilter = document.getElementById('agent-filter');
        const statusFilter = document.getElementById('status-filter');
        const calltypeFilter = document.getElementById('calltype-filter');
        
        if (campaignFilter && filters.campaigns) {
            const currentValue = campaignFilter.value;
            campaignFilter.innerHTML = '<option value="">All Campaigns</option>' +
                filters.campaigns.map(c => `<option value="${c}">${c}</option>`).join('');
            campaignFilter.value = currentValue;
        }
        
        if (agentFilter && filters.agents) {
            const currentValue = agentFilter.value;
            agentFilter.innerHTML = '<option value="">All Agents</option>' +
                filters.agents.map(a => `<option value="${a}">${a}</option>`).join('');
            agentFilter.value = currentValue;
        }
        
        if (statusFilter && filters.statuses) {
            const currentValue = statusFilter.value;
            statusFilter.innerHTML = '<option value="">All Status</option>' +
                filters.statuses.map(s => `<option value="${s}">${s}</option>`).join('');
            statusFilter.value = currentValue;
        }
        
        if (calltypeFilter && filters.calltypes) {
            const currentValue = calltypeFilter.value;
            calltypeFilter.innerHTML = '<option value="">All Types</option>' +
                filters.calltypes.map(t => `<option value="${t}">${t}</option>`).join('');
            calltypeFilter.value = currentValue;
        }
    }

    applyFilters() {
        this.fetchCallLogs();
    }

    renderCallsTable() {
        const tbody = document.getElementById('calls-table-body');
        if (!tbody) return;

        if (this.data.filteredCalls.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="no-data">No calls match the selected filters.</td></tr>';
            return;
        }

        tbody.innerHTML = this.data.filteredCalls.map(call => `
            <tr>
                <td><input type="checkbox" class="call-checkbox" data-call-id="${call.call_id}"></td>
                <td>${call.lead_id || 'N/A'}</td>
                <td>${call.agent || 'Unknown'}</td>
                <td><span class="status-badge">${call.status_name || call.status || 'N/A'}</span></td>
                <td>${call.customer_name || 'N/A'}</td>
                <td>${this.formatDuration(call.duration)}</td>
                <td>${this.formatDateTime(call.datetime)}</td>
                <td><span class="status-badge status-${call.processing_status}">${call.processing_status}</span></td>
                <td>
                    <button class="btn-icon" onclick="dashboard.viewCallDetails('${call.call_id}')" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        document.querySelectorAll('.call-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => this.toggleCallSelection(e.target.dataset.callId, e.target.checked));
        });

        this.updateSelectedCount();
    }

    toggleCallSelection(callId, selected) {
        if (selected) {
            this.data.selectedCalls.add(callId);
        } else {
            this.data.selectedCalls.delete(callId);
        }
        this.updateSelectedCount();
    }

    toggleSelectAll(selected) {
        document.querySelectorAll('.call-checkbox').forEach(checkbox => {
            checkbox.checked = selected;
            this.toggleCallSelection(checkbox.dataset.callId, selected);
        });
    }

    updateSelectedCount() {
        const processBtn = document.getElementById('process-selected-btn');
        if (processBtn) {
            processBtn.disabled = this.data.selectedCalls.size === 0;
            const count = this.data.selectedCalls.size;
            processBtn.innerHTML = `<i class="fas fa-cog"></i> Process Selected (${count})`;
        }
    }

    async processSelectedCalls() {
        if (this.data.selectedCalls.size === 0) return;

        this.showProcessingModal();
        const selectedCallIds = Array.from(this.data.selectedCalls);
        
        for (let i = 0; i < selectedCallIds.length; i++) {
            const callId = selectedCallIds[i];
            const progress = ((i + 1) / selectedCallIds.length) * 100;
            
            this.updateProgress(progress, `Processing call ${i + 1} of ${selectedCallIds.length}...`);
            this.addLogEntry(`Processing call ${callId}...`, 'info');
            
            await this.simulateProcessing(1000);
            
            this.addLogEntry(`Call ${callId} processed successfully`, 'success');
        }
        
        this.updateProgress(100, 'Processing complete!');
        this.addLogEntry('All calls processed successfully', 'success');
        
        setTimeout(() => {
            this.closeProcessingModal();
            this.data.selectedCalls.clear();
            this.fetchCallLogs();
        }, 2000);
    }

    showProcessingModal() {
        const modal = document.getElementById('processing-modal');
        if (modal) {
            modal.style.display = 'block';
            document.getElementById('progress-fill').style.width = '0%';
            document.getElementById('progress-text').textContent = 'Preparing to process calls...';
            document.getElementById('processing-log').innerHTML = '';
        }
    }

    closeProcessingModal() {
        const modal = document.getElementById('processing-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    updateProgress(percent, text) {
        const fill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        if (fill) fill.style.width = `${percent}%`;
        if (progressText) progressText.textContent = text;
    }

    addLogEntry(message, type = 'info') {
        const log = document.getElementById('processing-log');
        if (log) {
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
    }

    simulateProcessing(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    viewCallDetails(callId) {
        console.log('View details for call:', callId);
        alert('Call details view coming soon');
    }

    formatDuration(seconds) {
        if (!seconds) return 'N/A';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatDateTime(datetime) {
        if (!datetime) return 'N/A';
        const date = new Date(datetime);
        return date.toLocaleString();
    }

    setupCharts() {
        this.setupEngagementGauge();
        this.setupQAScoreGauge();
        this.setupVolumeChart();
        this.setupDispositionChart();
    }

    setupEngagementGauge() {
        const ctx = document.getElementById('engagementGauge');
        if (!ctx) return;

        const score = this.dashboardData.agent.engagementScore;

        this.charts.engagement = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: ['rgba(255, 255, 255, 0.9)', 'rgba(255, 255, 255, 0.2)'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutCubic'
                }
            }
        });
    }

    setupQAScoreGauge() {
        const ctx = document.getElementById('qaScoreGauge');
        if (!ctx) return;

        const score = this.dashboardData.metrics.qaScore;

        this.charts.qaScore = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: ['rgba(255, 255, 255, 0.9)', 'rgba(255, 255, 255, 0.3)'],
                    borderWidth: 0,
                    cutout: '70%'
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutCubic'
                }
            }
        });
    }

    setupVolumeChart() {
        const ctx = document.getElementById('volumeChart');
        if (!ctx) return;

        const chartData = this.dashboardData.chartData.callVolume;

        this.charts.volume = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Call Volume',
                    data: chartData.data,
                    backgroundColor: '#1ABC9C',
                    borderColor: '#16A085',
                    borderWidth: 0,
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#2C3E50',
                        titleColor: '#FFFFFF',
                        bodyColor: '#FFFFFF',
                        borderColor: '#34495E',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return context.parsed.y + ' calls';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: '#7F8C8D',
                            font: { size: 11, weight: 500 }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#F8F9FA',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#7F8C8D',
                            font: { size: 11, weight: 500 }
                        }
                    }
                },
                animation: {
                    duration: 1500,
                    easing: 'easeInOutCubic'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    setupDispositionChart() {
        const ctx = document.getElementById('dispositionChart');
        if (!ctx) return;

        const chartData = this.dashboardData.chartData.dispositions;

        this.charts.disposition = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: chartData.labels,
                datasets: [{
                    data: chartData.data,
                    backgroundColor: chartData.colors,
                    borderWidth: 0,
                    cutout: '60%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#2C3E50',
                        titleColor: '#FFFFFF',
                        bodyColor: '#FFFFFF',
                        borderColor: '#34495E',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return context.parsed + '% of calls';
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutCubic'
                },
                interaction: {
                    intersect: false
                }
            }
        });
    }

    bindEvents() {
        // Chart action buttons
        const chartActionBtns = document.querySelectorAll('.chart-action-btn');
        chartActionBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleChartAction(btn);
            });
        });

        // Metric card hover effects
        const metricCards = document.querySelectorAll('.metric-card');
        metricCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-4px)';
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(-2px)';
            });
        });

        // Window resize handler for charts
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                Object.values(this.charts).forEach(chart => {
                    if (chart && typeof chart.resize === 'function') {
                        chart.resize();
                    }
                });
            }, 100);
        });
    }

    handleChartAction(btn) {
        const isExpand = btn.querySelector('i').classList.contains('fa-expand');
        const chartCard = btn.closest('.chart-card');
        const chartTitle = chartCard.querySelector('.chart-title').textContent;

        if (isExpand) {
            this.expandChart(chartTitle);
        } else {
            this.showChartOptions(chartTitle);
        }
    }

    expandChart(chartTitle) {
        console.log(`Expanding chart: ${chartTitle}`);
        // Future implementation for chart expansion
    }

    showChartOptions(chartTitle) {
        console.log(`Showing options for chart: ${chartTitle}`);
        // Future implementation for chart options menu
    }

    // Data refresh simulation
    startDataRefresh() {
        setInterval(() => {
            this.updateMetrics();
        }, 30000); // Update every 30 seconds
    }

    updateMetrics() {
        // Simulate small random changes in metrics
        const variations = {
            qaScore: (Math.random() - 0.5) * 2,
            conversionRate: (Math.random() - 0.5) * 1,
            totalCalls: Math.floor((Math.random() - 0.5) * 20),
            transferSuccess: (Math.random() - 0.5) * 3,
            flaggedCalls: Math.floor((Math.random() - 0.5) * 6)
        };

        // Update metrics with bounds checking
        Object.keys(variations).forEach(key => {
            if (key === 'qaScore' || key === 'conversionRate' || key === 'transferSuccess') {
                this.dashboardData.metrics[key] = Math.max(0, Math.min(100, this.dashboardData.metrics[key] + variations[key]));
            } else {
                this.dashboardData.metrics[key] = Math.max(0, this.dashboardData.metrics[key] + variations[key]);
            }
        });

        // Update QA Score chart
        if (this.charts.qaScore) {
            const newScore = this.dashboardData.metrics.qaScore;
            this.charts.qaScore.data.datasets[0].data = [newScore, 100 - newScore];
            this.charts.qaScore.update('none');

            // Update the gauge text
            const gaugeText = document.querySelector('.gauge-text');
            if (gaugeText) {
                gaugeText.textContent = Math.round(newScore);
            }
        }

        console.log('Metrics updated:', this.dashboardData.metrics);
    }

    // Convoso API Integration
    async fetchConvosoData() {
        try {
            const response = await this.makeConvosoRequest('/log/retrieve', {
                limit: 100,
                start_time: this.getDateString(-30),
                end_time: this.getDateString(0)
            });

            if (response && response.data) {
                this.processConvosoData(response.data);
            }
        } catch (error) {
            console.warn('Convoso API call failed, using simulated data:', error.message);
        }
    }

    async makeConvosoRequest(endpoint, data) {
        const apiEndpoints = {
            '/log/retrieve': '/api/convoso/call-logs',
            '/leads/search': '/api/convoso/lead-search',
            '/leads/get-recordings': '/api/convoso/recordings'
        };
        
        const backendEndpoint = apiEndpoints[endpoint] || endpoint;
        const response = await fetch(backendEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    processConvosoData(callLogs) {
        // Process real call logs to update metrics
        const totalCalls = callLogs.length;
        const sales = callLogs.filter(call => call.status === 'SALE').length;
        const drops = callLogs.filter(call => call.status === 'DROP').length;
        const appointments = callLogs.filter(call => call.status === 'AM').length;

        // Update metrics based on real data
        this.dashboardData.metrics.totalCalls = totalCalls;
        this.dashboardData.metrics.conversionRate = totalCalls > 0 ? (sales / totalCalls * 100) : 0;

        // Update disposition chart data
        this.dashboardData.chartData.dispositions.data = [
            parseFloat((sales / totalCalls * 100).toFixed(1)),
            parseFloat((drops / totalCalls * 100).toFixed(1)),
            parseFloat((appointments / totalCalls * 100).toFixed(1)),
        ];

        // Refresh charts with new data
        if (this.charts.disposition) {
            this.charts.disposition.data.datasets[0].data = this.dashboardData.chartData.dispositions.data;
            this.charts.disposition.update();
        }

        console.log('Updated metrics from Convoso data:', this.dashboardData.metrics);
    }

    getDateString(daysOffset) {
        const date = new Date();
        date.setDate(date.getDate() + daysOffset);
        return date.toISOString().slice(0, 19).replace('T', ' ');
    }
}

const dashboard = new CallQADashboard();
window.dashboard = dashboard;

});
