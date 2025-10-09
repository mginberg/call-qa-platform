// Clean Professional Call QA Dashboard - Fixed Version

class CallQADashboard {
    constructor() {
        this.convosoConfig = {
            endpoints: {
                callLogs: '/log/retrieve',
                leadSearch: '/leads/search',
                recordings: '/leads/get-recordings'
            }
        };

        // Dashboard Data
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
        this.startDataRefresh();
        console.log('Call QA Dashboard initialized successfully');
    }

    setupTabNavigation() {
        const navTabs = document.querySelectorAll('.nav-tab');
        const tabContents = document.querySelectorAll('.tab-content');

        navTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;

                // Remove active class from all tabs and contents
                navTabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                const targetContent = document.getElementById(`${targetTab}-tab`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
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

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CallQADashboard();
});
