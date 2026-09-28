import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

from inference import MODEL_PATH, evaluate


def evaluate_all(episodes: int, model_path: str):
    cartons_to_test = [4, 8, 12]
    results = {}

    print("\n" + "=" * 70)
    print("    HiveMind Curriculum Inference Evaluation Suite")
    print("=" * 70)
    print(f"  Model        : {model_path}")
    print(f"  Episodes/stage: {episodes}")
    print(f"  Curriculum   : {cartons_to_test} cartons")
    print("=" * 70 + "\n")

    for c in cartons_to_test:
        print(f"\n[ RUNNING EVALUATION FOR {c} CARTONS ]")
        stats = evaluate(episodes=episodes, num_cartons=c, model_path=model_path)
        if stats:
            results[c] = stats

    print("\n\n" + "=" * 70)
    print("               CURRICULUM EVALUATION SUMMARY")
    print("=" * 70)

    if not results:
        print("  [ERROR] No results collected. Check model path.")
        return

    # Print Table Header
    print(
        f"  {'Cartons':<10} | {'Success Rate':<15} | {'Avg Makespan':<15} | {'Avg Collisions':<15}"
    )
    print("-" * 70)

    for c in cartons_to_test:
        if c in results:
            stats = results[c]
            success_str = f"{stats['success_rate'] * 100:.1f}% ({stats['successes']}/{stats['episodes']})"
            makespan_str = f"{stats['avg_makespan']:.1f}"
            collisions_str = f"{stats['avg_collisions']:.2f}"
            print(
                f"  {c:<10} | {success_str:<15} | {makespan_str:<15} | {collisions_str:<15}"
            )
        else:
            print(f"  {c:<10} | {'FAILED':<15} | {'-':<15} | {'-':<15}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the HiveMind RecurrentPPO model across all curriculum stages."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes per stage (default: 10).",
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH, help="Path to the model to load."
    )
    args = parser.parse_args()

    evaluate_all(episodes=args.episodes, model_path=args.model_path)
