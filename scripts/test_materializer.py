"""Test materializer improvements."""

from ops_api.materializer import Materializer
from ops_api.event_store import EventStore


def main():
    """Test the materializer."""
    print("🧪 Testing materializer...")

    materializer = Materializer()

    # Test list_runs
    print("\n📊 Run Summaries:")
    runs = materializer.list_runs()
    for run in runs:
        print(f"  • Run ID: {run.run_id}")
        print(f"    Status: {run.status}")  # Should be "running" if recent events, "idle" otherwise
        print(f"    Mode: {run.mode}")      # Should be actual mode from config, not hardcoded "paper"
        print(f"    Last updated: {run.last_updated}")
        print()

    # Test block reasons
    print("📊 Block Reasons:")
    blocks = materializer.block_reasons()
    print(f"  Total block types: {len(blocks.reasons)}")
    for reason in blocks.reasons:
        print(f"    {reason.reason}: {reason.count}")

    # Test fills
    print("\n📊 Fills:")
    fills = materializer.list_fills()
    print(f"  Total fills: {len(fills)}")
    for fill in fills[:3]:  # Show first 3
        print(f"    {fill.symbol} {fill.side} {fill.qty} @ {fill.price}")

    print("\n✅ Materializer test complete!")


if __name__ == "__main__":
    main()
