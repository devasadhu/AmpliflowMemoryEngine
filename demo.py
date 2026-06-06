"""
demo.py — AmpliflowMemoryEngine retrieval demonstration
Shows the engine routing 4 different queries to 4 different ICP profiles.
"""

from memory_engine import AmpliflowMemoryEngine

DEMO_QUERIES = [

    {
        "label": "Local Service Business",
        "query": "Write a neighbourhood Facebook post for a local trades business celebrating 5 years and offering a seasonal promotion to drive footfall",
        "expected_icp": "icp_local_service_business.md",
    },
    {
        "label": "Tech / AI Founder",
        "query": "Write a Product Hunt launch post for our SaaS startup targeting early adopters and developers",
        "expected_icp": "icp_tech_ai_founder.md",
    },
    {
        "label": "Nonprofit / Charity",
        "query": "Write a donor thank-you email for our charity's annual fundraising campaign",
        "expected_icp": "icp_nonprofit_charity.md",
    },
    {
        "label": "Industrial / Manufacturing",
        "query": "Write a press release announcing our factory's ISO certification and safety audit results",
        "expected_icp": "icp_industrial_manufacturing.md",
    },
]


def run_demo():
    print("\n" + "=" * 65)
    print("  AMPLIFLOW MEMORY ENGINE — ICP ROUTING DEMO")
    print("=" * 65)

    # Initialise engine once — vectorstore loads from disk cache
    engine = AmpliflowMemoryEngine(user_id="user_102")
    engine.load_session_state()
    engine.build_vectorstore()

    results = []

    for i, demo in enumerate(DEMO_QUERIES, start=1):
        print(f"\n{'─' * 65}")
        print(f"  TEST {i}: {demo['label']}")
        print(f"{'─' * 65}")
        print(f"  QUERY    : {demo['query']}")

        context = engine.assemble_context(demo["query"])

        icp_source = context["icp_source"]
        tokens = context["token_budget"]["total"]
        budget_ok = context["token_budget"]["safe"]
        match = icp_source == demo["expected_icp"]

        print(f"  ICP HIT  : {icp_source}")
        print(f"  EXPECTED : {demo['expected_icp']}")
        print(f"  MATCH    : {'✓ CORRECT' if match else '✗ UNEXPECTED'}")
        print(f"  TOKENS   : {tokens}/4000 ({'OK' if budget_ok else 'EXCEEDED'})")

        # Show first 200 chars of retrieved ICP
        print(f"\n  ICP PREVIEW:")
        preview = context["icp_context"][:200].replace("\n", "\n    ")
        print(f"    {preview}...")

        results.append({
            "test": i,
            "label": demo["label"],
            "icp_hit": icp_source,
            "match": match,
            "tokens": tokens,
        })

    # Summary table
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'#':<4} {'Label':<30} {'Match':<10} {'Tokens'}")
    print(f"  {'─'*4} {'─'*30} {'─'*10} {'─'*10}")
    for r in results:
        status = "✓" if r["match"] else "✗"
        print(f"  {r['test']:<4} {r['label']:<30} {status:<10} {r['tokens']}/4000")

    correct = sum(1 for r in results if r["match"])
    print(f"\n  Score: {correct}/{len(results)} correct ICP matches")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()