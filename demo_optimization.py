#!/usr/bin/env python3
"""
Demonstration of the Token/Cost Optimization Layer

This script shows how the semantic caching system works to reduce API costs
and improve response times for similar queries.
"""

import os
import sys
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization_layer import optimized_pipeline


def expensive_debate_simulation(query: str) -> dict:
    """
    Simulate an expensive debate operation that would normally cost API tokens.
    In reality, this would call AI models and cost money/time.
    """
    print(f"🤖 Running expensive debate simulation for: '{query}'")
    time.sleep(1)  # Simulate API call delay

    return {
        "topic": query,
        "conclusion": f"After careful analysis, the debate concludes that {query.lower()} is a complex topic requiring further discussion.",
        "confidence": 0.87,
        "participants": ["AI Advocate", "AI Skeptic", "AI Moderator"],
        "rounds_completed": 3,
        "total_tokens_used": 1500,  # Simulated token cost
        "processing_time": 2.5
    }


def demo_caching():
    """Demonstrate the caching functionality."""
    print("🚀 Truth Analyzer Optimization Layer Demo\n")
    print("=" * 60)

    # Test 1: First query (expensive computation)
    print("\n1️⃣ First Query (Expensive - No Cache)")
    query1 = "Should social media platforms be regulated by governments?"
    start_time = time.time()
    result1 = optimized_pipeline(query1, expensive_debate_simulation)
    first_duration = time.time() - start_time

    print(f"   Source: {result1['source']}")
    print(".2f")
    print(f"   Tokens saved: 0 (first run)")

    # Test 2: Exact same query (should hit cache)
    print("\n2️⃣ Exact Same Query (Cache Hit)")
    query2 = query1  # Exact match
    start_time = time.time()
    result2 = optimized_pipeline(query2, expensive_debate_simulation)
    second_duration = time.time() - start_time

    print(f"   Source: {result2['source']}")
    print(".2f")
    tokens_saved = result1['result'].get('total_tokens_used', 0)
    print(f"   Tokens saved: {tokens_saved}")

    # Test 3: Semantically similar query (may or may not hit cache)
    print("\n3️⃣ Similar Query (Semantic Similarity)")
    query3 = "Do governments need to regulate social media companies?"
    start_time = time.time()
    result3 = optimized_pipeline(query3, expensive_debate_simulation)
    third_duration = time.time() - start_time

    print(f"   Source: {result3['source']}")
    print(".2f")
    if result3['source'] == 'CACHE':
        print(f"   Tokens saved: {tokens_saved}")
    else:
        print(f"   Tokens saved: 0 (different enough to require new computation)")

    # Test 4: Different topic (definitely no cache hit)
    print("\n4️⃣ Different Topic (No Cache)")
    query4 = "What are the benefits of renewable energy?"
    start_time = time.time()
    result4 = optimized_pipeline(query4, expensive_debate_simulation)
    fourth_duration = time.time() - start_time

    print(f"   Source: {result4['source']}")
    print(".2f")
    print(f"   Tokens saved: 0 (completely different topic)")

    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    total_time_original = first_duration + second_duration + third_duration + fourth_duration
    cache_hits = sum(1 for r in [result1, result2, result3, result4] if r['source'] == 'CACHE')
    total_tokens_saved = cache_hits * tokens_saved

    print(f"Total queries processed: 4")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache hit rate: {cache_hits/4*100:.1f}%")
    print(f"Total tokens saved: {total_tokens_saved}")
    print(".1f")

    print("\n💡 KEY BENEFITS:")
    print("  • Automatic semantic similarity detection")
    print("  • Zero-code changes to existing pipelines")
    print("  • Persistent vector storage across sessions")
    print("  • Cost reduction for repeated similar queries")
    print("  • Faster response times for cached results")

    print("\n🔧 CONFIGURATION:")
    print("  • Similarity threshold: 0.92 (configurable)")
    print("  • Embedding model: all-mpnet-base-v2")
    print("  • Vector store: ChromaDB with cosine similarity")
    print("  • Storage: Persistent SQLite/parquet files")


def demo_api_usage():
    """Show how to integrate with existing API."""
    print("\n" + "=" * 60)
    print("🔌 API INTEGRATION EXAMPLE")
    print("=" * 60)

    print("""
# In your existing Flask route:

from optimization_layer import optimized_pipeline
from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator

@app.route('/api/debate', methods=['POST'])
def optimized_debate():
    data = request.json
    topic = data.get('topic')
    use_cache = data.get('use_cache', True)

    if use_cache:
        # Use optimized pipeline
        result = optimized_pipeline(topic, lambda t: run_debate_logic(t))
        return jsonify({
            'source': result['source'],
            'data': result['result']
        })
    else:
        # Use regular pipeline
        result = run_debate_logic(topic)
        return jsonify({
            'source': 'PIPELINE',
            'data': result
        })
""")


if __name__ == "__main__":
    try:
        demo_caching()
        demo_api_usage()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
