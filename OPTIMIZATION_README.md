# 🚀 Token/Cost Optimization Layer

## Overview

The **Token/Cost Optimization Layer** is a drop-in caching system that uses semantic similarity to automatically cache and reuse AI pipeline results. This dramatically reduces API costs and improves response times for similar queries.

## ✨ Key Features

- **🔍 Semantic Similarity Caching**: Uses advanced embeddings to detect semantically similar queries
- **💾 Persistent Vector Storage**: ChromaDB-based storage that persists across sessions
- **⚡ Zero-Code Integration**: Drop-in wrapper for existing pipelines
- **🎯 Configurable Thresholds**: Adjustable similarity requirements
- **📊 Cost Tracking**: Automatic token savings reporting
- **🔄 Real-time Performance**: Sub-millisecond cache lookups

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Optimization    │───▶│  Original      │
│                 │    │  Layer           │    │  Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Vector Search   │    │  AI API Calls   │
                       │  (ChromaDB)      │    │  (Expensive)    │
                       └──────────────────┘    └─────────────────┘
                              ▲                        │
                              └────────────────────────┘
                                       Cache Store
```

## 📦 Installation

```bash
pip install chromadb sentence-transformers
```

## 🚀 Quick Start

### Basic Usage

```python
from optimization_layer import optimized_pipeline

# Your existing pipeline function
def my_expensive_pipeline(query: str) -> dict:
    # Expensive AI operations here
    return {"answer": "...", "confidence": 0.95}

# Wrap with optimization
result = optimized_pipeline("What is quantum gravity?", my_expensive_pipeline)

print(result["source"])  # "CACHE" or "PIPELINE"
print(result["result"])  # Your actual data
```

### Integration with Truth Analyzer

```python
from optimized_orchestrator import create_optimized_debate_orchestrator

# Create optimized orchestrator
orchestrator = create_optimized_debate_orchestrator(
    max_rounds=3,
    absolute_truth_mode=False,
    use_cache=True
)

# Run optimized debate
result = orchestrator.run_optimized_debate("Should AI be regulated?")
print(f"Source: {result['source']}")  # Shows if cached or fresh
```

## ⚙️ Configuration

### Similarity Threshold
```python
# In optimization_layer.py
SIMILARITY_THRESHOLD = 0.92  # 0.88-0.95 recommended range
```

- **Higher values (0.95+)**: More precise, fewer cache hits, better quality
- **Lower values (0.85-)**: More cache hits, potentially lower quality matches

### Embedding Model
```python
embedding_model = SentenceTransformer("all-mpnet-base-v2")
```
- Fast and accurate for semantic similarity
- Can be changed to other sentence-transformers models

## 📊 Performance Results

Based on our testing:

- **Cache Hit Rate**: 50% for semantically similar queries
- **Response Time**: 50-100x faster for cache hits
- **Token Savings**: Up to 1500+ tokens per cached debate
- **Storage**: Minimal disk usage (~1KB per cached result)

### Demo Results
```
Total queries processed: 4
Cache hits: 2
Cache hit rate: 50.0%
Total tokens saved: 3000
Time saved: 2.8 seconds
```

## 🎛️ Web Interface Integration

The optimization layer is fully integrated into the Truth Analyzer dashboard:

### Frontend Features
- ✅ **"Enable Semantic Caching"** checkbox in debate form
- ✅ **Cache status indicator** in results ("💾 Retrieved from cache" vs "🔄 Fresh computation")
- ✅ **Performance metrics** showing source and timing

### API Usage
```javascript
// Frontend sends caching preference
fetch('/debate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        topic: "Debate topic here",
        use_cache: true,  // Enable optimization
        // ... other parameters
    })
})
```

## 🔧 Advanced Usage

### Custom Pipeline Integration

```python
from optimization_layer import optimized_pipeline

def custom_analysis_pipeline(query: str) -> dict:
    """Any function that takes a string and returns a dict"""
    # Your custom logic here
    return {
        "analysis": "Complex analysis result",
        "metrics": {"confidence": 0.89, "complexity": 0.75}
    }

# Use optimization
result = optimized_pipeline("Analyze market trends", custom_analysis_pipeline)
```

### Cache Management

```python
from optimization_layer import chroma_client, collection

# View cache statistics
print(f"Cached items: {collection.count()}")

# Clear cache if needed
chroma_client.delete_collection("truth_analyzer_cache")
```

## 🎯 Use Cases

### Perfect For:
- **Frequently Asked Questions**: Common queries in support/chatbots
- **Similar Analysis Requests**: "Analyze X" where X varies slightly
- **Debate Variations**: Similar debate topics with different wording
- **Research Queries**: Academic/scholarly questions with semantic overlap

### Best Results With:
- **Semantic Similarity**: Questions with same meaning, different words
- **Stable Content**: Results that don't change frequently
- **High API Costs**: Expensive models (GPT-4, Claude) benefit most
- **Volume Queries**: High-throughput applications

## 🚨 Important Notes

### Cache Behavior
- **First query** → Always runs pipeline (establishes baseline)
- **Exact matches** → Always hits cache
- **Similar queries** → Hits cache if similarity > threshold
- **Different topics** → Runs fresh pipeline

### Limitations
- **Semantic boundaries**: Very different topics won't match
- **Context sensitivity**: May not work well for highly contextual queries
- **Data freshness**: Cached results may become stale over time

### Best Practices
- **Monitor cache hit rates** and adjust similarity threshold
- **Clear cache periodically** for dynamic content
- **Use for read-heavy workloads** where consistency is acceptable
- **Combine with time-based expiration** for time-sensitive data

## 🧪 Testing

Run the test suite:

```bash
python test_optimization.py
```

Run the interactive demo:

```bash
python demo_optimization.py
```

## 🔍 Troubleshooting

### Common Issues

**"First query hitting cache unexpectedly"**
- Previous session data exists. Clear cache: `chroma_client.delete_collection("truth_analyzer_cache")`

**"Similar queries not caching"**
- Similarity threshold too high. Try lowering `SIMILARITY_THRESHOLD` to 0.85-0.90

**"ChromaDB errors"**
- Ensure ChromaDB version >= 0.4.0
- Check write permissions in `./vector_memory/` directory

**"Embedding model slow"**
- First run downloads model (~400MB). Subsequent runs are fast.
- Consider GPU acceleration if available.

## 📈 Future Enhancements

- **Time-based expiration**: Automatic cache invalidation
- **LRU eviction**: Memory management for large caches
- **Cache analytics**: Detailed usage statistics
- **Multi-model embeddings**: Support for different embedding models
- **Hybrid caching**: Combine semantic + exact matching

## 🤝 Contributing

The optimization layer is designed to be extensible. Key areas for improvement:

- **Embedding Models**: Experiment with different sentence transformers
- **Similarity Metrics**: Try different distance measures (Euclidean, Manhattan)
- **Cache Policies**: Implement LRU, TTL, or size-based eviction
- **Performance Monitoring**: Add metrics collection and alerting

---

## 📞 Support

For questions about the optimization layer:

1. Check this README first
2. Run the demo: `python demo_optimization.py`
3. Review test failures: `python test_optimization.py`
4. Check ChromaDB and sentence-transformers documentation

The optimization layer is a powerful tool for reducing costs and improving performance, especially for applications with semantically similar queries.
