# Multi-AI Orchestrator System

A clean and simple multi-agent AI orchestration system that runs AI models in parallel or sequential patterns. Features a flexible MetaPrompt system for dynamic prompt construction.

## Overview

This system provides three execution modes:
- **Parallel**: All agents process the same prompt simultaneously
- **Sequential**: Agents build upon each other's outputs progressively
- **Verification**: Cross-verification with semantic similarity analysis and consensus scoring

## Key Features

### MetaPrompt System
A flexible prompt templating system that:
- **Loads templates** from `data/prompts/` directory
- **Replaces placeholders** dynamically ({context}, {role})
- **Supports multiple tasks** (verification, brainstorming, analysis)
- **Falls back gracefully** when templates are missing

### Cross-Validation System
Semantic similarity-based validation that:
- **Compares outputs** using cosine similarity on text embeddings
- **Identifies consensus** statements across multiple agents
- **Flags discrepancies** when similarity falls below threshold
- **Generates detailed reports** with agreement scores and confidence metrics

## Architecture

### Core Components

- **main.py**: Entry point with CLI interface
- **agents/orchestrator.py**: Multi-agent orchestration (parallel/sequential)
- **agents/model_agent.py**: AI model wrapper with mock implementation
- **agents/meta_prompt.py**: Dynamic prompt construction from templates
- **data/prompts/**: Template files for different tasks

### Key Features

#### Multi-Model Verification
- Cross-validates responses from multiple AI models
- Calculates consensus scores and confidence levels
- Resolves conflicts using various strategies (weighted average, majority vote, expert consensus)

#### Iterative Brainstorming
- Generates diverse ideas across multiple iterations
- Clusters similar ideas and identifies unique approaches
- Refines prompts based on previous results

#### Semantic Analysis
- Uses embeddings for text similarity calculations
- Provides clustering and keyword extraction
- Supports both transformer-based and fallback similarity methods

#### Comprehensive Logging
- Audit trail for all operations
- Performance metrics and trend analysis
- Structured logging with component separation

## Installation

1. Clone or download the project
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. The system uses a mock AI implementation for demonstration purposes

## Usage

### Basic Usage

```python
from multi_ai_system.main import main

# Run verification analysis
main(query="What are the benefits of renewable energy?")

# Run brainstorming analysis
main(query="Innovative approaches to urban transportation", mode="brainstorming")

# Run both verification and brainstorming
main(query="Climate change solutions", mode="both")
```

### Command Line

```bash
# Parallel mode (default)
python main.py "What is quantum computing?"

# Sequential mode
python main.py --mode sequential "Build a comprehensive analysis"

# Verification mode
python main.py --mode verification "Machine learning is a subset of AI"
```

## Configuration

Edit `config.py` to:
- Add new AI models and providers
- Adjust consensus thresholds
- Modify prompt templates
- Configure brainstorming parameters

### Supported Models

The system supports multiple AI providers through a modular adapter architecture:

#### **Free Providers (No API Key Required)**
- **generic**: Mock responses for testing and development
- **huggingface**: HF Inference API (rate-limited free tier)

#### **Free Providers (API Key Required)**
- **groq**: Groq API with free tier - fast inference, Llama models
  - Set `GROQ_API_KEY` environment variable
  - Example: `{"name": "groq-agent", "provider": "groq", "model": "llama-3.1-8b-instant"}`

#### **Commercial Providers**
- **together**: Together AI (free credits available)
- **openrouter**: Multiple model access (some free options)
- **replicate**: Community models (may require token)

#### **Integration**

The system uses adapter classes in `agents/adapters/` that implement the `ModelAgent` interface. Add new providers by:

1. Create `agents/adapters/new_provider.py` inheriting from `ModelAgent`
2. Implement the `generate()` method
3. Add to `ADAPTER_MAP` in `agents/factory.py`
4. Update `enforce_free_providers.py` if it's a free provider

## API Reference

### Orchestrator

```python
from agents.orchestrator import Orchestrator
from agents.model_agent import ModelAgent

# Create agents
agents = [
    ModelAgent("cursor-fast", "expert"),
    ModelAgent("cursor-slow", "creative"),
    ModelAgent("cursor-balanced", "analyst")
]

# Initialize orchestrator
orchestrator = Orchestrator(agents)

# Run in different modes
parallel_results = orchestrator.run_parallel("Analyze this topic")
sequential_results = orchestrator.run_sequential(["Step 1", "Step 2", "Step 3"])
verification_results = orchestrator.run_analysis("Claim to verify", mode="verification")
```

### Verification

```python
from verification.cross_validate import CrossValidator

validator = CrossValidator(config)
validation_results = validator.validate_responses(responses)
```

### Brainstorming

```python
from brainstorming.idea_aggregator import IdeaAggregator

aggregator = IdeaAggregator(config)
ideas = aggregator.extract_ideas(responses)
clustered = aggregator.aggregate_and_cluster_ideas(ideas)
```

### MetaPrompt

```python
from agents.meta_prompt import MetaPrompt

meta_prompt = MetaPrompt()

# Generate prompt from template
prompt = meta_prompt.generate_prompt("verification", "Your question", "expert")

# Save custom template
meta_prompt.save_template("custom", "You are a {role} analyzing {context}")

# List available templates
templates = meta_prompt.list_templates()
```

### Cross-Validation

```python
from utils.cross_validation import cross_validate, analyze_consensus, generate_consensus_report

# Basic cross-validation
outputs = {"expert": "response1", "creative": "response2", "analyst": "response3"}
verified, flagged = cross_validate(outputs, threshold=0.8)

# Comprehensive consensus analysis
analysis = analyze_consensus(outputs)
print(f"Agreement Score: {analysis['agreement_score']:.2f}")

# Generate detailed report
report = generate_consensus_report(outputs)
print(report)
```

## Output Formats

The system provides structured JSON outputs with:
- Individual model responses
- Consensus answers with confidence scores
- Clustered ideas with themes
- Performance metrics and metadata

## Logging and Auditability

All operations are logged with:
- Component-level tracking
- Performance metrics
- Error handling and recovery
- Exportable audit trails

## Quick Start

### Unified Orchestrator Runner

Use the unified command-line interface to run any orchestration mode:

```bash
# Verification mode (3-agent consensus)
python run_orchestrator.py --mode verify --input "The Earth is round and orbits the Sun." --dry_run

# Triangulation mode (advanced multi-agent with retries)
python run_orchestrator.py --mode triangulate --input "Paris is the capital of France." --max_agents 5 --dry_run

# Brainstorming mode (multi-agent ideation)
python run_orchestrator.py --mode brainstorm --input "Generate innovative ideas for reducing plastic waste in cities" --dry_run

# Brainstorming with verification
python run_orchestrator.py --mode brainstorm --input "What are the benefits of renewable energy?" --run_verification --dry_run
```

## Core Features

### Verification Pipeline

Run 3-agent consensus verification with semantic similarity analysis:

```python
from orchestrator.pipelines.verify_pipeline import VerificationPipeline
from config import Config

pipeline = VerificationPipeline(similarity_threshold=0.75)
conf = Config()
agent_configs = conf.models[:3]

report = pipeline.run(
    text="The Eiffel Tower is in Paris and was completed in 1889.",
    agent_configs=agent_configs,
    dry_run=True  # Safe mode
)

print(f"Verdict: {report['verdict']}")
print(f"Confidence: {report['consensus']['confidence']}")
```

### Triangulation Orchestrator

Advanced multi-agent orchestration with automatic retries and dynamic scaling:

```python
from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
from config import Config

tri = TriangulationOrchestrator(
    similarity_threshold=0.75,
    max_agents=5,
    max_retries=2
)
conf = Config()
agent_configs = conf.models[:5]  # 3-9 agents

report = tri.run(
    text="Complex verification query here",
    agent_configs=agent_configs,
    dry_run=True
)

print(f"Final verdict: {report['verdict']}")
print(f"Overall confidence: {report['consensus']['confidence']}")
```

### Brainstorm Orchestrator

Multi-agent brainstorming with optional verification integration:

```python
from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator
from config import Config

brainstorm = BrainstormOrchestrator(
    max_agents=5,
    max_retries=2,
    dry_run=True
)
conf = Config()
agent_configs = conf.models[:5]

report = brainstorm.run(
    prompt="Generate innovative ideas for reducing plastic waste in cities",
    agent_configs=agent_configs,
    run_verification=True  # Optional: verify each idea
)

print(f"Generated {len(report['ideas'])} ideas")
for idea in report['ideas']:
    print(f"Idea: {idea['idea'][:100]}...")
    if 'verification' in idea:
        print(f"  Verified: {idea['verification'].get('verdict', 'unknown')}")
```

Features:
- **Multi-Agent Ideation**: 3-9 agents generate diverse ideas
- **Integrated Verification**: Optional verification of each idea using TriangulationOrchestrator
- **Retry Logic**: Automatic retries for failed idea generation
- **Structured Output**: JSON-serializable reports with agent metadata
- **Safe by Default**: Dry-run mode prevents accidental API calls

## Extending the System

### Adding New Models

1. Add model configuration to `Config.model_configs` with Cursor AI model names
2. Configure appropriate roles, temperatures, and token limits
3. Update validation logic if needed

### Custom Model Roles

Define specialized roles for different analysis tasks:
- **verification**: For fact-checking and consensus building
- **brainstorming**: For creative idea generation
- **analysis**: For in-depth examination and evaluation

### Adding New Providers

1. Create `agents/adapters/new_provider.py` inheriting from `ModelAgent`
2. Implement the `generate()` method
3. Add to `ADAPTER_MAP` in `agents/factory.py`
4. Update `enforce_free_providers.py` if it's a free provider

Example:
```python
from agents.base import ModelAgent

class NewProviderAdapter(ModelAgent):
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        # Your implementation here
        return "response"
```

### Creating Custom Templates

Add new prompt templates to `data/prompts/`:

```python
from agents.meta_prompt import MetaPrompt

meta_prompt = MetaPrompt()
meta_prompt.save_template("custom_task", """
You are a {role} handling: {context}

Please provide your specialized analysis.
""")
```

Templates support placeholders:
- `{context}`: The query or topic
- `{role}`: The model role (expert, creative, etc.)

### Custom Verification Methods

Extend `ConflictResolver` with new resolution strategies:
- Domain-specific consensus building
- Weighted voting schemes
- Expert model prioritization

### Enhanced Brainstorming

Modify `IdeaAggregator` for:
- Domain-specific idea evaluation
- Custom clustering algorithms
- Integration with external knowledge bases

## Dependencies

- `numpy`: Numerical computations for embeddings and metrics
- `sentence-transformers`: Text embeddings (optional, fallback available)
- `pickle`: Caching support

**Note**: This demo version uses mock AI responses. For production use, integrate with real AI APIs.

## License

This project is open-source. Please refer to individual component licenses for third-party dependencies.

## Contributing

Contributions welcome! Please:
1. Test your changes thoroughly
2. Update documentation
3. Follow the existing code style
4. Add appropriate logging and error handling
