# ==============================================

# TOKEN / COST OPTIMIZATION LAYER (DROP-IN)

# ==============================================



# REQUIREMENTS:

#   pip install chromadb sentence-transformers

#

# PLACE THIS FILE ANYWHERE IN YOUR BACKEND.

# IMPORT `optimized_pipeline` INSTEAD OF CALLING

# YOUR ORIGINAL PIPELINE ENTRY FUNCTION.

#

#   from optimization_layer import optimized_pipeline

#

# Then call:

#   result = optimized_pipeline("your query", debate_pipeline)

#

# Where debate_pipeline is your existing pipeline function.

# ==============================================



import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import json
import os

# -----------------------------

# 1. Embedding Model (Fast)

# -----------------------------

embedding_model = SentenceTransformer("all-mpnet-base-v2")

def embed(text: str):
    return embedding_model.encode(text).tolist()

# -----------------------------

# 2. Persistent Vector Store

# -----------------------------

# Initialize ChromaDB client with new API
chroma_client = chromadb.PersistentClient(path="./vector_memory")



# Create or load collection (reset for testing)

try:
    chroma_client.delete_collection("truth_analyzer_cache")
except:
    pass  # Collection might not exist

collection = chroma_client.get_or_create_collection(

    name="truth_analyzer_cache",

    metadata={"hnsw:space": "cosine"}    # cosine similarity

)





# -----------------------------

# 3. Cache Lookup Function

# -----------------------------

SIMILARITY_THRESHOLD = 0.92  # Recommended optimal band (0.88–0.95)



def check_cache(query: str):

    """Return cached result if query is similar to existing memory."""

    query_vec = embed(query)

    results = collection.query(

        query_embeddings=[query_vec],

        n_results=1

    )



    if len(results["ids"][0]) == 0:

        return None  # Empty DB



    similarity_score = results["distances"][0][0]

    cached_payload = results["documents"][0][0]



    # Chroma returns "distance", not similarity

    # Cosine distance = 1 - cosine similarity

    cosine_similarity = 1 - similarity_score



    if cosine_similarity >= SIMILARITY_THRESHOLD:

        return json.loads(cached_payload)



    return None





# -----------------------------

# 4. Insert Result Into Cache

# -----------------------------

def store_result(query: str, result_data: dict):

    result_json = json.dumps(result_data)

    collection.add(

        ids=[str(uuid.uuid4())],

        embeddings=[embed(query)],

        documents=[result_json]

    )

    # Note: Persistence is automatic with PersistentClient in newer ChromaDB versions





# -----------------------------

# 5. Optimization Wrapper

# -----------------------------

def optimized_pipeline(query: str, pipeline_fn):

    """

    Wrapper that:

    1. Checks cache using embeddings

    2. If similar → returns cached result

    3. If not → calls original pipeline and stores result

    """



    # Step 1 — Check cache

    cached = check_cache(query)

    if cached is not None:

        return {

            "source": "CACHE",

            "result": cached

        }



    # Step 2 — Run original pipeline

    fresh_output = pipeline_fn(query)



    # Step 3 — Store results for future reuse

    store_result(query, fresh_output)



    return {

        "source": "PIPELINE",

        "result": fresh_output

    }





# -----------------------------

# USAGE:

# -----------------------------

# Example of using the optimization layer.

# Pipeline functions must accept (query: str) and return a dict.

#

# def debate_pipeline(query: str):

#     ... your existing debate pipeline ...

#     return {"answer": "...", "confidence": 0.97}

#

# result = optimized_pipeline("What is quantum gravity?", debate_pipeline)

#

# print(result["source"])  # CACHE or PIPELINE

# print(result["result"])  # Actual data

# -----------------------------
