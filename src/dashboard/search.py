# src/dashboard/search.py

from typing import List, Dict, Any, Set
import numpy as np
import re
import string


# Common English stopwords to filter out
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
}


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Steps:
    - Lowercase
    - Remove punctuation
    - Collapse repeated whitespace
    
    Args:
        text: Raw text string
    
    Returns:
        Normalized text string
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation but keep spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords from text.
    
    Rules:
    - Keep tokens with length >= 3
    - Remove stopwords
    - Preserve technical terms (AWS, GPU, etc.)
    
    Args:
        text: Normalized text string
    
    Returns:
        Set of keyword strings
    """
    # Normalize first
    normalized = normalize_text(text)
    
    # Split into tokens
    tokens = normalized.split()
    
    # Filter: length >= 3 and not stopwords
    keywords = {
        token for token in tokens
        if len(token) >= 3 and token not in STOPWORDS
    }
    
    return keywords


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_lexical_score(query_keywords: Set[str], cluster_signals: List[Dict[str, Any]]) -> float:
    """
    Compute lexical overlap score between query and cluster signals.
    
    Args:
        query_keywords: Set of keywords from user query
        cluster_signals: List of signal dictionaries with 'text' field
    
    Returns:
        Lexical overlap ratio (0.0 to 1.0)
    """
    if not query_keywords:
        return 0.0
    
    # Extract keywords from all signal titles in cluster
    cluster_keywords = set()
    for signal in cluster_signals:
        signal_text = signal.get('text', '')
        cluster_keywords.update(extract_keywords(signal_text))
    
    # Compute overlap
    overlap = query_keywords.intersection(cluster_keywords)
    
    # Overlap ratio: (# overlapping keywords) / (# query keywords)
    lexical_score = len(overlap) / len(query_keywords) if query_keywords else 0.0
    
    return lexical_score


def search_clusters_hybrid(
    query: str,
    clusters: List[Dict[str, Any]],
    embedding_model,
    min_final_score: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining semantic similarity and lexical overlap.
    
    Final score = 0.7 * semantic_score + 0.3 * lexical_score
    
    Filters:
    - Keep if semantic_score >= 0.45 OR lexical_score >= 0.2
    - Keep if final_score >= min_final_score
    
    Args:
        query: User's search query (e.g., "AWS Trainium3")
        clusters: List of all clusters (active + candidate)
        embedding_model: The embedding model to encode the query
        min_final_score: Minimum final score threshold (default: 0.35)
    
    Returns:
        List of matching clusters sorted by final_score, with metadata:
        - semantic_score
        - lexical_score
        - final_score
        - cluster_type (Active/Candidate)
        - All original cluster fields
    """
    if not query.strip():
        return []
    
    # Normalize query and extract keywords
    query_keywords = extract_keywords(query)
    
    # Embed the user query
    try:
        query_embedding = embedding_model.embed(query)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []
    
    results = []
    
    for cluster in clusters:
        # Skip clusters without centroids
        if "centroid" not in cluster:
            continue
        
        # 1. Compute semantic score (embedding-based)
        semantic_score = cosine_similarity(query_embedding, cluster["centroid"])
        
        # 2. Compute lexical score (keyword-based)
        cluster_signals = cluster.get("signals", [])
        lexical_score = compute_lexical_score(query_keywords, cluster_signals)
        
        # 3. Compute final score (weighted combination)
        final_score = 0.7 * semantic_score + 0.3 * lexical_score
        
        # 4. Filter by thresholds
        # Keep if: (semantic >= 0.40 OR lexical >= 0.15) AND final >= min_final_score
        if (semantic_score >= 0.40 or lexical_score >= 0.15) and final_score >= min_final_score:
            # Determine cluster type
            cluster_type = "Active" if cluster.get("signal_count", 0) >= 3 else "Candidate"
            
            # Add metadata
            result = {
                **cluster,  # Include all original cluster data
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "final_score": final_score,
                "cluster_type": cluster_type
            }
            results.append(result)
    
    # Sort by final_score (desc), then signal_count (desc) as tie-breaker
    results.sort(
        key=lambda x: (x["final_score"], x.get("signal_count", 0)),
        reverse=True
    )
    
    return results


# Legacy function for backward compatibility
def search_clusters(
    query: str,
    clusters: List[Dict[str, Any]],
    embedding_model,
    similarity_threshold: float = 0.55
) -> List[Dict[str, Any]]:
    """
    Legacy search using only semantic similarity.
    Deprecated - use search_clusters_hybrid instead.
    """
    # Call hybrid search with equivalent settings
    results = search_clusters_hybrid(
        query=query,
        clusters=clusters,
        embedding_model=embedding_model,
        min_final_score=similarity_threshold * 0.7  # Approximate conversion
    )
    
    # Add similarity_score alias for backward compatibility
    for result in results:
        result["similarity_score"] = result["semantic_score"]
    
    return results
