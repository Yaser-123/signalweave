import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import time
import hashlib
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Persistent cache file
CACHE_FILE = Path("cluster_title_cache.json")

# Cache for cluster titles to avoid redundant API calls
_title_cache = {}


def _load_cache():
    """Load title cache from disk."""
    global _title_cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                _title_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load title cache: {e}")
            _title_cache = {}
    else:
        _title_cache = {}


def _save_cache():
    """Save title cache to disk."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_title_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save title cache: {e}")


# Load cache on module import
_load_cache()


def _get_cache_key(signals: List[str]) -> str:
    """Generate a unique cache key for a list of signals."""
    # Use first 3 signals sorted - more stable than all 10
    signal_sample = sorted(signals[:3])
    signal_text = "|".join(signal_sample)
    return hashlib.md5(signal_text.encode('utf-8')).hexdigest()


def generate_human_cluster_title(signals: List[str], cluster_id: str = None, use_cache: bool = True) -> str:
    """
    Generate a meaningful, human-readable cluster title using Gemini.
    
    Args:
        signals: List of signal texts in the cluster
        cluster_id: Optional cluster ID for stable caching
        use_cache: Whether to use cached titles (default: True)
    
    Returns:
        A short, descriptive title (max 8-10 words)
    """
    if not GEMINI_API_KEY:
        # Fallback to simple extraction if API key not available
        return _fallback_title(signals)
    
    # Use cluster_id as cache key if available, otherwise fall back to signal-based key
    if cluster_id:
        cache_key = f"cluster_{cluster_id}"
    else:
        cache_key = _get_cache_key(signals)
    
    # Check cache first
    if use_cache and cache_key in _title_cache:
        return _title_cache[cache_key]
    
    try:
        # Prepare signal texts (limit to first 5 for API efficiency)
        signal_sample = signals[:5]
        signal_text = "\n".join([f"- {s[:150]}" for s in signal_sample])  # Truncate long signals
        
        # Prompt for title generation
        prompt = f"""Analyze these emerging technology signals and create a single, clear title that explains what trend is emerging.

Signals:
{signal_text}

Requirements:
- Maximum 8-10 words
- Non-technical language
- Describes the trend, not just keywords
- Understandable by non-technical users
- Avoid jargon

Output ONLY the title, nothing else."""

        # Use gemini-2.5-flash (latest stable model)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        title = response.text.strip()
        
        # Ensure it's not too long
        if len(title.split()) > 12:
            title = " ".join(title.split()[:10]) + "..."
        
        # Cache the result and save to disk
        _title_cache[cache_key] = title
        _save_cache()
        
        return title
        
    except Exception as e:
        print(f"Gemini API error in title generation: {e}")
        # Cache fallback too
        fallback = _fallback_title(signals)
        _title_cache[cache_key] = fallback
        _save_cache()
        return fallback


def explain_cluster_with_gemini(cluster_signals: List[str], user_question: str) -> str:
    """
    Answer user questions about an emerging cluster using Gemini.
    
    Args:
        cluster_signals: List of signal texts in the cluster
        user_question: User's question about the cluster
    
    Returns:
        A clear, non-technical explanation
    """
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your .env file."
    
    try:
        # Prepare cluster context (limit signals and truncate for efficiency)
        signal_sample = cluster_signals[:10]
        signal_text = "\n".join([f"- {s[:200]}" for s in signal_sample])
        
        # System prompt + user question
        prompt = f"""You are an analyst explaining an emerging technology trend to a non-technical audience.

Use simple language.
Be factual and grounded in the provided evidence.
Avoid hype or speculation.
Keep responses concise (2-3 sentences).

CLUSTER SIGNALS:
{signal_text}

USER QUESTION: {user_question}

Provide a clear, helpful answer based only on the signals above."""

        # Use gemini-2.5-flash with retry
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as retry_error:
                if "429" in str(retry_error) or "quota" in str(retry_error).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        time.sleep(wait_time)
                        continue
                raise retry_error
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        return f"⚠️ Error generating explanation: {error_msg[:200]}"


def _fallback_title(signals: List[str]) -> str:
    """Fallback title generation when Gemini API is unavailable."""
    # Simple keyword extraction as fallback
    from collections import Counter
    import re
    
    words = []
    for s in signals[:5]:
        # Extract capitalized words and meaningful terms
        words.extend(re.findall(r'\b[A-Z][a-z]+\b', s))
    
    if not words:
        return "Emerging Technology Cluster"
    
    # Get top 3 most common words
    common = Counter(words).most_common(3)
    title_words = [w for w, _ in common]
    
    return " / ".join(title_words)
