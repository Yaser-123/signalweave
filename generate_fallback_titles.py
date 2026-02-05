"""
Generate fallback titles for all clusters (no Gemini API needed)
Uses simple keyword extraction from signal texts
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter
import re

load_dotenv()

def _fallback_title(signals):
    """Fallback title generation - extract capitalized words"""
    words = []
    for s in signals[:5]:
        words.extend(re.findall(r'\b[A-Z][a-z]+\b', s))
    
    if not words:
        return "Emerging Technology Cluster"
    
    common = Counter(words).most_common(3)
    title_words = [w for w, _ in common]
    return " / ".join(title_words)

def main():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    print("[INFO] Loading all signals...")
    all_signals = {}
    offset = None
    
    while True:
        response = client.scroll(
            collection_name="signals_hot",
            limit=100,
            offset=offset,
            with_payload=True
        )
        points, next_offset = response
        if not points:
            break
        
        for point in points:
            sig_id = point.payload.get('signal_id')
            text = point.payload.get('text', '')
            if sig_id and text:
                all_signals[sig_id] = text
        
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"[INFO] Loaded {len(all_signals)} signals")
    
    print("[INFO] Loading clusters...")
    offset = None
    clusters = []
    
    while True:
        response = client.scroll(
            collection_name="clusters_warm",
            limit=100,
            offset=offset,
            with_payload=True
        )
        points, next_offset = response
        if not points:
            break
        
        for point in points:
            cluster_id = point.payload.get('cluster_id')
            member_ids = point.payload.get('member_signal_ids', [])
            
            # Get signal texts
            signal_texts = [all_signals.get(sid, '') for sid in member_ids if sid in all_signals]
            
            if signal_texts:
                clusters.append({
                    'cluster_id': cluster_id,
                    'signal_texts': signal_texts
                })
        
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"[INFO] Found {len(clusters)} clusters with signals\n")
    
    success_count = 0
    
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        signal_texts = cluster['signal_texts']
        
        # Generate fallback title
        title = _fallback_title(signal_texts)
        
        # Save to Qdrant cluster_titles collection
        try:
            # Use simple vector for title cache
            client.upsert(
                collection_name="cluster_titles",
                points=[{
                    "id": cluster_id,
                    "vector": [0.5] * 384,  # Dummy embedding
                    "payload": {
                        "cluster_id": cluster_id,
                        "title": title
                    }
                }]
            )
            print(f"  ✅ [{cluster_id[:8]}...] → {title}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ [{cluster_id[:8]}...] → {title} (failed: {e})")
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully generated {success_count} fallback titles")
    print(f"{'='*60}")
    print("\nThese are keyword-based titles. When Gemini API quota resets,")
    print("run generate_missing_titles.py to upgrade to AI-generated titles.")

if __name__ == "__main__":
    main()
