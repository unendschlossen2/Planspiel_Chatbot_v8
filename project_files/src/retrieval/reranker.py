from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np

def load_reranker_model(model_id: str, device: Optional[str] = None) -> Optional[CrossEncoder]:
    print(f"Versuche, Reranker-Modell '{model_id}' zu laden.")
    try:
        model = CrossEncoder(model_id, device=device)
        print(f"Reranker-Modell '{model_id}' erfolgreich geladen.")
        return model
    except Exception as e:
        print(f"Fehler beim Laden des Reranker-Modells '{model_id}': {e}")
        return None

def gap_based_rerank_and_filter(
        user_query: str,
        initial_retrieved_docs: List[Dict[str, Any]],
        reranker_model: CrossEncoder,
        min_absolute_rerank_score_threshold: float = 0.001, # NEUER PARAMETER
        min_chunks_to_llm: int = 1,
        max_chunks_to_llm: int = 5,
        min_chunks_for_gap_detection: int = 4,
        gap_detection_factor: float = 0.2,
        small_epsilon: float = 1e-5
) -> List[Dict[str, Any]]:

    print(f"Starte Reranking und Gap-basierte Filterung für {len(initial_retrieved_docs)} initiale Dokumente.")

    if not initial_retrieved_docs:
        print("Keine initialen Dokumente zum Reranken erhalten.")
        return []

    if not reranker_model:
        print("Kein Reranker-Modell bereitgestellt. Reranking wird übersprungen.")
        # Gebe bis zu max_chunks_to_llm der initialen Dokumente zurück, falls vorhanden
        return initial_retrieved_docs[:min(len(initial_retrieved_docs), max_chunks_to_llm)]


    doc_contents = []
    valid_docs_indices = []
    for i, doc in enumerate(initial_retrieved_docs):
        content = doc.get("document", "")
        if isinstance(content, str) and content.strip():
            doc_contents.append(content)
            valid_docs_indices.append(i)
        else:
            print(f"Dokument mit Index {i} (ID: {doc.get('id', 'N/A')}) hat keinen validen Inhalt und wird für Reranking übersprungen.")

    if not doc_contents:
        print("Keine Dokumente mit validem Inhalt für Reranking gefunden.")
        return []

    print(f"Reranke {len(doc_contents)} Dokumente mit validem Inhalt...")
    sentence_pairs = [[user_query, doc_content] for doc_content in doc_contents]

    try:
        scores = reranker_model.predict(sentence_pairs, show_progress_bar=True)
    except Exception as e:
        print(f"Fehler während der Vorhersage mit dem Reranker-Modell: {e}")
        return initial_retrieved_docs[:min(len(initial_retrieved_docs), max_chunks_to_llm)]

    reranked_docs_with_scores = []
    for i in range(len(scores)): # Iterate based on actual scores returned
        original_doc_idx = valid_docs_indices[i]
        doc_copy = initial_retrieved_docs[original_doc_idx].copy()
        doc_copy['rerank_score'] = float(scores[i]) # Sicherstellen, dass es ein float ist
        reranked_docs_with_scores.append(doc_copy)

    reranked_docs_with_scores.sort(key=lambda x: x['rerank_score'], reverse=True)

    print("Top gerankte Scores (bis zu 15): " + ", ".join([f"{doc['rerank_score']:.4f}" for doc in reranked_docs_with_scores[:15]]))

    # NEUER SCHRITT: Absolute Score Vorfilterung
    potentially_relevant_docs = [
        doc for doc in reranked_docs_with_scores if doc['rerank_score'] >= min_absolute_rerank_score_threshold
    ]
    num_potentially_relevant_docs = len(potentially_relevant_docs)
    print(f"{num_potentially_relevant_docs} Dokumente haben den absoluten Score-Schwellenwert von {min_absolute_rerank_score_threshold:.4f} erreicht oder überschritten.")

    if num_potentially_relevant_docs == 0:
        print("Keine Dokumente haben den absoluten Score-Schwellenwert erreicht. Gebe leere Liste zurück (oder min_chunks_to_llm der Original gerankten).")
        # Überlegung: Eventuell hier die Top min_chunks_to_llm der *original* gerankten Liste nehmen, falls alle unter Threshold?
        # Fürs Erste: leere Liste, wenn nichts "potentiell relevant" ist.
        # Oder, wenn min_chunks_to_llm > 0, die besten der *original* gerankten nehmen.
        if min_chunks_to_llm > 0 and reranked_docs_with_scores:
            print(f"Da min_chunks_to_llm > 0, nehme die besten {min(min_chunks_to_llm, len(reranked_docs_with_scores))} Dokumente der ursprünglichen Reranking-Liste.")
            return reranked_docs_with_scores[:min(min_chunks_to_llm, len(reranked_docs_with_scores))]
        return []


    if num_potentially_relevant_docs <= min_chunks_to_llm:
        print(f"Anzahl potenziell relevanter Dokumente ({num_potentially_relevant_docs}) ist <= min_chunks_to_llm ({min_chunks_to_llm}). Gebe diese Dokumente zurück.")
        return potentially_relevant_docs[:num_potentially_relevant_docs] # Gibt die potenziell relevanten zurück

    if num_potentially_relevant_docs < min_chunks_for_gap_detection:
        print(f"Weniger potenziell relevante Dokumente ({num_potentially_relevant_docs}) als min_chunks_for_gap_detection ({min_chunks_for_gap_detection}). Gap-Analyse nicht sinnvoll.")
        return potentially_relevant_docs[:min(num_potentially_relevant_docs, max_chunks_to_llm)]

    # Gap-Analyse wird nun auf 'potentially_relevant_docs' durchgeführt
    scores_only_relevant = np.array([doc['rerank_score'] for doc in potentially_relevant_docs])

    # Sicherstellen, dass wir mindestens 2 Scores für diff haben
    if len(scores_only_relevant) < 2:
        print("Weniger als 2 potenziell relevante Dokumente für die Delta-Berechnung. Gebe diese Dokumente bis max_chunks_to_llm zurück.")
        return potentially_relevant_docs[:min(len(scores_only_relevant), max_chunks_to_llm)]

    deltas = np.diff(scores_only_relevant) * -1
    print(f"Berechnete Deltas (Anzahl: {len(deltas)}) aus potenziell relevanten Scores: {deltas}")


    if len(deltas) == 0: # Sollte durch vorherigen Check abgedeckt sein, aber als Sicherheit
        print("Keine Deltas berechenbar. Gebe potenziell relevante Dokumente bis max_chunks_to_llm zurück.")
        return potentially_relevant_docs[:min(num_potentially_relevant_docs, max_chunks_to_llm)]

    median_of_deltas = np.median(deltas)
    print(f"Median der Score-Differenzen (Deltas) aus potenziell relevanten Scores: {median_of_deltas:.4f}")

    if median_of_deltas < small_epsilon:
        print("Median der Deltas aus potenziell relevanten Scores ist sehr klein. Keine Gap-Analyse, gebe potenziell relevante Dokumente bis max_chunks_to_llm zurück.")
        return potentially_relevant_docs[:min(num_potentially_relevant_docs, max_chunks_to_llm)]

    significant_gap_value = (gap_detection_factor * len(deltas)) * float(median_of_deltas)
    print(f"Signifikanter Gap-Wert (Faktor * Median der relevanten Deltas): {significant_gap_value:.4f}")

    num_to_take_after_gap = num_potentially_relevant_docs # Fallback: alle potenziell relevanten nehmen

    for i in range(len(deltas)):
        current_delta = deltas[i]
        # print(f"Prüfe Delta an Index {i} (zwischen Chunk {i+1} und {i+2} der pot. relevanten Liste): {current_delta:.4f}")
        if current_delta > significant_gap_value:
            num_to_take_after_gap = i + 1
            print(f"Signifikante Lücke nach Chunk {num_to_take_after_gap} (der potenziell relevanten Liste) gefunden (Delta {current_delta:.4f} > {significant_gap_value:.4f}).")
            break

    if num_to_take_after_gap == num_potentially_relevant_docs: # num_to_take_after_gap wurde nicht geändert
        print("Keine signifikante Lücke gemäß Faktor in potenziell relevanten Dokumenten gefunden.")
        # In diesem Fall bleiben wir bei allen potenziell relevanten (bis max_chunks_to_llm)

    # Endgültige Anzahl basierend auf Gap-Analyse und Min/Max-Grenzen
    final_num_to_take = max(min_chunks_to_llm, num_to_take_after_gap)
    final_num_to_take = min(final_num_to_take, max_chunks_to_llm)
    # Wichtig: Stelle sicher, dass wir nicht mehr nehmen, als potenziell relevant sind
    final_num_to_take = min(final_num_to_take, num_potentially_relevant_docs)


    print(f"Filterungsergebnis: Wähle die Top {final_num_to_take} Dokumente aus der potenziell relevanten Liste aus.")
    return potentially_relevant_docs[:final_num_to_take]