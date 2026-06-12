import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy.orm import Session

from models.models import Incident, Category
from database import SessionLocal


# Persistent ChromaDB storage
# This creates/uses a local folder named chroma_db in your project.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="incidents",
    embedding_function=embedding_function
)

# Semantic similarity distance cutoff threshold
MAX_DISTANCE = 1.2


def format_incident_for_chroma(inc: Incident, db: Session = None) -> str:
    """
    Convert an incident into a clean text document for semantic search.
    Only useful data should be indexed in ChromaDB.
    """
    category_name = "Inconnu"
    if inc.category_id:
        if db:
            cat = db.query(Category).filter(Category.id == inc.category_id).first()
            if cat:
                category_name = cat.nom
        else:
            temp_db = SessionLocal()
            try:
                cat = temp_db.query(Category).filter(Category.id == inc.category_id).first()
                if cat:
                    category_name = cat.nom
            finally:
                temp_db.close()

    return f"""Category: {category_name}
Problem: {inc.description or inc.titre or ""}
Cause: {inc.cause or ""}
Solution: {inc.solution or ""}
Result: Resolved"""


def load_incidents_to_chroma():
    """
    Reload all incidents from MariaDB into ChromaDB.

    Useful when initializing or rebuilding the vector database.
    """

    db: Session = SessionLocal()

    try:
        incidents = db.query(Incident).all()

        documents = []
        ids = []
        metadatas = []

        for inc in incidents:
            documents.append(format_incident_for_chroma(inc, db))
            ids.append(str(inc.id))
            metadatas.append({
                "incident_id": inc.id,
                "statut": inc.statut or "",
                "type_probleme": inc.type_probleme or "",
                "equipement": inc.equipement or "",
            })

        if not ids:
            return 0

        # Safer than delete + add:
        # If the id exists, it updates it.
        # If the id does not exist, it creates it.
        collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        return len(incidents)

    finally:
        db.close()


def add_solved_incident_to_chroma(incident: Incident, db: Session = None):
    """
    Add or update one validated solved incident in ChromaDB.

    This should be called only after the technician confirms
    that the solution works and the incident is saved as resolved in MariaDB.
    """

    if incident is None:
        return False

    if incident.id is None:
        return False

    if incident.statut != "resolu":
        return False

    document = format_incident_for_chroma(incident, db)

    metadata = {
        "incident_id": incident.id,
        "statut": incident.statut or "",
        "type_probleme": incident.type_probleme or "",
        "equipement": incident.equipement or "",
        "validated": "true"
    }

    collection.upsert(
        documents=[document],
        ids=[str(incident.id)],
        metadatas=[metadata]
    )

    return True


def extract_important_terms(text: str):
    """
    Extract important technical keywords.
    Stopwords and generic words like test/probleme/erreur/failed are ignored.
    """
    if not text:
        return set()

    # Clean text: replace punctuation with space, lowercase, and split
    import re
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', text.lower())
    words = cleaned.split()

    stopwords = {
        # English stopwords
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
        "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
        "with", "about", "against", "between", "into", "through", "during", "before", "after", 
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", 
        "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "will", "just", "don", 
        "should", "now",
        # French stopwords
        "de", "la", "le", "les", "et", "un", "une", "des", "du", "en", "pour", "dans", "par", 
        "sur", "avec", "est", "a", "qui", "que", "ne", "pas", "ce", "ces", "dans", "en", "par", 
        "pour", "qui", "que", "quoi", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", 
        "ton", "tu", "un", "une", "vos", "votre", "vous", "y", "ou", "mais", "donc", "or", "ni", "car",
        "au", "aux", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", 
        "votre", "vos", "leur", "leurs", "cette", "cet", "cettes", "dont",
        # Generic terms to ignore
        "probleme", "problem", "erreur", "error", "test", "incident", "solution", "help", 
        "aide", "avertissement", "warning", "fails", "fail", "failed", "connecter", "connecting",
        "connection", "connexion", "connect", "connected", "t", "d", "l", "s", "c", "m",
        "niveau", "panne", "dysfonctionnement", "defaut", "défaillance", "défaut", "status", 
        "statut", "titre", "description", "type", "equipement"
    }

    terms = set()
    for w in words:
        w = w.strip(".-")
        if len(w) >= 2 and w not in stopwords:
            terms.add(w)
            
    return terms


def _execute_vector_search(query: str, n_results: int, max_distance: float) -> list:
    """
    Core vector query execution logic against ChromaDB and MariaDB.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    incident_ids = results["ids"][0]
    distances = results["distances"][0]

    db: Session = SessionLocal()
    response = []

    try:
        for incident_id, distance in zip(incident_ids, distances):
            if distance > max_distance:
                continue

            incident = db.query(Incident).filter(
                Incident.id == int(incident_id)
            ).first()

            if not incident:
                continue

            similarity_score = 1 / (1 + distance)

            response.append({
                "id": str(incident.id),
                "titre": incident.titre,
                "description": incident.description,
                "equipement": incident.equipement,
                "type_probleme": incident.type_probleme,
                "cause": incident.cause,
                "solution": incident.solution,
                "distance": float(distance),
                "similarity_score": similarity_score
            })
    finally:
        db.close()

    return response


def search_similar_incidents(query: str, n_results: int = 10):
    """
    Search similar incidents in ChromaDB with multi-tier fallback:
    Tier 1: strict semantic match (distance <= 1.2, top 10)
    Tier 2: relaxed semantic match (distance <= 1.4, top 20)
    Tier 3: keyword-augmented search (distance <= 1.6, top 20)
    """
    # Tier 1: Strict semantic match
    response = _execute_vector_search(query, n_results=n_results, max_distance=1.2)
    if response:
        print(f"RAG RECALL | tier=1 | candidates={len(response)}")
        return response

    print("RAG RECALL | tier=1 | candidates=0")

    # Tier 2: Relaxed semantic match
    response = _execute_vector_search(query, n_results=20, max_distance=1.4)
    if response:
        print(f"RAG RECALL | tier=2 | candidates={len(response)}")
        return response

    print("RAG RECALL | tier=2 | candidates=0")

    # Tier 3: Keyword-augmented search
    keywords = extract_important_terms(query)
    if keywords:
        enhanced_query = query + " " + " ".join(keywords)
        response = _execute_vector_search(enhanced_query, n_results=20, max_distance=1.6)
        if response:
            print(f"RAG RECALL | tier=3 | candidates={len(response)}")
            return response

    print("RAG RECALL | tier=3 | candidates=0")
    print("WARNING - RAG: all fallback tiers failed for query:", query)
    return []


def generate_smart_response(query: str, n_results: int = 3):
    """
    Generate a structured summary from similar incidents.
    """

    incidents = search_similar_incidents(query, n_results)

    causes = []
    solutions = []
    types = []

    for inc in incidents:
        if inc["type_probleme"]:
            types.append(inc["type_probleme"])

        if inc["cause_probable"]:
            causes.append(inc["cause_probable"])

        if inc["solution_proposee"]:
            solutions.append(inc["solution_proposee"])

    return {
        "probleme_recu": query,
        "type_probable": list(set(types)),
        "causes_probables": list(set(causes)),
        "solutions_recommandees": list(set(solutions)),
        "incidents_similaires": incidents
    }
