from fastapi import APIRouter
from pydantic import BaseModel
from rag.rag_service import load_incidents_to_chroma, search_similar_incidents, generate_smart_response


router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


@router.post("/rag/index")
def index_incidents():
    count = load_incidents_to_chroma()
    return {
        "message": "Incidents indexés dans ChromaDB avec succès",
        "nombre_incidents": count
    }


@router.post("/rag/search")
def search_incidents(request: SearchRequest):
    results = search_similar_incidents(
        query=request.query,
        n_results=request.n_results
    )

    return {
        "query": request.query,
        "resultats": results
    }

@router.post("/rag/assistant")
def rag_assistant(request: SearchRequest):
    response = generate_smart_response(
        query=request.query,
        n_results=request.n_results
    )

    return response
