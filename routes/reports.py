from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models.models import Rapport
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RapportCreate(BaseModel):
    titre: str
    contenu: str
    type_rapport: Optional[str] = "analyse"
    id_test: Optional[int] = None
    id_incident: Optional[int] = None
    id_user: Optional[int] = None


class RapportUpdate(BaseModel):
    titre: Optional[str] = None
    contenu: Optional[str] = None
    type_rapport: Optional[str] = None
    id_test: Optional[int] = None
    id_incident: Optional[int] = None
    id_user: Optional[int] = None


@router.post("/rapports")
def create_rapport(rapport: RapportCreate, db: Session = Depends(get_db)):
    new_rapport = Rapport(
        titre=rapport.titre,
        contenu=rapport.contenu,
        type_rapport=rapport.type_rapport,
        id_test=rapport.id_test,
        id_incident=rapport.id_incident,
        id_user=rapport.id_user
    )

    db.add(new_rapport)
    db.commit()
    db.refresh(new_rapport)

    return new_rapport


@router.get("/rapports")
def get_rapports(db: Session = Depends(get_db)):
    return db.query(Rapport).all()


@router.get("/rapports/{rapport_id}")
def get_rapport(rapport_id: int, db: Session = Depends(get_db)):
    rapport = db.query(Rapport).filter(Rapport.id == rapport_id).first()

    if not rapport:
        raise HTTPException(status_code=404, detail="Rapport non trouvé")

    return rapport


@router.put("/rapports/{rapport_id}")
def update_rapport(
    rapport_id: int,
    rapport_update: RapportUpdate,
    db: Session = Depends(get_db)
):
    rapport = db.query(Rapport).filter(Rapport.id == rapport_id).first()

    if not rapport:
        raise HTTPException(status_code=404, detail="Rapport non trouvé")

    if rapport_update.titre is not None:
        rapport.titre = rapport_update.titre

    if rapport_update.contenu is not None:
        rapport.contenu = rapport_update.contenu

    if rapport_update.type_rapport is not None:
        rapport.type_rapport = rapport_update.type_rapport

    if rapport_update.id_test is not None:
        rapport.id_test = rapport_update.id_test

    if rapport_update.id_incident is not None:
        rapport.id_incident = rapport_update.id_incident

    if rapport_update.id_user is not None:
        rapport.id_user = rapport_update.id_user

    db.commit()
    db.refresh(rapport)

    return rapport


@router.delete("/rapports/{rapport_id}")
def delete_rapport(rapport_id: int, db: Session = Depends(get_db)):
    rapport = db.query(Rapport).filter(Rapport.id == rapport_id).first()

    if not rapport:
        raise HTTPException(status_code=404, detail="Rapport non trouvé")

    db.delete(rapport)
    db.commit()

    return {"message": "Rapport supprimé avec succès"}
