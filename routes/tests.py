from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models.models import Test
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TestCreate(BaseModel):
    nom_test: str
    equipement: str
    client: Optional[str] = None
    resultat: Optional[str] = "EN_COURS"
    ingenieur: Optional[str] = None
    rapport_ia: Optional[str] = None
    id_user: Optional[int] = None


class TestUpdate(BaseModel):
    nom_test: Optional[str] = None
    equipement: Optional[str] = None
    client: Optional[str] = None
    resultat: Optional[str] = None
    ingenieur: Optional[str] = None
    rapport_ia: Optional[str] = None
    id_user: Optional[int] = None


@router.post("/tests")
def create_test(test: TestCreate, db: Session = Depends(get_db)):
    new_test = Test(
        nom_test=test.nom_test,
        equipement=test.equipement,
        client=test.client,
        resultat=test.resultat,
        ingenieur=test.ingenieur,
        rapport_ia=test.rapport_ia,
        id_user=test.id_user
    )

    db.add(new_test)
    db.commit()
    db.refresh(new_test)

    return new_test


@router.get("/tests")
def get_tests(db: Session = Depends(get_db)):
    return db.query(Test).all()


@router.get("/tests/{test_id}")
def get_test(test_id: int, db: Session = Depends(get_db)):
    test = db.query(Test).filter(Test.id == test_id).first()

    if not test:
        raise HTTPException(status_code=404, detail="Test non trouvé")

    return test


@router.put("/tests/{test_id}")
def update_test(test_id: int, test_update: TestUpdate, db: Session = Depends(get_db)):
    test = db.query(Test).filter(Test.id == test_id).first()

    if not test:
        raise HTTPException(status_code=404, detail="Test non trouvé")

    if test_update.nom_test is not None:
        test.nom_test = test_update.nom_test

    if test_update.equipement is not None:
        test.equipement = test_update.equipement

    if test_update.client is not None:
        test.client = test_update.client

    if test_update.resultat is not None:
        test.resultat = test_update.resultat

    if test_update.ingenieur is not None:
        test.ingenieur = test_update.ingenieur

    if test_update.rapport_ia is not None:
        test.rapport_ia = test_update.rapport_ia

    if test_update.id_user is not None:
        test.id_user = test_update.id_user

    db.commit()
    db.refresh(test)

    return test


@router.delete("/tests/{test_id}")
def delete_test(test_id: int, db: Session = Depends(get_db)):
    test = db.query(Test).filter(Test.id == test_id).first()

    if not test:
        raise HTTPException(status_code=404, detail="Test non trouvé")

    db.delete(test)
    db.commit()

    return {"message": "Test supprimé avec succès"}
