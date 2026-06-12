from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal
from models.models import AssistantConversation, AssistantMessage, Incident, Solution, Category
from rag.rag_service import search_similar_incidents, add_solved_incident_to_chroma
from services.openclaw_service import ask_openclaw
import json
import re
from datetime import datetime

router = APIRouter()
DEBUG_MODE = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChatRequest(BaseModel):
    user_id: str
    message: str


# ---------------------------------------------------------
# LANGUAGE DETECTION
# ---------------------------------------------------------

def detect_language_local(text: str) -> str:
    """
    Simple local language detection.
    Returns: fr, en, ar, darija
    """

    msg = text.lower().strip()

    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)

    darija_arabic_markers = [
        "دابا", "واش", "مزيان", "خدام", "بزاف", "ديال",
        "شنو", "كيفاش", "علاش", "صافي", "راه", "حيت",
        "باقي", "ما خدمش", "مازال"
    ]

    if has_arabic:
        if any(word in msg for word in darija_arabic_markers):
            return "darija"
        return "ar"

    darija_latin_markers = [
        "safi", "daba", "mzyan", "kay", "dyal", "mlli",
        "wach", "bghit", "kifach", "hiya", "daz", "rah",
        "chno", "3lach", "hit", "kayn", "makaynch",
        "baqi", "ba9i", "mazal", "makhdamch", "khdam"
    ]

    if any(word in msg for word in darija_latin_markers):
        return "darija"

    words = (
        msg.replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("'", " ")
        .replace("’", " ")
        .split()
    )

    french_markers = [
        "le", "la", "les", "un", "une", "des",
        "donne", "pendant", "maintenant", "apres", "après",
        "demarrage", "démarrage", "coupure", "alimentation",
        "tension", "chute", "capteur", "moteur", "banc",
        "erreur", "problème", "probleme", "marche", "pas",
        "j", "ai", "c", "bon", "ça", "ca", "est"
    ]

    english_markers = [
        "the", "motor", "gives", "give",
        "fixed", "passes", "passed", "after", "enabling", "enable",
        "between", "working", "solved", "issue",
        "problem", "error", "now", "still", "not",
        "does", "work", "failed", "failure",
        "during", "startup", "supply"
    ]

    french_score = sum(1 for word in words if word in french_markers)
    english_score = sum(1 for word in words if word in english_markers)

    # Strong French indicators
    if (
        " le " in f" {msg} "
        or " la " in f" {msg} "
        or " une " in f" {msg} "
        or " donne " in f" {msg} "
        or " pendant " in f" {msg} "
        or " alimentation " in f" {msg} "
        or " démarrage" in msg
        or " demarrage" in msg
    ):
        if french_score >= english_score:
            return "fr"

    if english_score >= 2 and english_score > french_score:
        return "en"

    return "fr"



def language_name_for_prompt(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "English"
    if lang == "ar":
        return "Arabic"
    if lang == "darija":
        return "Moroccan Darija"

    return "French"


def empty_message_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "Empty message."
    if lang == "ar":
        return "الرسالة فارغة."
    if lang == "darija":
        return "الرسالة خاوية."

    return "Message vide."


def no_active_conversation_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "No active conversation. Send a clear new technical problem to start a diagnostic."
    if lang == "ar":
        return "لا توجد محادثة نشطة. أرسل مشكلا تقنيا واضحا لبدء التشخيص."
    if lang == "darija":
        return "ما كايناش محادثة نشطة. صيفط مشكل تقني واضح باش نبداو التشخيص."

    return "Aucune conversation active. Envoyez un nouveau problème technique clair pour démarrer le diagnostic."


def unclear_problem_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "I did not detect a clear technical problem. Describe the test, the error, the equipment, and the symptom."
    if lang == "ar":
        return "لم أكتشف مشكلا تقنيا واضحا. اذكر الاختبار، الخطأ، الجهاز، والأعراض."
    if lang == "darija":
        return "ما بانليش مشكل تقني واضح. شرح ليا التست، الخطأ، الجهاز، وشنو كايوقع."

    return "Je n'ai pas détecté un problème technique clair. Décrivez le test, l'erreur, l'équipement et le symptôme."


def solved_response_by_language(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return (
            "Problem marked as resolved. "
            "The validated cause and solution have been saved in the knowledge base."
        )

    if lang == "ar":
        return "تم تحديد المشكل كمحلول، وتم حفظ السبب والحل في قاعدة المعرفة."

    if lang == "darija":
        return "تسجل المشكل كمحلول، وتخزن السبب والحل فقاعدة المعرفة."

    return (
        "Problème marqué comme résolu. "
        "La cause et la solution validée ont été sauvegardées dans la base de connaissances."
    )

def build_clean_solution_text(incident, cause_text: str, solution_text: str, summary_text: str, history: str) -> str:
    """
    Build a clean structured solution for MariaDB and ChromaDB.
    This improves future RAG quality.
    """

    problem = incident.description if incident and incident.description else "Problème technique signalé par le technicien."

    full_context = (history or "").lower()
    problem_lower = (problem or "").lower()
    cause_lower = (cause_text or "").lower()
    solution_lower = (solution_text or "").lower()

    symptoms = []

    is_can_case = (
        "can" in problem_lower
        or "can" in cause_lower
        or "can" in solution_lower
    )

    is_lin_case = (
        "lin" in problem_lower
        or "lin" in cause_lower
        or "lin" in solution_lower
    )

    is_power_case = (
        "alimentation" in problem_lower
        or "power" in problem_lower
        or "tension" in problem_lower
        or "alimentation" in cause_lower
        or "power" in cause_lower
        or "tension" in cause_lower
        or "alimentation" in solution_lower
        or "power" in solution_lower
        or "tension" in solution_lower
    )

    # CAN symptoms
    if is_can_case and "timeout" in full_context:
        symptoms.append("Timeout de communication CAN observé pendant le test.")

    if is_can_case and ("120" in full_context or "120 ohm" in full_context):
        symptoms.append("Résistance CAN-H / CAN-L mesurée à environ 120 ohm.")

    if is_can_case and ("60" in full_context or "60 ohm" in full_context):
        symptoms.append("Après correction, résistance CAN-H / CAN-L mesurée à environ 60 ohm.")

    # LIN symptoms
    if is_lin_case:
        symptoms.append("Erreur de communication LIN observée pendant le test.")

    if is_lin_case and ("aucune trame" in full_context or "aucune trame lin" in full_context):
        symptoms.append("Aucune trame LIN reçue malgré l’alimentation du capteur.")

    if is_lin_case and ("12v" in full_context or "12 v" in full_context):
        symptoms.append("Capteur alimenté en 12V.")

    if is_lin_case and (
        "mal branche" in full_context
        or "mal branché" in full_context
        or "connecteur" in full_context
        or "fil lin" in full_context
        or "cablage lin" in full_context
        or "câblage lin" in full_context
    ):
        symptoms.append("Mauvais branchement détecté au niveau du connecteur ou du fil de communication.")

    # Power symptoms
    if is_power_case and (
        "chute de tension" in full_context
        or "tension chute" in full_context
        or "tension instable" in full_context
        or "alimentation instable" in full_context
        or "chute de 12v" in full_context
        or "chute de 12 v" in full_context
        or "12v à 9v" in full_context
        or "12v a 9v" in full_context
        or "12 v à 9 v" in full_context
        or "12 v a 9 v" in full_context
    ):
        symptoms.append("Tension d’alimentation instable observée pendant le test.")

    if not symptoms:
        symptoms.append("Symptômes décrits dans la conversation de diagnostic.")

    symptoms_text = "\n".join([f"- {s}" for s in symptoms])

    return f"""
Problème:
{problem}

Symptômes observés:
{symptoms_text}

Cause validée:
{cause_text or "Cause déduite à partir de la conversation technique."}

Solution appliquée:
{solution_text or "Solution validée par le technicien."}

Résultat:
{summary_text or "Le technicien a confirmé que le problème est résolu."}
""".strip()


def improve_cause_solution_from_history(cause_text: str, solution_text: str, summary_text: str, history: str, user_message: str):
    """
    Improve generic cause/solution using the conversation history.
    This keeps resolution local without calling OpenClaw.
    """

    full_context = (history + "\n" + user_message).lower()

    # LIN wiring case
    if (
        "lin" in full_context
        and (
            "mal branche" in full_context
            or "mal branché" in full_context
            or "mauvais branchement" in full_context
            or "cablage" in full_context
            or "câblage" in full_context
            or "connecteur" in full_context
            or "fil lin" in full_context
        )
    ):
        return {
            "cause": "Erreur de communication LIN causée par un mauvais branchement du fil LIN ou du connecteur.",
            "solution": "Correction du câblage LIN et remise du fil LIN sur le bon connecteur/pin.",
            "summary": "Le test capteur pression passe après correction du câblage LIN."
        }

    # Power supply instability case
    if (
        ("alimentation" in full_context or "power" in full_context)
        and (
            "chute" in full_context
            or "instable" in full_context
            or "9v" in full_context
            or "12v" in full_context
            or "tension" in full_context
        )
    ):
        return {
            "cause": "Alimentation instable provoquant une chute de tension pendant le test.",
            "solution": "Remplacement ou correction de l’alimentation afin de stabiliser la tension à 12V.",
            "summary": "Le test passe après stabilisation ou remplacement de l’alimentation."
        }

    # CAN termination case
    if (
        "can" in full_context
        and ("120" in full_context or "120 ohm" in full_context)
        and ("60" in full_context or "60 ohm" in full_context)
        and ("terminaison" in full_context or "termination" in full_context)
    ):
        return {
            "cause": (
                "Le timeout CAN était causé par une terminaison CAN manquante ou incorrecte. "
                "La résistance mesurée entre CAN-H et CAN-L était de 120 ohm au lieu d'environ 60 ohm."
            ),
            "solution": (
                "Activation de la terminaison CAN côté banc/interface afin d'obtenir environ 60 ohm "
                "entre CAN-H et CAN-L."
            ),
            "summary": "Le test moteur passe après activation de la terminaison CAN."
        }

    return {
        "cause": cause_text,
        "solution": solution_text,
        "summary": summary_text
    }


# ---------------------------------------------------------
# LOCAL RESOLUTION DETECTION
# No OpenClaw call here.
# ---------------------------------------------------------

def detect_resolution_intent(user_message: str, history: str):
    msg = user_message.lower().strip()
    full_context = (history + "\n" + user_message).lower()

    negative_phrases = [
        # English
        "not solved", "not fixed", "not working",
        "still timeout", "still error", "still problem",
        "does not work", "doesn't work", "not work",
        "same problem", "same error",

        # French
        "pas résolu", "pas resolu", "pas réglé", "pas regle",
        "ne marche pas", "ça ne marche pas", "ca ne marche pas",
        "encore timeout", "toujours timeout",
        "toujours erreur", "toujours problème", "toujours probleme",
        "ça marche pas", "ca marche pas",

        # Arabic / Darija Arabic
        "مازال", "ما زال", "ما خدامش", "ما خدمش",
        "ما تصلحش", "باقي", "باقي نفس المشكل",

        # Darija Latin
        "makhdamch", "ma khdamch", "mazal", "baqi", "ba9i",
        "baqi timeout", "ba9i timeout"
    ]

    new_problem_phrases = [
        # English
        "but now", "another problem", "another issue", "new problem",

        # French
        "mais maintenant", "autre problème", "autre probleme",
        "nouveau problème", "nouveau probleme",

        # Arabic / Darija
        "دابا عندي", "ولكن دابا", "مشكل آخر"
    ]

    resolution_triggers = [
        # French
        "resolu", "résolu", "c bon", "c'est bon",
        "ca marche", "ça marche", "test ok", "test passe",
        "test pass", "le test passe", "le test pass",
        "probleme regle", "problème réglé",
        "probleme corrige", "problème corrigé",
        "plus de timeout", "plus d'erreur", "plus erreur",
        "tout est ok", "tout est normal",
        "ca fonctionne", "ça fonctionne",

        # English
        "solved", "fixed", "problem solved", "issue solved",
        "it works", "working now", "no timeout", "timeout gone",
        "test passed", "the test passes", "the test passed",

        # Arabic / Darija
        "المشكل تحل", "تحل المشكل", "خدام دابا", "دابا خدام",

        # Darija Latin
        "safi", "daz mzyan", "khdam daba", "mzyan daba"
    ]

    if any(phrase in msg for phrase in negative_phrases):
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    if any(phrase in msg for phrase in new_problem_phrases):
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    if any(trigger in msg for trigger in resolution_triggers):

        # Specific CAN termination case
        if (
            "can" in full_context
            and ("120" in full_context or "120 ohm" in full_context)
            and ("60" in full_context or "60 ohm" in full_context)
            and ("terminaison" in full_context or "termination" in full_context)
        ):
            return {
                "is_resolved": True,
                "cause": (
                    "Le timeout CAN était causé par une terminaison CAN manquante ou incorrecte. "
                    "La résistance mesurée entre CAN-H et CAN-L était de 120 ohm au lieu d'environ 60 ohm, "
                    "ce qui indique qu'une seule terminaison était présente sur le bus."
                ),
                "solution": (
                    "Activation de la terminaison CAN côté banc/interface afin d'obtenir environ 60 ohm "
                    "entre CAN-H et CAN-L. Après correction, le test moteur est passé correctement."
                ),
                "summary": (
                    "Le problème de timeout CAN a été résolu après activation de la terminaison CAN."
                )
            }

        return {
            "is_resolved": True,
            "cause": "Cause déduite à partir de la conversation technique.",
            "solution": user_message,
            "summary": "Le technicien indique que le problème est résolu."
        }

    return {
        "is_resolved": False,
        "cause": "",
        "solution": "",
        "summary": ""
    }

def suggest_category_name(text: str) -> tuple[str, str]:
    """
    Suggest a general category name from the incident text.
    The goal is to create reusable categories, not one category per incident.
    """

    msg = text.lower()

    if "can" in msg:
        return (
            "Communication CAN",
            "Incidents liés au bus CAN, timeout CAN, trames CAN, câblage CAN, terminaison CAN ou configuration CAN."
        )

    if "lin" in msg:
        return (
            "Communication LIN",
            "Incidents liés au bus LIN, trames LIN, câblage LIN, capteurs LIN ou erreurs de communication LIN."
        )

    if (
        "alimentation" in msg
        or "tension" in msg
        or "12v" in msg
        or "9v" in msg
        or "power" in msg
        or "voltage" in msg
    ):
        return (
            "Alimentation",
            "Incidents liés à l'alimentation électrique, tension instable, chute de tension, courant ou bloc d'alimentation."
        )

    if "capteur" in msg or "sensor" in msg:
        return (
            "Capteurs",
            "Incidents liés aux capteurs, mesures, signaux capteurs ou comportement anormal d'un capteur."
        )

    if "ecu" in msg or "flash" in msg or "calibration" in msg or "firmware" in msg:
        return (
            "ECU / Flash / Calibration",
            "Incidents liés à l'ECU, au flash, au firmware, aux fichiers de calibration ou à la configuration calculateur."
        )

    if "rapport" in msg or "report" in msg or "pdf" in msg:
        return (
            "Rapports",
            "Incidents liés à la génération, l'export ou la consultation des rapports."
        )

    if "database" in msg or "mariadb" in msg or "mysql" in msg or "base de données" in msg:
        return (
            "Base de données",
            "Incidents liés à la base de données, requêtes SQL, connexion ou stockage."
        )

    if "banc" in msg or "bench" in msg:
        return (
            "Banc de test",
            "Incidents liés au banc de test, configuration banc, interface ou équipement de test."
        )

    return (
        "Autre",
        "Incidents techniques ne correspondant pas encore à une catégorie spécialisée."
    )


def get_or_create_category(db: Session, text: str):
    """
    Find an existing category or create a new one if needed.
    """

    category_name, description = suggest_category_name(text)

    category = db.query(Category).filter(
        Category.nom == category_name
    ).first()

    if category is None:
        category = Category(
            nom=category_name,
            description=description
        )
        db.add(category)
        db.commit()
        db.refresh(category)

    return category

def extract_json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def generate_validated_knowledge_with_openclaw(
    initial_problem: str,
    user_history: str,
    current_message: str
):
    """
    Ask OpenClaw to generate clean structured knowledge after the technician confirms the solution works.
    This is called only after local resolution detection.
    """

    prompt = f"""
You are a technical knowledge extraction assistant for an industrial test incident management system.

The technician has confirmed that the problem is solved.

Your task:
Analyze ONLY the technician's messages and extract the validated knowledge that should be saved in MariaDB and ChromaDB.

Do not invent facts.
Do not use assistant suggestions unless the technician confirmed them.
Use the technician's observations, measurements, actions, and final confirmation.

Initial problem:
{initial_problem}

Technician conversation history:
{user_history}

Final technician message:
{current_message}

Return ONLY valid JSON, no markdown, no explanation.

JSON format:
{{
  "category_name": "short general category name",
  "category_description": "short category description",
  "cause": "validated technical cause",
  "solution": "validated solution applied",
  "symptoms": [
    "observed symptom 1",
    "observed symptom 2"
  ],
  "result": "final confirmed result",
  "structured_solution": "clean human-readable structured solution",
  "chroma_document": "clean searchable document for future RAG"
}}

Rules:
- category_name must be general, not too specific.
  Good examples:
  "Communication CAN", "Communication LIN", "Alimentation", "Pneumatique / Étanchéité", "Température / Refroidissement", "Câblage / Connectique", "Capteurs", "Banc de test", "Logiciel".
- If the problem is about temperature, cooling, fan, overheating, use "Température / Refroidissement".
- If the problem is about pressure leak, sealing, pneumatic tube, use "Pneumatique / Étanchéité".
- If the problem is about voltage drop, 12V, 9V, power supply, use "Alimentation".
- If the problem is about CAN, CAN-H, CAN-L, termination, use "Communication CAN".
- If the problem is about LIN, LIN frame, LIN wire, use "Communication LIN".
- structured_solution must include:
  Problème:
  Symptômes observés:
  Cause validée:
  Solution appliquée:
  Résultat:
- chroma_document must be concise but complete for future similarity search.
"""

    result = ask_openclaw(
        session_id="knowledge_extractor",
        prompt=prompt
    )

    if not result.get("success"):
        return None

    data = extract_json_from_text(result.get("response", ""))

    if not data:
        return None

    return data

def get_or_create_category_by_name(db: Session, category_name: str, category_description: str = ""):
    """
    Get an existing category by name or create it if it does not exist.
    Used after OpenClaw extracts the validated category.
    """

    if not category_name:
        category_name = "Autre"

    category_name = category_name.strip()

    category = db.query(Category).filter(
        Category.nom == category_name
    ).first()

    if category is None:
        category = Category(
            nom=category_name,
            description=category_description or "Catégorie créée automatiquement par l'assistant."
        )
        db.add(category)
        db.commit()
        db.refresh(category)

    return category


def normalize_id(x):
    return str(x).strip()


def compute_confidence_threshold(conf_list):
    if not conf_list:
        return 0.6
    avg = sum(conf_list) / len(conf_list)
    return max(0.55, min(0.75, avg))


def filter_rag_with_openai(user_problem: str, candidates: list) -> tuple:
    """
    Use OpenAI (via ask_openclaw) as a binary relevance classifier with confidence scoring.
    Decides relevant (true/false) and confidence (0.0 to 1.0) for each candidate.
    Returns (top_candidates, confidence_values).
    """
    if not candidates:
        return [], []

    def safe_float(v):
        try:
            return float(v)
        except:
            return 0.0

    # Sort candidates by distance and take top 10 (with min check for length safety)
    top_candidates = sorted(candidates, key=lambda x: x.get("distance", 999.0))[:min(10, len(candidates))]

    candidates_text = ""
    for case in top_candidates:
        candidates_text += f"""
Candidate ID: {case.get('id')}
Title: {case.get('titre') or 'unknown'}
Problem Description: {case.get('description') or 'unknown'}
Equipment: {case.get('equipement') or 'unknown'}
Technical Domain: {case.get('type_probleme') or 'unknown'}
Solution: {case.get('solution') or 'unknown'}
"""

    prompt = f"""
You are a RAG relevance evaluator for technical incident resolution.

You will receive:
- A user query describing a technical issue
- A list of candidate incident cases

TASK:
For each candidate, determine:
1. Is it relevant to the user issue?
2. Assign a confidence score (0.0 to 1.0)

USER QUERY:
{user_problem}

CANDIDATES:
{candidates_text}

RULES:

You must focus on:
- equipment family alignment (e.g., PLC/industrial control, TV/display, barcode scanner/peripheral, motor/conveyor, network infrastructure/routers. Note: these are examples only, not a complete list; use the same reasoning for any equipment type.)
- system/component similarity (WiFi, PLC, motor, power supply, HMI, network, authentication)
- failure mode similarity (timeout, disconnect, overheating, authentication failure, voltage drop)
- root-cause similarity (not wording)

IMPORTANT SEMANTIC & CONTEXTUAL RULES:
- Treat technical equivalents as relevant:
  * "authentication failed" ≈ "incorrect password" ≈ "wrong credentials"
  * "no internet" ≈ "WAN failure" ≈ "router issue"
  * "disconnects randomly" ≈ "network instability"
- Equipment Type Constraint: Different equipment families should lower confidence even if generic symptoms overlap. A barcode scanner is different from a TV; a PLC is different from a conveyor motor.
- Troubleshooting Relevance: A candidate is relevant only if its validated cause could realistically help diagnose the current issue. For example, a thermal overload trip cause has no troubleshooting value for a PLC communication drop.
- Generic Terms Rule: Words like "network", "connection", "communication", "disconnect", "device", or "error" must not automatically create a high-confidence match. You must look for equipment family, subsystem, and failure mode alignment first.

RELEVANCE EXAMPLES (GENERAL PATTERNS):

HIGH RELEVANCE (0.8–1.0)
- Same equipment family
- Same subsystem
- Same failure mode
- Similar validated cause
Example:
User issue: authentication failure
Candidate: incorrect credentials
Reason: same subsystem, same failure mode, same diagnostic path

MEDIUM RELEVANCE (0.6)
- Same equipment family
- Same subsystem
- Different wording or symptoms
- Candidate's validated cause could realistically help diagnose the issue
Example:
User issue: intermittent connection loss
Candidate: communication instability caused by network configuration
Reason: similar troubleshooting path

LOW RELEVANCE (0.3)
- Partial subsystem overlap only
- Different equipment family or different operational context
- Candidate may provide limited diagnostic value
Example:
User issue and candidate both involve communications or networking, but affect different equipment categories and likely require different troubleshooting procedures.

NOT RELEVANT (0.0)
- Different equipment family
- Different subsystem
- Different failure mode
- Candidate's validated cause would not realistically help diagnose the issue
Example:
A communication problem compared to a thermal protection problem.

SCORING RULES:
- 1.0 -> Exact same root cause, same equipment family, and same subsystem.
- 0.8 -> Same equipment family + same failure mode (different wording).
- 0.6 -> Meaningful troubleshooting relevance (similar diagnostic path or shared subsystem within the same equipment family/context).
- 0.3 -> Weak subsystem overlap but different equipment family (e.g., PLC communication issue vs. TV WiFi issue, or Barcode scanner vs. TV WiFi issue).
- 0.0 -> Completely unrelated subsystem AND unrelated equipment family (e.g., PLC communication issue vs. conveyor motor thermal overload).

HARD RULES:
- Every candidate MUST be evaluated (never skip any)
- Never omit IDs
- Never return empty evaluations
- If a candidate shares subsystem AND has plausible troubleshooting relevance, minimum score is 0.3. A shared subsystem alone is not enough. A shared failure mode alone is not enough. Use 0.0 when the candidate's validated cause would not realistically help diagnose the current issue, even if generic terms overlap.
- Shared generic terms such as "communication", "connection", "disconnect", "timeout", "network", or "error" are not sufficient by themselves to establish troubleshooting relevance or force a 0.3 minimum.
- If any candidate shares subsystem or failure mode, at least one candidate should usually have confidence >= 0.3 unless all candidates are clearly unrelated.
- If multiple candidates are similar, distribute confidence scores relatively (do not collapse all to identical values unless truly identical).
- If one candidate is clearly stronger than others, it should have significantly higher confidence than the rest.

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "evaluations": [
    {{
      "id": "string",
      "relevant": true,
      "confidence": 0.0
    }}
  ]
}}
"""
    try:
        # Pre-initialize safety defaults
        for case in top_candidates:
            case.setdefault("relevant", False)
            case.setdefault("confidence", 0.0)

        result = ask_openclaw(
            session_id="rag_filter",
            prompt=prompt
        )
        response_text = result.get("response", "")
        data = extract_json_from_text(response_text)

        items = []
        if isinstance(data, dict):
            items = (
                data.get("evaluations")
                or data.get("results")
                or data.get("items")
                or []
            )

        relevance_map = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            if "id" not in item or "relevant" not in item:
                continue
            relevance_map[str(item.get("id")).strip()] = item

        # Apply relevance annotation and track confidence list
        confidence_values = []
        for case in top_candidates:
            case_id = str(case.get("id")).strip()
            r = relevance_map.get(case_id)
            if r:
                case["relevant"] = bool(r.get("relevant", False))
                confidence = safe_float(r.get("confidence"))
                case["confidence"] = confidence
            else:
                case["relevant"] = False
                confidence = 0.0
                case["confidence"] = confidence
            confidence_values.append(confidence)

        return top_candidates, confidence_values

    except Exception as e:
        print("LLM FILTERING GATE ERROR:", e)
        return [], []  # Strict safety fallback: return empty list if API fails


def detect_user_intent_with_openclaw(
    current_message: str,
    conversation_history: str = ""
):
    """
    AI intent classifier.
    OpenClaw decides what the user wants.
    """

    prompt = f"""
You are an intent classifier.

Conversation history:
{conversation_history}

Latest user message:
{current_message}

Choose ONLY one intent from:

- continue
- resolved
- cancel
- new_problem

Definitions:

continue:
The user is continuing the current diagnosis.
IMPORTANT: You cannot choose 'continue' if the conversation history is empty. If the history is empty or the last conversation was cancelled/closed, choose 'new_problem' instead (unless the user explicitly says resolved or cancel).

resolved:
The user confirms that the original problem is fixed, solved, working correctly, disappeared, or can reasonably be closed.

Examples:
- it is solved now
- solved
- fixed
- it works now
- problem solved
- issue solved
- test passed
- c'est bon
- ça marche
- résolu
- safi
- khdam daba

cancel:
The user wants to stop, abandon, ignore, or cancel the current diagnosis.

Examples:
- cancel
- stop
- forget this
- ignore this
- never mind
- annuler
- laisse tomber

new_problem:
The user introduces a different problem that requires a different diagnosis.

Examples:
- another issue
- new problem
- but now I have a LIN timeout
- different problem
- now my Wi-Fi does not work
- now I have another error

Resolution reasoning rules:

Do NOT classify an incident as resolved simply because the user answered with short confirmations such as:

- yes
- ok
- done
- tested
- measured
- 2.5V
- 60 ohms

These are only examples.

Such replies are often:

- diagnostic measurements
- test results
- observations
- acknowledgements
- answers to a troubleshooting question

Do not rely on keywords alone.

Always analyze:

1. the user's latest message
2. the previous assistant question
3. the recent conversation history
4. the original problem being diagnosed
5. whether the issue itself has actually disappeared

Examples:

Assistant:
"Did you measure CAN_H and CAN_L?"
User:
"yes"
Intent:
continue

Assistant:
"What voltage do you measure?"
User:
"2.5V"
Intent:
continue

Assistant:
"What resistance do you measure?"
User:
"60 ohms"
Intent:
continue

Assistant:
"After replacing the transceiver, is the CAN timeout gone?"
User:
"yes"
Intent:
resolved

Assistant:
"Does the taskbar work now after restarting Explorer?"
User:
"yes"
Intent:
resolved

Assistant:
"Can you see the TV now?"
User:
"yes"
Intent:
continue

Assistant:
"Can you now connect successfully to the TV?"
User:
"yes"
Intent:
resolved

Assistant:
"Did the proposed fix solve the issue?"
User:
"yes"
Intent:
resolved

Assistant:
"Did you perform the test?"
User:
"yes"
Intent:
continue

These examples are not exhaustive.

Use reasoning and context, not keyword matching.

Only classify as resolved when there is sufficient evidence that:

- the original issue disappeared
- the proposed fix solved the issue
- the user clearly confirms the issue is fixed
- the conversation strongly indicates the incident can be closed

Otherwise choose continue.

New problem reasoning rules:

Choose "new_problem" only when the user introduces a different issue that requires a different diagnosis.

Examples:

Current incident:
CAN timeout

User:
"Now I have a LIN timeout."
Intent:
new_problem

Current incident:
Taskbar frozen

User:
"My Wi-Fi no longer works."
Intent:
new_problem

Current incident:
CAN timeout

User:
"I measured 60 ohms."
Intent:
continue

Current incident:
Taskbar frozen

User:
"yes"
Intent:
continue or resolved depending on the previous assistant question.

Do not classify as new_problem simply because the user provides additional information about the current issue.

Cancel reasoning rules:

Choose "cancel" only when the user explicitly wants to stop the current diagnosis.

Examples:

- cancel
- stop
- forget this issue
- never mind
- leave it
- annuler
- laisse tomber

Do not classify as cancel simply because:

- the user does not know the answer
- the user cannot perform a test
- the user asks for clarification
- the user says "I don't know"

These situations are usually continue.

Decision priority:

1. If the conversation history is empty and the user describes a problem -> new_problem
2. Else if the user introduces a different issue -> new_problem
3. Else if the user explicitly wants to stop -> cancel
4. Else if there is strong evidence that the original issue is fixed -> resolved
5. Otherwise -> continue


Special rule:
- If the Conversation history is empty, the intent must be new_problem (unless the user is explicitly saying resolved/cancel/greeting).

When uncertain, choose continue.

Return ONLY valid JSON.

Example:

{{
  "intent": "continue"
}}
"""
    result = ask_openclaw(
        session_id="intent_classifier_v2",
        prompt=prompt
    )

    print("INTENT RAW RESPONSE:")
    print(result.get("response", ""))

    if not result.get("success"):
        return {"intent": "continue"}

    data = extract_json_from_text(
        result.get("response", "")
    )

    if not data:
        return {"intent": "continue"}

    intent = data.get("intent", "continue")

    allowed = [
        "continue",
        "resolved",
        "cancel",
        "new_problem"
    ]

    if intent not in allowed:
        intent = "continue"

    return {
        "intent": intent
    }


# ---------------------------------------------------------
# MAIN CHAT ROUTE
# ---------------------------------------------------------

@router.post("/assistant/chat")
def assistant_chat(request: ChatRequest, db: Session = Depends(get_db)):
    message_text = request.message.strip()

    if not message_text:
        return {
            "success": False,
            "response": empty_message_response(request.message)
        }

    # ---------------------------------------------------------
    # STEP 1: LOAD ACTIVE CONVERSATION (BEFORE INTENT)
    # FIX: Must happen first so resolved/continue have context.
    # ---------------------------------------------------------

    conversation = None
    history = ""
    user_history = ""
    old_messages = []
    similar_cases = []
    raw_cases = []
    filtered_cases = []
    final_cases = []
    confidence_values = []
    threshold = 0.6

    active_conversations = (
        db.query(AssistantConversation)
        .filter(
            AssistantConversation.discord_user_id == request.user_id,
            AssistantConversation.status == "active"
        )
        .order_by(AssistantConversation.updated_at.desc())
        .all()
    )

    if active_conversations:
        conversation = active_conversations[0]

        # Ensure only 1 active conversation per user
        for c in active_conversations[1:]:
            c.status = "closed"
        db.commit()

    # ---------------------------------------------------------
    # STEP 2: BUILD CONVERSATION HISTORY
    # FIX: Needed by intent classifier and diagnostic prompt.
    # ---------------------------------------------------------

    if conversation is not None:
        old_messages = db.query(AssistantMessage).filter(
            AssistantMessage.conversation_id == conversation.id
        ).order_by(AssistantMessage.created_at.asc()).all()

        if not old_messages:
            old_messages = []

        history = "\n\n".join([
            f"{msg.role.upper()} : {msg.message}"
            for msg in old_messages
            if msg.message
        ])

        user_history = "\n\n".join([
            msg.message
            for msg in old_messages
            if msg.role == "user" and msg.message
        ])

    # ---------------------------------------------------------
    # STEP 3: INTENT DETECTION (WITH HISTORY)
    # FIX: Now receives conversation_history so "yes"/"60 ohm"
    # can be classified correctly based on context.
    # ---------------------------------------------------------

    try:
        intent_result = detect_user_intent_with_openclaw(
            current_message=message_text,
            conversation_history=history if conversation else ""
        )
        intent = intent_result.get("intent") if isinstance(intent_result, dict) else intent_result
    except Exception as e:
        print("OPENCLAW INTENT ERROR:", str(e))
        intent = None

    if intent is None:
        intent = "continue"

    # ---------------------------------------------------------
    # STEP 4: ADJUST INTENT BASED ON CONVERSATION STATE
    # ---------------------------------------------------------

    # No active conversation + resolved/cancel = nothing to act on
    if conversation is None and intent in ("resolved", "cancel"):
        return {
            "success": False,
            "response": no_active_conversation_response(message_text)
        }

    # No active conversation + continue = check if real problem
    if conversation is None and intent == "continue":
        msg_lower = message_text.lower().strip()
        not_problem_words = [
            "ok", "oui", "non", "merci", "thanks", "thank you",
            "bonjour", "hello", "hi", "salut",
            "resolu", "résolu", "solved", "fixed",
            "c bon", "c'est bon", "ca marche", "ça marche",
            "test ok", "le test passe",
            "safi", "daba safi"
        ]

        if any(w == msg_lower for w in not_problem_words):
            return {
                "success": False,
                "response": no_active_conversation_response(message_text)
            }

        # Looks like a real technical problem, treat as new_problem
        intent = "new_problem"

    print(f"INTENT={intent} | ACTIVE_CONVERSATION={'YES' if conversation else 'NO'}")

    # ---------------------------------------------------------
    # STEP 5: HANDLE CANCEL
    # Close conversation, save nothing.
    # Next message will create a new incident.
    # ---------------------------------------------------------

    if intent == "cancel":
        conversation.status = "closed"

        user_msg = AssistantMessage(
            conversation_id=conversation.id,
            role="user",
            message=message_text
        )
        db.add(user_msg)
        db.commit()

        lang = detect_language_local(message_text)
        if lang == "en":
            cancel_response = "Request cancelled. No data was saved. Send a new technical problem to start a new diagnostic."
        elif lang == "ar":
            cancel_response = "تم إلغاء الطلب. لم يتم حفظ أي بيانات. أرسل مشكلا تقنيا جديدا لبدء تشخيص جديد."
        elif lang == "darija":
            cancel_response = "تلغا الطلب. ما تسجل والو. صيفط مشكل تقني جديد باش نبداو تشخيص جديد."
        else:
            cancel_response = "Demande annulée. Aucune donnée n'a été sauvegardée. Envoyez un nouveau problème technique pour démarrer un nouveau diagnostic."

        return {
            "success": True,
            "cancelled": True,
            "response": cancel_response
        }

    # ---------------------------------------------------------
    # STEP 6: HANDLE RESOLVED
    # FIX: Conversation is already loaded from Step 1.
    # The old code skipped loading because force_new_incident=True.
    # Now we properly use the loaded conversation to resolve.
    # ---------------------------------------------------------

    if intent == "resolved":
        incident = None

        if conversation.incident_id is not None:
            incident = db.query(Incident).filter(
                Incident.id == conversation.incident_id
            ).first()

        cause_text = ""
        solution_text = message_text
        summary_text = ""

        # Ask AI to extract validated knowledge
        knowledge = generate_validated_knowledge_with_openclaw(
            initial_problem=incident.description if incident else conversation.initial_question,
            user_history=user_history,
            current_message=message_text
        )

        category_from_ai = None
        clean_solution_text = solution_text

        if knowledge:
            cause_text = knowledge.get("cause") or cause_text
            solution_text = knowledge.get("solution") or solution_text
            summary_text = knowledge.get("result") or summary_text
            clean_solution_text = knowledge.get("structured_solution") or solution_text

            category_name = knowledge.get("category_name") or ""
            category_description = knowledge.get("category_description") or ""

            if category_name:
                category_from_ai = get_or_create_category_by_name(
                    db=db,
                    category_name=category_name,
                    category_description=category_description
                )
        else:
            # Fallback: local resolution detection when AI fails
            resolution = detect_resolution_intent(
                user_message=message_text,
                history=history
            )
            cause_text = resolution.get("cause") or "Cause déduite à partir de la conversation technique."
            solution_text = resolution.get("solution") or message_text
            summary_text = resolution.get("summary") or "Le technicien indique que le problème est résolu."

            clean_solution_text = build_clean_solution_text(
                incident=incident,
                cause_text=cause_text,
                solution_text=solution_text,
                summary_text=summary_text,
                history=user_history
            )

        # Update incident
        if incident is not None:
            incident.statut = "resolu"
            incident.cause = cause_text
            incident.solution = clean_solution_text

            if category_from_ai is not None:
                incident.category_id = category_from_ai.id

            solution_record = Solution(
                titre=f"Solution validée - Incident {incident.id}",
                description=clean_solution_text,
                type_probleme=incident.type_probleme,
                equipement=incident.equipement,
                efficacite=1,
                id_incident=incident.id,
                id_user=None
            )
            db.add(solution_record)

        # Close conversation
        conversation.status = "solved"

        # Save messages
        user_msg = AssistantMessage(
            conversation_id=conversation.id,
            role="user",
            message=message_text
        )
        db.add(user_msg)

        assistant_response = solved_response_by_language(message_text)

        assistant_msg = AssistantMessage(
            conversation_id=conversation.id,
            role="assistant",
            message=assistant_response
        )
        db.add(assistant_msg)

        conversation.updated_at = datetime.utcnow()
        db.commit()

        # Add to ChromaDB
        if incident is not None:
            db.refresh(incident)
            try:
                add_solved_incident_to_chroma(incident, db)
                print(f"Incident {incident.id} added/updated in ChromaDB.")
            except Exception as e:
                print("Error adding resolved incident to ChromaDB:", e)

        return {
            "success": True,
            "resolved": True,
            "conversation_id": conversation.id,
            "incident_id": conversation.incident_id,
            "cause": cause_text,
            "solution": clean_solution_text,
            "response": assistant_response
        }

    # ---------------------------------------------------------
    # STEP 7: HANDLE NEW_PROBLEM
    # Close old conversation if any, create new incident.
    # ---------------------------------------------------------

    if intent == "new_problem":
        if conversation is not None:
            conversation.status = "closed"
            db.commit()

        # Close any remaining stale active conversations
        stale = (
            db.query(AssistantConversation)
            .filter(
                AssistantConversation.discord_user_id == request.user_id,
                AssistantConversation.status == "active"
            )
            .all()
        )
        for conv in stale:
            conv.status = "closed"
        db.commit()

        # RAG search for similar solved incidents
        try:
            raw_cases = search_similar_incidents(message_text, n_results=10)
        except Exception as e:
            print("RAG SEARCH ERROR:", e)
            raw_cases = []

        try:
            raw_cases_with_relevance, confidence_values = filter_rag_with_openai(message_text, raw_cases)
            threshold = compute_confidence_threshold(confidence_values)
            filtered_cases = [
                c for c in raw_cases_with_relevance
                if c.get("confidence", 0.0) >= threshold
            ]
        except Exception as e:
            print("OPENAI FILTERING ERROR:", e)
            filtered_cases = []
            threshold = 0.6
            confidence_values = []

        final_cases = sorted(filtered_cases, key=lambda x: (x["distance"], -x.get("confidence", 0.0), str(x["id"])))[:3]
        similar_cases = final_cases

        if final_cases:
            rag_context = "\n\n".join(
                f"Similar case {i + 1}:\n"
                f"Category: {case.get('type_probleme') or 'Inconnu'}\n"
                f"Problem: {case.get('description') or case.get('titre') or 'Inconnu'}\n"
                f"Cause: {case.get('cause') or 'Inconnu'}\n"
                f"Solution: {case.get('solution') or 'Inconnu'}\n"
                f"Result: Resolved"
                for i, case in enumerate(final_cases)
            )
        else:
            rag_context = "No similar internal cases found in the database."

        # Create incident with category as NULL initially.
        # It will be set to a final resolved category upon resolution.
        incident = Incident(
            titre=message_text[:100],
            description=message_text,
            statut="ouvert",
            type_probleme="Assistant AI",
            equipement="Inconnu",
            category_id=None
        )

        db.add(incident)
        db.commit()
        db.refresh(incident)

        # Create conversation
        conversation = AssistantConversation(
            discord_user_id=request.user_id,
            incident_id=incident.id,
            initial_question=message_text,
            rag_context=rag_context,
            status="active"
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        # Reset context for new conversation
        history = ""
        user_history = ""
        old_messages = []

    # ---------------------------------------------------------
    # STEP 8: CONTINUE — Build diagnostic prompt WITH history
    # FIX: History is now included in the prompt so OpenAI
    # has full context for meaningful follow-up responses.
    # ---------------------------------------------------------

    is_follow_up = len(old_messages) > 0
    rag_context = conversation.rag_context or ""

    prompt = f"""
You are an intelligent technical diagnostic assistant for an electronic test management platform.

CRITICAL LANGUAGE RULE:
You must determine the language directly from the user's latest message.
Answer entirely in the same language as the user's latest message.

Very important:
- If the latest user message is in French, answer fully in French.
- If the latest user message is in English, answer fully in English.
- If the latest user message is in Arabic, answer fully in Arabic.
- If the latest user message is in Moroccan Darija, answer in simple Moroccan Darija.
- If internal similar cases are written in another language, translate and reformulate them into the user's language.
- Do not copy the language of the internal case.
- The opening sentence, section titles, step labels, and final question must also be in the user's language.
- Do not write "This matches", "Step", or "Question" if the user wrote in French. Use "J'ai trouvé...", "Étape actuelle :", and "Question :".

Conversation history:
{history}

Technician/engineer latest message:
{message_text}

Similar internal cases found by ChromaDB:
{rag_context}

Conversation state:
Is follow-up message: {is_follow_up}

Internal case relevance rule:
- Candidate incidents returned by RAG are only prioritized suggestions.
- Do NOT assume they are relevant.
- Before using a candidate incident, compare its problem, symptoms, equipment, and technical domain with the current user problem.
- If the candidate belongs to a different technical domain or addresses a different type of issue, ignore it completely.
- Only consider a candidate relevant if there is a clear technical relationship between the problems, symptoms, equipment involved, failure mode, or root cause.
- If all candidates are irrelevant, continue normal troubleshooting and behave as if no internal case was found.
- Never announce that a similar internal case exists unless you have first determined that the similarity is genuine.
- When you decide that a candidate incident is relevant and you use information from it in your answer, you must explicitly acknowledge that you are using a previously resolved internal case.
- If you use a candidate incident to guide your diagnosis, troubleshooting steps, probable cause, or solution, then it is considered relevant.
- Do not mention ChromaDB, the vector database, the RAG pipeline, or the absence of matches. Only mention internal cases as instructed above.
- Continue with normal technical reasoning as an experienced engineer.
- Ask diagnostic questions and propose the next troubleshooting step.

RAG Memory Rule:
- Candidate cases are memory suggestions only. Do not assume they are authoritative.
- Prioritize technical diagnosis and logical reasoning over RAG suggestions if they differ.
- Never override safety rules or manual diagnostics based on retrieved memories.
- Only announce similar internal cases if they are present in the list and genuinely similar.


Conversation behavior:
- If this is a follow-up message, do not repeat whether an internal similar case was found.
- If this is a follow-up message, do not say again "J'ai trouvé un cas interne similaire", "I found a similar case", or "This matches".
- Mention the internal case only in the first answer of a new incident.
- Continue directly from the technician's latest result.
- Do not restart the analysis.
- Do not repeat the same introduction.
- Only give the next useful step and one question.

Core mission:
You are not a simple search bot.
You are an interactive diagnostic assistant for technicians and engineers.

Internal knowledge behavior:
- Always analyze the internal similar cases first.
- If a similar internal case contains a useful solution, treat it as company knowledge.
- Do not copy the internal solution directly.
- Understand it, translate it if needed, reformulate it clearly, and turn it into practical guidance.
- Start with only the first useful diagnostic step.
- Ask the technician for the result of that step.
- Continue step by step until the technician confirms that the solution works.

Mandatory wording rule:
- If you have determined that a similar internal case is genuinely relevant:
  * If the user writes in French, start your response with: "J'ai trouvé un cas interne similaire déjà résolu."
  * If the user writes in English, start your response with: "I found a similar internal case already resolved."
- Otherwise (if no cases are relevant or if the list is empty), do NOT include either sentence under any circumstances.
- For labels, if the user writes in French, use:
  "Étape actuelle :"
  "Question :"
- If the user writes in English, use English labels.
- If the user writes in Arabic, use Arabic labels.
- If the user writes in Moroccan Darija, use simple Darija labels.

If no useful internal solution is found:
- Propose one reasonable diagnostic path based on technical knowledge.
- Do not present it as validated company knowledge.
- Ask the technician to test one step and report the result.

If the technician does not understand:
- Explain more simply.
- Break the action into smaller steps.
- Avoid long theory.
- Give one clear action to do now.

If the technician says the proposed solution did not work:
- Do not repeat the same solution.
- Move to another possible cause or another diagnostic path.
- Ask one clear question to continue.

If the technician confirms the solution works:
- The backend will close the incident locally.
- Do not generate a long final message.

Tool usage rule:
- Do not use external tools unless absolutely necessary.
- First use the provided internal context and the conversation history.
- If the internal context is sufficient, answer directly.
- Do not search the web for every follow-up message.
- Only search externally if no internal solution exists or if the previous proposed solution failed.

Important:
- Maximum response length: 1200 characters.
- Do not give all diagnostic steps at once.
- Give only the current diagnostic step and one final question.
- If this is a follow-up answer, do not repeat the whole initial analysis.
- Keep the answer practical and technical.
- Use simple language suitable for a technician working in the field.
"""

    if DEBUG_MODE:
        print("\n===== RAG DEBUG LOGS =====")
        print("RAG RECALL: RAW CANDIDATES COUNT:", len(raw_cases))
        print("RAG RECALL: RAW CANDIDATE IDS:", [case.get("id") for case in raw_cases])
        print("RAG FILTER: CANDIDATES MARKED RELEVANT COUNT:", len(filtered_cases))
        print("RAG FILTER: CANDIDATES MARKED RELEVANT IDS:", [case.get("id") for case in filtered_cases])
        print("CONFIDENCE VALUES:", confidence_values)
        print("DYNAMIC THRESHOLD:", threshold)
        print("FINAL SELECTED CASES COUNT:", len(final_cases))
        print("FINAL SELECTED CASE IDS:", [case.get("id") for case in final_cases])
        print("RAG CONTEXT INJECTED:\n", rag_context)
        print("==========================\n")

    result = ask_openclaw(
        session_id=f"conv_{conversation.id}",
        prompt=prompt
    )

    response_text = result.get("response", "")

    user_msg = AssistantMessage(
        conversation_id=conversation.id,
        role="user",
        message=message_text
    )
    db.add(user_msg)

    assistant_msg = AssistantMessage(
        conversation_id=conversation.id,
        role="assistant",
        message=response_text
    )
    db.add(assistant_msg)

    conversation.updated_at = datetime.utcnow()
    db.commit()

    return {
        "user_id": request.user_id,
        "conversation_id": conversation.id,
        "incident_id": conversation.incident_id,
        "message": message_text,
        "response": response_text,
        "success": result.get("success", False),
        "resolved": False
    }


# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------

@router.post("/assistant/report/{conversation_id}")
def generate_assistant_report(conversation_id: int, db: Session = Depends(get_db)):
    conversation = db.query(AssistantConversation).filter(
        AssistantConversation.id == conversation_id
    ).first()

    if conversation is None:
        return {
            "success": False,
            "error": "Conversation introuvable"
        }

    messages = db.query(AssistantMessage).filter(
        AssistantMessage.conversation_id == conversation_id
    ).order_by(AssistantMessage.created_at.asc()).all()

    history = "\n\n".join([
        f"{msg.role.upper()} : {msg.message}"
        for msg in messages
    ])

    prompt = f"""
Tu es un assistant technique chargé de générer un rapport d'incident professionnel.

Le rapport doit être très lisible pour un humain.

Règles de formatage :
- Utilise des titres clairs
- Laisse des lignes vides entre sections
- Utilise des bullet points quand nécessaire
- Réponse bien espacée
- Style professionnel industriel
- Pas de texte compact

Problème initial :
{conversation.initial_question}

Contexte RAG :
{conversation.rag_context}

Historique complet :
{history}

Génère ce format EXACT :

# RAPPORT D'INCIDENT

## 1. Titre de l'incident

## 2. Description du problème

## 3. Symptômes observés

## 4. Analyse des causes probables

## 5. Étapes de diagnostic réalisées

## 6. Solution finale ou recommandée

## 7. Statut

## 8. Recommandations futures
"""

    result = ask_openclaw(
        session_id=f"report_{conversation_id}",
        prompt=prompt
    )

    return {
        "success": result.get("success", False),
        "conversation_id": conversation_id,
        "incident_id": conversation.incident_id,
        "report": result.get("response", "")
    }
