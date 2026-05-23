from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
import joblib
import os
import json
from datetime import datetime
import sys
import config
from clean_logging import setup_logging
from evidence_collector import collect_all_evidence
from debate_engine import run_debate
from peer_review import anonymize_transcript, collect_peer_reviews
from judge import judge_synthesis
import requests

# ------------------------------------------------------------------
# sklearn pickle compatibility guard (temporary but necessary)
# ------------------------------------------------------------------
import sklearn
sys.modules["sklearn"] = sklearn

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
logger, _ = setup_logging("aether_simple")

app = FastAPI(title="Orchestrator Query Classifier")

class tldr(BaseModel):
    query:str
    resp:str

@app.post("/tsdc/tldr")
def tldr(payload:tldr):
    prompt=f"""
        Your job is to form a High Level TL;DR of the given content. 
        The query is {payload.query}.
        The response to be summarized is {payload.resp}

        RULES:
        1. NEVER REVEAL YOURSELF IN ANY OF THE QUERIES.
        2. ALWAYS TRY TO GIVE A BRIEF OVERVIEW OF THE RESPONSE.
        3. YOUR JOB IS TO GIVE THE USER A SHORT YET DESCRIPTIVE GLANCE OF THE CONTENT.
    """
    response = requests.post(
        "http://164.52.211.24/yashas/models",
        json={
            "query": prompt,
            "model_name": "gpt-oss:20b",
            "stream": False
        },
        timeout=300
    )
    response.raise_for_status()

    data = response.json()

    return data

async def levelones(query,model_name,history=""):
    prompt=f"""
        You are a basic conversational Chatbot named AETHER. Your job is to answer questions. 

        RULES:
        1. NO ANSWER SHOULD BE GREATER THAN 1-2 LINES.
        2. NEVER AT ANY POINT REVEAL THAT YOU ARE AN AI. ALSO, NEVER REVEAL YOUR ORIGIN.
        3. THE LAST CONVERSATION HISTORY IS {history}.
        3. YOUR JOB IS TO GIVE THE MOST APPROPRIATE RESPONSE TO THE QUESTION {query}.
    """
    response = requests.post(
        "http://164.52.211.24/yashas/models",
        json={
            "query": prompt,
            "model_name": model_name,
            "stream": False
        },
        timeout=300
    )
    response.raise_for_status()

    data = response.json()
    data["history"]=f"The given question is {query} and the response is {data['response']}"
    return data


class levelone(BaseModel):
    query:str
    model_name:str
    history:str 

@app.post("/tsdc/levelone")
def call_one(payload:levelone):
    return levelones(payload.history,payload.query,payload.model_name)




class AnalysisRequests(BaseModel):
    content: str
    llm_count: int = 3

@app.post("/tsdc/option")
async def option(payload:AnalysisRequests):
    if payload.llm_count==5:
        resp = await process_content(content=payload.content, llm_count=5, use_web_search=True)
    else:
        resp = await process_content(content=payload.content, llm_count=3, use_web_search=True)
    return resp



class AnalysisRequest(BaseModel):
    content: str
    use_web_search: bool = False
    llm_count: int = 3  # 3 or 5

class AnalysisResponse(BaseModel):
    verdict: str
    debate_transcript: str
    peer_reviews: dict
    sources: List[str]
    metadata: dict

async def process_content(content: str, llm_count: int, use_web_search: bool):
    # Ensure outputs directory
    import os
    os.makedirs("outputs", exist_ok=True)
    
    if llm_count not in [3, 5]:
        # Fallback to nearest or error? User asked for choice.
        # Let's default to 5 if invalid, or error. 
        if llm_count < 4:
            llm_count = 3
        else:
            llm_count = 5
            
    logger.info(f"🚀 Starting analysis with {llm_count} LLMs. Search: {use_web_search}")
    
    start_time = datetime.now()
    
    try:
        # 1. Evidence Collection
        evidence = {"pro": [], "con": []}
        sources = []
        if use_web_search:
            # We treat the content as the factor/topic for simplicity or extract it?
            # User said "content". Let's use content as the topic.
            # Truncate for search query if too long
            topic = content[:200].replace("\n", " ")
            logger.info("🔍 Collecting evidence...")
            evidence = collect_all_evidence(topic, enable_scraping=True)
            
            # Extract sources
            for e in evidence.get("pro", []) + evidence.get("con", []):
                if e.get("source") and e["source"] not in sources:
                    sources.append(e["source"])
                    
        # 2. Debate
        logger.info(f"⚔️ Running debate ({llm_count} agents)...")
        # Treat content as both report and factor for simplicity in this mode
        debate_transcript = run_debate(report=content, factor=content[:100]+"...", evidence=evidence, agent_count=llm_count)
        
        # 3. Anonymize
        anonymized = anonymize_transcript(debate_transcript)
        
        # 4. Peer Review
        logger.info("👥 Collecting peer reviews...")
        reviews = collect_peer_reviews(anonymized, agent_count=llm_count)
        
        # 5. Judge Synthesis
        logger.info("⚖️ Judge synthesizing...")
        verdict = judge_synthesis(factor=content[:100]+"...", debate_transcript=debate_transcript, peer_reviews=reviews)
        
        # Save transcript to file (as requested)
        output_filename = f"outputs/debate_{int(start_time.timestamp())}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(debate_transcript)
        logger.info(f"💾 Transcript saved to {output_filename}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "verdict": verdict,
            "debate_transcript": debate_transcript,
            "peer_reviews": reviews,
            "sources": sources,
            "metadata": {
                "duration_seconds": duration,
                "llm_count": llm_count,
                "model": config.PRO_MODEL_1,
                "transcript_file": output_filename
            }
        }

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------
DEEP_THRESHOLD = 5
ML_CONFIDENCE_THRESHOLD = 0.65

# ------------------------------------------------------------------
# Paths (PORTABLE – works on any PC)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR,"models","query_classifier.pkl")
print(MODEL_PATH)
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "query_logs.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        f"Expected structure:\n"
        f"backend/\n"
        f"  classifier_api.py\n"
        f"  models/query_classifier.pkl"
    )

model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Keyword lists
# ------------------------------------------------------------------
DECISION_KEYWORDS = [
    "should", "ought", "must", "need to",
    "better", "worse", "stricter", "looser",
    "good idea", "bad idea", "recommend",
    "policy", "regulation", "law", "govern"
]

RISK_KEYWORDS = [
    "risk", "danger", "harm", "impact",
    "ethical", "safety", "bias",
    "privacy", "security", "liability"
]

COMPARE_KEYWORDS = [
    "compare", "vs", "versus", "difference",
    "better than", "pros and cons"
]

WEB_KEYWORDS = [
    "latest", "current", "today", "now", "recent",
    "news", "update", "price", "rate", "rates",
    "stock", "market", "score", "who won",
    "regulation", "regulations", "law", "policy",
    "released", "launch", "announced"
]

# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def extract_features(text: str) -> dict:
    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)

    return {
        "token_count": len(tokens),
        "sentence_count": max(1, text.count("?") + text.count(".")),
        "has_decision": any(k in text_lower for k in DECISION_KEYWORDS),
        "has_risk": any(k in text_lower for k in RISK_KEYWORDS),
        "has_compare": any(k in text_lower for k in COMPARE_KEYWORDS),
    }

def needs_web_search(text: str) -> bool:
    text_lower = text.lower()
    return any(k in text_lower for k in WEB_KEYWORDS)

# ------------------------------------------------------------------
# Rule-based scoring
# ------------------------------------------------------------------
def calculate_risk_score(features: dict) -> int:
    score = 0
    token_count = features["token_count"]

    if token_count > 25:
        score += 2
    if token_count > 50:
        score += 2

    if features["has_decision"]:
        score += 5
    if features["has_risk"]:
        score += 3
    if features["has_compare"]:
        score += 3

    return score

# ------------------------------------------------------------------
# ML prediction
# ------------------------------------------------------------------
def ml_predict(features: dict) -> float:
    vector = [[
        features["token_count"],
        features["sentence_count"],
        int(features["has_compare"]),
        int(features["has_risk"]),
        int(features["has_decision"])
    ]]
    return float(model.predict_proba(vector)[0][1])

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def log_query(text, decision, rule_score, ml_confidence, reasons, features, web_required):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": text,
        "decision": decision,
        "risk_score": rule_score,
        "ml_confidence": round(ml_confidence, 2),
        "web_required": web_required,
        "reasons": reasons,
        "features": features
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# ------------------------------------------------------------------
# Core classifier
# ------------------------------------------------------------------
async def classify_query(text: str, history:str) -> dict:
    features = extract_features(text)
    rule_score = calculate_risk_score(features)
    ml_confidence = ml_predict(features)
    web_required = needs_web_search(text)

    reasons = []

    if features["has_compare"]:
        reasons.append("Comparative analysis detected")
    if features["has_risk"]:
        reasons.append("High-risk domain keywords detected")
    if features["has_decision"]:
        reasons.append("Decision-oriented query detected")
    if features["token_count"] > 30:
        reasons.append("Long-form or multi-step query")
    if web_required:
        reasons.append("Fresh or real-time information required")

    decision = "NORMAL"

    if rule_score >= DEEP_THRESHOLD:
        decision = "DEEP_RESEARCH"
        reasons.append("Rule-based risk threshold exceeded")
    elif ml_confidence >= ML_CONFIDENCE_THRESHOLD and not web_required:
        decision = "DEEP_RESEARCH"
        reasons.append("ML confidence exceeded threshold")

    log_query(
        text, decision, rule_score, ml_confidence,
        reasons, features, web_required
    )
    if decision == "NORMAL" and len(reasons) == 0:
        resp = await levelones(text,"gemma3:4b",history)
    elif decision == "NORMAL" and len(reasons) != 0:
        resp = await process_content(content=text, llm_count=3, use_web_search=True)
    else:
        resp = await process_content(content=text, llm_count=5, use_web_search=True)

    return resp

# ------------------------------------------------------------------
# API schemas
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    history:str

class ClassificationResponse(BaseModel):
    decision: str
    risk_score: int
    ml_confidence: float
    reasons: List[str]
    features: Dict
    web_required: bool

# ------------------------------------------------------------------
# API endpoint
# ------------------------------------------------------------------
@app.post("/tsdc/classify")
async def classify(request: QueryRequest):
    return await classify_query(request.query,request.history)


class vision(BaseModel):
    query:str

@app.post('/tsdc/vision')
def status(payload:vision):
    return {"response":f"Sab changa si!!! Received {payload.query}"}