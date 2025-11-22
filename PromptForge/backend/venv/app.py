# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import json
import re
import sqlite3
import datetime

load_dotenv()

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not set. Using placeholder data.")

app = FastAPI(title="Prompt Enhancer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = "history.db"


# =====================================================
#                 REQUEST MODEL
# =====================================================
class PromptRequest(BaseModel):
    prompt: str
    mode: str = "general"
    tone: str = "neutral"
    length: str | None = None
    language: str = "english"


# =====================================================
#                 DATABASE
# =====================================================
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        prompt TEXT,
        enhanced TEXT,
        score INTEGER,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


# =====================================================
#              SYSTEM PROMPT BUILDER
# =====================================================
def build_system_prompt(mode: str, tone: str, language: str):
    base = (
        "You are a Prompt Enhancement assistant. "
        "Return ONLY JSON with keys: "
        "'enhanced_prompt', 'score' (0-100), 'suggestions' (array)."
    )

    mode_map = {
        "seo": "Include SEO keywords and meta description.",
        "technical": "Include constraints, inputs, outputs, examples.",
        "creative": "Add storytelling, creativity, vivid details.",
        "marketing": "Make it persuasive with benefits + CTA.",
        "general": ""
    }

    return (
        f"{base} "
        f"Mode: {mode_map.get(mode, '')} "
        f"Tone: {tone}. "
        f"Language: {language}."
    )


# =====================================================
#           CALL GROQ API
# =====================================================
def call_groq_chat(system_prompt: str, user_prompt: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API error: {resp.status_code} {resp.text}"
        )

    return resp.json()


# =====================================================
#                 SCORING
# =====================================================
def heuristic_score(enhanced_text: str):
    score = 50
    l = len(enhanced_text.split())

    if l > 20: score += 10
    if l > 50: score += 10

    for kw in ["format", "constraints", "examples", "audience", "tone"]:
        if kw in enhanced_text.lower():
            score += 5

    return min(100, score)


# =====================================================
#                 JSON EXTRACTION
# =====================================================
def extract_json(content: str):
    # try valid JSON first
    try:
        return json.loads(content)
    except:
        pass

    # try regex fallback
    m = re.search(r"(\{.*\})", content, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    raise HTTPException(500, f"Model did not provide valid JSON: {content}")


# =====================================================
#                 MAIN ROUTE /enhance
# =====================================================
@app.post("/enhance")
async def enhance(req: PromptRequest):

    system_prompt = build_system_prompt(req.mode, req.tone, req.language)

    user_msg = req.prompt
    if req.tone:
        user_msg += f"\nTone: {req.tone}"
    if req.length:
        user_msg += f"\nDesired length: {req.length}"
    if req.language:
        user_msg += f"\nOutput language: {req.language}"

    # ---- FALLBACK MODE (no API KEY) ----
    if not GROQ_API_KEY:
        enhanced_prompt = f"Enhanced ({req.mode}/{req.tone}/{req.language}): {req.prompt}"
        score = 80
        suggestions = ["Add more details", "Improve clarity"]
        raw = {"placeholder": True}

    else:
        raw = call_groq_chat(system_prompt, user_msg)

        try:
            content = raw["choices"][0]["message"]["content"]
        except:
            raise HTTPException(500, f"Invalid response: {raw}")

        obj = extract_json(content)
        enhanced_prompt = obj.get("enhanced_prompt", "")
        score = obj.get("score", heuristic_score(enhanced_prompt))
        suggestions = obj.get("suggestions", [])

    # ---- SAVE TO DB ----
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO history (prompt, enhanced, score, created_at) VALUES (?, ?, ?, ?)",
        (req.prompt, enhanced_prompt, score, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "enhanced_prompt": enhanced_prompt,
        "score": score,
        "suggestions": suggestions,
        "raw_llm_output": raw
    }


# =====================================================
#              MULTI-VERSION ROUTE  /enhance/multi
# =====================================================
@app.post("/enhance/multi")
async def enhance_multi(req: PromptRequest):

    # FIX: include language
    system_prompt = (
        build_system_prompt(req.mode, req.tone, req.language)
        + """
        Generate 3 enhanced versions of the same prompt.
        Respond ONLY in this JSON structure:
        {
          "versions": [
            {"id": "v1", "enhanced_prompt": "", "score": 0, "suggestions": []},
            {"id": "v2", "enhanced_prompt": "", "score": 0, "suggestions": []},
            {"id": "v3", "enhanced_prompt": "", "score": 0, "suggestions": []}
          ]
        }
        """
    )

    user_msg = req.prompt
    if req.tone:
        user_msg += f"\nTone: {req.tone}"
    if req.length:
        user_msg += f"\nLength: {req.length}"
    if req.language:
        user_msg += f"\nOutput language: {req.language}"

    raw = call_groq_chat(system_prompt, user_msg)

    try:
        content = raw["choices"][0]["message"]["content"]
    except:
        raise HTTPException(500, f"Invalid LLM structure: {raw}")

    obj = extract_json(content)

    versions = obj.get("versions", [])

    # compute score if missing
    for v in versions:
        if not v.get("score"):
            v["score"] = heuristic_score(v["enhanced_prompt"])

    return {
        "versions": versions,
        "raw_llm_output": raw
    }
