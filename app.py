"""
🇩🇪 German News Agent
Fetches the latest news from DW & Tagesschau, combines them into
a single CEFR-level briefing, validated by a LangGraph pipeline.
"""

import os
import json
import re
import time
import requests
import feedparser
import streamlit as st
from datetime import datetime
from typing import List, Optional, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PYDANTIC MODELS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VocabularyItem(BaseModel):
    word: str = Field(description="The German word")
    translation: str = Field(description="English translation")
    example: str = Field(description="Example sentence in German")


class GrammarSpotlight(BaseModel):
    rule_name: str = Field(description="Name of the grammar rule")
    explanation: str = Field(description="Simple explanation in German")
    example_from_text: str = Field(description="Sentence from the summary using this rule")


class NewsBriefing(BaseModel):
    headline: str = Field(description="A short, level-appropriate briefing headline")
    summary: str = Field(description="Comprehensive multi-paragraph summary of all major news")
    bullet_points: List[str] = Field(description="5-7 key points across all stories")
    vocabulary: List[VocabularyItem] = Field(description="8-10 key vocabulary words")
    grammar_spotlights: List[GrammarSpotlight] = Field(description="3 grammar rule spotlights")
    source_count: int = Field(default=0, description="Number of source articles used")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEWS FETCHER  (DW + Tagesschau – both at once)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DW_FEED = "https://rss.dw.com/rdf/rss-de-top"
TAGESSCHAU_FEED = "https://www.tagesschau.de/index~rss2.xml"


def _parse_feed(url: str, source_name: str, max_items: int = 8) -> list[dict]:
    """Parse a single RSS feed and return cleaned articles."""
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        desc = entry.get("summary", entry.get("description", ""))
        desc = re.sub(r"<[^>]+>", "", desc).strip()
        if title.strip():
            articles.append({
                "title": title,
                "description": desc,
                "source": source_name,
                "url": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
    return articles


def fetch_all_news() -> tuple[list[dict], str]:
    """
    Fetch top news from BOTH DW and Tagesschau.
    Returns (list_of_articles, combined_text_for_llm).
    """
    dw_articles = _parse_feed(DW_FEED, "Deutsche Welle")
    ts_articles = _parse_feed(TAGESSCHAU_FEED, "Tagesschau")

    all_articles = dw_articles + ts_articles

    # Build a combined text block for the LLM
    text_parts = []
    for i, a in enumerate(all_articles, 1):
        text_parts.append(
            f"[{i}] ({a['source']}) {a['title']}\n{a['description']}"
        )
    combined_text = "\n\n".join(text_parts)

    return all_articles, combined_text


def try_enrich_article(url: str) -> Optional[str]:
    """Try to scrape fuller text from an article URL."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", resp.text, re.DOTALL)
        texts = [re.sub(r"<[^>]+>", "", p).strip() for p in paragraphs]
        texts = [t for t in texts if len(t) > 40]
        if texts:
            return "\n".join(texts[:10])  # cap to avoid token overflow
    except Exception:
        pass
    return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MASTER PROMPT  (multi-article briefing)                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

MASTER_PROMPT = """You are an expert German Language Teacher (DaF – Deutsch als Fremdsprache).
You will receive multiple news items from Deutsche Welle and Tagesschau.
Your task is to create a **comprehensive daily news briefing** at the user's CEFR level.

## Task Instructions
1. **Read all articles**: Identify the most important stories.
2. **Write a unified briefing**: Cover ALL major stories in a flowing, multi-paragraph summary (4-6 paragraphs). 
   Each paragraph should cover a different story or theme. Be thorough — the user wants rich detail.
3. **Extract 5-7 bullet points**: These should capture the key facts across all stories.
4. **Extract 8-10 vocabulary words**: Pick important words characteristic of this CEFR level.
   Each word MUST include a realistic example sentence from a news context.
5. **Grammar Spotlights (3)**: Highlight THREE different grammar rules used in your writing.
   For each, explain the rule clearly and show a real example from your summary.
   Pick diverse rules (e.g., one about verb tense, one about sentence structure, one about cases or connectors).
6. **Output**: Return ONLY a JSON object, no extra text.

## Level-Specific Rules

### B1
- Grammar: Use Perfekt (not Präteritum). Use "man" instead of passive. Max 15 words per sentence.
  Use weil, dass, wenn.
- Vocabulary: Top 2,000 common words. Explain jargon in brackets.

### B2
- Grammar: Use Passiv (wird gemacht). Use complex connectors (trotzdem, obwohl, während).
  Max 25 words per sentence.
- Vocabulary: Thematic/professional terms.

### C1
- Grammar: Nominalstil (noun-heavy). Konjunktiv I for indirect speech. Genitiv. No length limit.
- Vocabulary: Academic, nuanced, idiomatic German.

## JSON Output Schema
Return ONLY this JSON — no markdown fences, no commentary:

{{
  "headline": "A short daily briefing headline for this level",
  "summary": "Comprehensive summary covering all major stories (4-6 paragraphs, detailed)",
  "bullet_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
  "vocabulary": [
    {{"word": "German Word", "translation": "English", "example": "A full German example sentence"}},
    {{"word": "...", "translation": "...", "example": "..."}}
  ],
  "grammar_spotlights": [
    {{
      "rule_name": "Name of rule 1",
      "explanation": "Clear explanation in German",
      "example_from_text": "Sentence from your summary using this rule"
    }},
    {{
      "rule_name": "Name of rule 2",
      "explanation": "Clear explanation in German",
      "example_from_text": "Sentence from your summary using this rule"
    }},
    {{
      "rule_name": "Name of rule 3",
      "explanation": "Clear explanation in German",
      "example_from_text": "Sentence from your summary using this rule"
    }}
  ],
  "source_count": <number of articles you referenced>
}}
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LANGGRAPH PIPELINE  (Rewrite → Validate → Retry)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AgentState(TypedDict):
    combined_text: str
    target_level: str
    json_text: str
    result: Optional[NewsBriefing]
    validation_feedback: str
    attempts: int
    error: str


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file.\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)


def rewrite_node(state: AgentState) -> dict:
    """Call the LLM to produce a combined news briefing."""
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", MASTER_PROMPT),
        ("human", "Level: {target_level}\n\nArticles:\n{combined_text}"),
    ])

    # Append feedback on retry
    combined = state["combined_text"]
    if state.get("validation_feedback"):
        combined += (
            f"\n\n IMPORTANT: Your previous output had issues. "
            f"Fix these problems: {state['validation_feedback']}"
        )

    chain = prompt | llm
    try:
        response = chain.invoke({
            "target_level": state["target_level"],
            "combined_text": combined,
        })
        text = response.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]).strip()
        return {"json_text": text, "attempts": state.get("attempts", 0) + 1, "error": ""}
    except Exception as e:
        return {"error": str(e), "attempts": state.get("attempts", 0) + 1}


def validate_node(state: AgentState) -> dict:
    """Validate with Pydantic + quality checks, optionally with LLM."""
    json_text = state.get("json_text", "")
    if not json_text:
        return {"validation_feedback": "No output produced. Try again.", "result": None}

    # 1) Parse JSON
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        return {"validation_feedback": f"Invalid JSON: {e}", "result": None}

    # 2) Pydantic validation
    try:
        result = NewsBriefing(**parsed)
    except ValidationError as e:
        errors = "; ".join(err["msg"] for err in e.errors())
        return {"validation_feedback": f"Schema errors: {errors}", "result": None}

    # 3) Quality checks
    issues = []
    if len(result.bullet_points) < 4:
        issues.append("Need at least 5 bullet points")
    if len(result.vocabulary) < 6:
        issues.append("Need at least 8 vocabulary items with example sentences")
    if len(result.grammar_spotlights) < 2:
        issues.append("Need 3 grammar spotlights with explanations and examples")
    if len(result.summary) < 200:
        issues.append("Summary is too short — need 4-6 detailed paragraphs")

    if issues:
        return {"validation_feedback": "; ".join(issues), "result": None}

    # 4) LLM quality check on first attempt
    if state.get("attempts", 0) <= 1:
        try:
            llm = _get_llm()
            val_prompt = ChatPromptTemplate.from_messages([("human",
                "You are a strict quality validator. Check this JSON news briefing for CEFR level "
                "{target_level}. Is it well-written, comprehensive, and level-appropriate? "
                "Does it have enough detail? If yes, reply EXACTLY: VALID. "
                "Otherwise, briefly say what to fix.\n\nJSON:\n{json_text}"
            )])
            resp = (val_prompt | llm).invoke({
                "target_level": state["target_level"],
                "json_text": json_text,
            })
            feedback = resp.content.strip()
            if "VALID" in feedback.upper() and len(feedback) < 30:
                return {"result": result, "validation_feedback": ""}
            else:
                return {"validation_feedback": feedback, "result": None}
        except Exception:
            return {"result": result, "validation_feedback": ""}

    return {"result": result, "validation_feedback": ""}


def should_retry(state: AgentState) -> str:
    if state.get("result") is not None:
        return "done"
    if state.get("attempts", 0) >= 3:
        return "done"
    if state.get("error"):
        return "done"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "validate")
    graph.add_conditional_edges("validate", should_retry, {"retry": "rewrite", "done": END})
    return graph.compile()


def generate_briefing(combined_text: str, level: str) -> NewsBriefing:
    """Run the full LangGraph pipeline."""
    graph = build_graph()
    final = graph.invoke({
        "combined_text": combined_text,
        "target_level": level,
        "json_text": "",
        "result": None,
        "validation_feedback": "",
        "attempts": 0,
        "error": "",
    })

    if final.get("result"):
        return final["result"]

    if final.get("json_text"):
        try:
            return NewsBriefing(**json.loads(final["json_text"]))
        except Exception:
            pass

    raise RuntimeError(
        f"Failed after {final.get('attempts', 0)} attempts. "
        f"Error: {final.get('error', '')} | Feedback: {final.get('validation_feedback', '')}"
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STREAMLIT UI                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="🇩🇪 German News Agent",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #000000; /* Pure Black Background */
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 1rem;
        border-bottom: 1px solid #333;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Level badges - Monochrome */
    .level-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border: 1px solid #FFFFFF;
    }
    .level-B1 { background: #333; color: #FFF; border-color: #444; }
    .level-B2 { background: #666; color: #FFF; border-color: #777; }
    .level-C1 { background: #FFF; color: #000; border-color: #FFF; }

    /* Glass card - High Contrast */
    .glass-card {
        background: #111111;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: #FFFFFF;
    }
    .glass-card h3 {
        margin-top: 0;
        color: #FFFFFF;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #222;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Vocab table - Monochrome */
    .vocab-table {
        width: 100%;
        border-collapse: collapse;
    }
    .vocab-table th {
        text-align: left;
        color: #666;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #222;
    }
    .vocab-table td {
        padding: 1rem;
        color: #CCC;
        font-size: 0.95rem;
        border-bottom: 1px solid #1a1a1a;
    }
    .vocab-table tr td:first-child {
        font-weight: 700;
        color: #FFFFFF;
    }

    /* Bullet list - Minimalist */
    .bullet-item {
        border-left: 2px solid #FFFFFF;
        padding: 1rem 1.5rem;
        margin-bottom: 0.8rem;
        background: #0a0a0a;
        color: #BBB;
        font-size: 0.95rem;
    }

    /* Grammar card - Inverted */
    .grammar-card {
        background: #FFFFFF;
        color: #000000;
        border-radius: 4px;
        padding: 2rem;
    }
    .grammar-card h3 { 
        color: #000000 !important; 
        margin-top: 0; 
        border-bottom: 1px solid #DDD;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .grammar-rule { 
        color: #000000 !important; 
        font-weight: 800; 
        font-size: 1.2rem; 
        text-transform: uppercase;
    }
    .grammar-explanation { 
        color: #333 !important; 
        margin: 1rem 0; 
        line-height: 1.6; 
    }
    .grammar-example {
        background: #f0f0f0;
        border-radius: 2px;
        padding: 1.2rem;
        color: #000 !important;
        font-style: italic;
        font-size: 0.95rem;
        border-left: 4px solid #000;
        margin-top: 1rem;
    }

    .source-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.5rem;
        text-transform: uppercase;
        border: 1px solid #333;
    }
    .source-dw { background: #000; color: #FFF; border-color: #444; }
    .source-ts { background: #FFF; color: #000; border-color: #FFF; }

    .meta-bar {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1rem 1.5rem;
        background: #0a0a0a;
        border: 1px solid #222;
        border-radius: 4px;
        margin-bottom: 2rem;
        color: #666;
        font-size: 0.85rem;
    }
    .meta-bar strong { color: #FFF; }

    .graph-info {
        background: #111;
        border: 1px solid #333;
        color: #888;
        padding: 1rem;
        font-size: 0.8rem;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid #222;
    }

    /* Primary buttons as white-on-black */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 2px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        border: none !important;
        padding: 0.6rem 1rem !important;
    }
    div.stButton > button:hover {
        background-color: #CCCCCC !important;
        color: #000000 !important;
    }


    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🇩🇪 German News Agent</h1>
    <p>Your daily German news briefing from DW & Tagesschau — at your CEFR level</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    target_level = st.selectbox(
        "CEFR Level",
        options=["B1", "B2", "C1"],
        index=0,
        help="B1 = Intermediate · B2 = Upper-Intermediate · C1 = Advanced",
    )

    st.markdown("---")
    st.markdown("### Auto-Refresh")
    auto_refresh = st.toggle("Enable auto-refresh", value=False)
    if auto_refresh:
        refresh_minutes = st.slider(
            "Refresh every (minutes)", min_value=5, max_value=120, value=30, step=5
        )
        st.info(f"🔄 Will auto-refresh every {refresh_minutes} min")

    st.markdown("---")
    st.markdown("### Sources")
    st.markdown(
        "<span class='source-tag source-dw'>Deutsche Welle</span>"
        "<span class='source-tag source-ts'>Tagesschau</span>",
        unsafe_allow_html=True,
    )
    st.caption("Top stories from both sources are combined into one briefing.")


    st.markdown("---")
    st.markdown(
        "<div style='color:#8892b0; font-size:0.78rem; text-align:center;'>"
        "Powered by OpenAI · DW · Tagesschau</div>",
        unsafe_allow_html=True,
    )

# ── Session State ────────────────────────────────────────────────────────────
if "briefing" not in st.session_state:
    st.session_state.briefing = None
if "articles" not in st.session_state:
    st.session_state.articles = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "fetch_time" not in st.session_state:
    st.session_state.fetch_time = None

# ── Auto-refresh logic ──────────────────────────────────────────────────────
needs_auto_refresh = False
if auto_refresh and st.session_state.last_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if elapsed >= refresh_minutes * 60:
        needs_auto_refresh = True

# ── Main Action ──────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    get_summary = st.button(
        "Get Latest Summary",
        use_container_width=True,
        type="primary",
        help="Fetches latest news from DW & Tagesschau and creates a combined briefing",
    )

should_generate = get_summary or needs_auto_refresh

if should_generate:
    # Step 1: Fetch
    with st.spinner(" Fetching latest news from Deutsche Welle & Tagesschau..."):
        try:
            articles, combined_text = fetch_all_news()
            st.session_state.articles = articles
            st.session_state.fetch_time = datetime.now()
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            st.stop()

    if not articles:
        st.warning("No articles found. Please try again later.")
        st.stop()

    # Step 2: Generate briefing
    with st.spinner(
        f" Creating {target_level}-level briefing from {len(articles)} articles "
        
    ):
        try:
            briefing = generate_briefing(combined_text, target_level)
            st.session_state.briefing = briefing
            st.session_state.last_refresh = datetime.now()
        except Exception as e:
            st.error(f"Error generating briefing: {e}")
            st.stop()

# ── Display Briefing ─────────────────────────────────────────────────────────
if st.session_state.briefing:
    briefing: NewsBriefing = st.session_state.briefing
    articles = st.session_state.articles

    # Meta bar
    dw_count = sum(1 for a in articles if a["source"] == "Deutsche Welle")
    ts_count = sum(1 for a in articles if a["source"] == "Tagesschau")
    refresh_str = st.session_state.last_refresh.strftime("%H:%M") if st.session_state.last_refresh else "—"

    st.markdown(
        f"<div class='meta-bar'>"
        f"<span class='level-badge level-{target_level}'>{target_level}</span>"
        f"<span><strong>{dw_count}</strong> DW + <strong>{ts_count}</strong> Tagesschau articles</span>"
        f"<span> Last updated: <strong>{refresh_str}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Headline
    st.markdown(
        f"<div class='glass-card'><h2 style='color:#FFF; margin:0;'>"
        f"{briefing.headline}</h2></div>",
        unsafe_allow_html=True,
    )

    # ── Row 1: Summary + Key Points ──────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        formatted_summary = briefing.summary.replace("\n\n", "</p><p style='color:#ccd6f6; line-height:1.8; margin-top:0.8rem;'>")
        formatted_summary = formatted_summary.replace("\n", "<br>")
        st.markdown(
            f"<div class='glass-card'><h3>News Briefing</h3>"
            f"<p style='color:#ccd6f6; line-height:1.8; font-size:1rem;'>"
            f"{formatted_summary}</p></div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("<div class='glass-card'><h3>Key Points</h3>", unsafe_allow_html=True)
        for bp in briefing.bullet_points:
            st.markdown(f"<div class='bullet-item'>{bp}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Sources (compact)
        st.markdown("<div class='glass-card'><h3>Sources</h3>", unsafe_allow_html=True)
        for a in articles[:10]:
            tag_class = "source-dw" if a["source"] == "Deutsche Welle" else "source-ts"
            st.markdown(
                f"<div style='margin-bottom:0.5rem;'>"
                f"<span class='source-tag {tag_class}'>{a['source']}</span>"
                f"<a href='{a['url']}' target='_blank' style='color:#8892b0; "
                f"text-decoration:none; font-size:0.85rem;'>{a['title']}</a></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Vocabulary (full width) ───────────────────────────────────
    st.markdown("---")
    vocab_rows = ""
    for v in briefing.vocabulary:
        vocab_rows += (
            f"<tr><td>{v.word}</td><td>{v.translation}</td>"
            f"<td style='font-style:italic; color:#8892b0;'>{v.example}</td></tr>"
        )
    st.markdown(
        f"<div class='glass-card'><h3>Vocabulary — {target_level} Level ({len(briefing.vocabulary)} Words)</h3>"
        f"<table class='vocab-table'>"
        f"<tr><th>Wort</th><th>Translation</th><th>Beispielsatz</th></tr>"
        f"{vocab_rows}</table></div>",
        unsafe_allow_html=True,
    )

    # ── Row 3: Grammar Spotlights (3 cards side by side) ─────────────────
    st.markdown("---")
    grammar_cols = st.columns(len(briefing.grammar_spotlights))
    for i, gs in enumerate(briefing.grammar_spotlights):
        with grammar_cols[i]:
            st.markdown(
                f"<div class='grammar-card'>"
                f"<h3>Grammar {i+1}</h3>"
                f"<div class='grammar-rule'>{gs.rule_name}</div>"
                f"<div class='grammar-explanation'>{gs.explanation}</div>"
                f"<div class='grammar-example'>„{gs.example_from_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;"> 🇩🇪 </div>
        <h2 style="color: #ccd6f6; font-weight: 300; margin-bottom: 1rem;">
            Your Daily German News Briefing
        </h2>
        <p style="color: #8892b0; max-width: 550px; margin: 0 auto; line-height: 1.7;">
            Click <strong style="color:#ffd700;">Get Latest Summary</strong> to fetch the latest
            news from <strong style="color:#00a8e8;">Deutsche Welle</strong> and
            <strong style="color:#5eb1ef;">Tagesschau</strong>, combined into a single
            comprehensive briefing at your CEFR level with vocabulary and grammar highlights.
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem;">
            <span class="level-badge level-B1">B1 Intermediate</span>
            <span class="level-badge level-B2">B2 Upper-Intermediate</span>
            <span class="level-badge level-C1">C1 Advanced</span>
        </div>
        <div style="margin-top: 2rem; color: #8892b0; font-size: 0.85rem;">
             Enable auto-refresh in the sidebar for scheduled updates
        </div>
        <div style="margin-top: 0.5rem; color: #00cec9; font-size: 0.8rem;">
             Quality validated with LangGraph (Rewrite → Validate → Retry)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Auto-refresh timer (triggers rerun at the right time) ────────────────────
if auto_refresh and st.session_state.last_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    remaining = max(0, refresh_minutes * 60 - elapsed)
    if remaining > 0:
        time.sleep(min(remaining, 5))  # check every 5 seconds
        st.rerun()
