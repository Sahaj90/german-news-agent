"""
🇩🇪 German News Agent
Fetches the latest news from DW & Tagesschau, combines them into
a single CEFR-level briefing, validated by a LangGraph pipeline.
"""

import os
import json
import re
import time
import calendar
import requests
import feedparser
import streamlit as st
from datetime import datetime
from typing import List, Optional, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dotenv import load_dotenv

def get_german_time() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        # Strip timezone info after calculation to prevent naive/aware comparison crashes 
        # with old session states during live reloads
        return datetime.now(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
    except Exception:
        return datetime.now()
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)

# PYDANTIC MODELS                                                       

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
    vocabulary: List[VocabularyItem] = Field(description="10 key vocabulary words")
    grammar_spotlights: List[GrammarSpotlight] = Field(description="5 grammar rule spotlights")
    source_count: int = Field(default=0, description="Number of source articles used")



# NEWS FETCHER  (DW + Tagesschau)                  


import concurrent.futures

DW_FEEDS = [
    "https://rss.dw.com/xml/rss-de-all",
    "https://rss.dw.com/xml/rss-de-top",
    "https://rss.dw.com/xml/rss-de-news",
    "https://rss.dw.com/xml/rss-de-eco",
    "https://rss.dw.com/xml/rss-de-wissenschaft",
    "https://rss.dw.com/xml/rss-de-sport",
    "https://rss.dw.com/xml/rss-de-cul",
    "https://rss.dw.com/xml/rss-de-cul-buch",
    "https://rss.dw.com/xml/rss-de-cul-film",
    "https://rss.dw.com/xml/rss-de-cul-musik",
    "https://rss.dw.com/xml/rss-de-deutschlandentdecken",
]

# TAGESSCHAU_FEEDS = [
#     "https://www.tagesschau.de/infoservices/alle-meldungen-100~rss2.xml",
#     "https://www.tagesschau.de/index~rss2.xml",
#     "https://www.tagesschau.de/inland/index~rss2.xml",
#     "https://www.tagesschau.de/inland/innenpolitik/index~rss2.xml",
#     "https://www.tagesschau.de/inland/gesellschaft/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/finanzen/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/unternehmen/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/verbraucher/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/technologie/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/weltwirtschaft/index~rss2.xml",
#     "https://www.tagesschau.de/wirtschaft/konjunktur/index~rss2.xml",
#     "https://www.tagesschau.de/wissen/index~rss2.xml",
#     "https://www.tagesschau.de/wissen/gesundheit/index~rss2.xml",
#     "https://www.tagesschau.de/wissen/klima/index~rss2.xml",
#     "https://www.tagesschau.de/wissen/forschung/index~rss2.xml",
#     "https://www.tagesschau.de/wissen/technologie/index~rss2.xml",
#     "https://www.tagesschau.de/ausland/index~rss2.xml",
#     "https://www.tagesschau.de/ausland/europa/index~rss2.xml",
#     "https://www.tagesschau.de/ausland/amerika/index~rss2.xml",
#     "https://www.tagesschau.de/ausland/afrika/index~rss2.xml",
#     "https://www.tagesschau.de/ausland/asien/index~rss2.xml",
#     "https://www.tagesschau.de/faktenfinder/index~rss2.xml",
#     "https://www.tagesschau.de/investigativ/index~rss2.xml",
# ]

TAGESSCHAU_FEEDS = [
    # Top / All News
    "https://www.tagesschau.de/infoservices/alle-meldungen-100~rss2.xml",
    
    # Business & Tech
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
    "https://www.tagesschau.de/wirtschaft/finanzen/index~rss2.xml",
    "https://www.tagesschau.de/wirtschaft/technologie/index~rss2.xml",
    "https://www.tagesschau.de/wirtschaft/unternehmen/index~rss2.xml",
    "https://www.tagesschau.de/wirtschaft/konjunktur/index~rss2.xml",
    
    # Politics & World
    "https://www.tagesschau.de/inland/index~rss2.xml",
    "https://www.tagesschau.de/ausland/index~rss2.xml",
    "https://www.tagesschau.de/ausland/europa/index~rss2.xml",
    "https://www.tagesschau.de/ausland/amerika/index~rss2.xml",
    "https://www.tagesschau.de/ausland/asien/index~rss2.xml",
    "https://www.tagesschau.de/ausland/afrika/index~rss2.xml",
    
    # Science, Health & Climate
    "https://www.tagesschau.de/wissen/index~rss2.xml",
    "https://www.tagesschau.de/wissen/gesundheit/index~rss2.xml",
    "https://www.tagesschau.de/wissen/klima/index~rss2.xml",
    "https://www.tagesschau.de/wissen/forschung/index~rss2.xml",
    
    # Sports (Including Sportschau for details)
    "https://www.tagesschau.de/sport/index~rss2.xml",
    "https://www.sportschau.de/fussball/index~rss2.xml",
    "https://www.sportschau.de/olympia/index~rss2.xml",
    
    # Society & Deep Dives
    "https://www.tagesschau.de/inland/gesellschaft/index~rss2.xml",
    "https://www.tagesschau.de/investigativ/index~rss2.xml",
    "https://www.tagesschau.de/faktenfinder/index~rss2.xml"
]

ARTICLES_PER_FEED = 25
ARTICLE_AGENT_POOL_SIZE = 20
ARTICLE_FINDER_SHORTLIST_SIZE = 10
ALLOWED_FUTURE_SKEW_SECONDS = 300


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_html(text: str) -> str:
    return _normalize_whitespace(re.sub(r"<[^>]+>", " ", text or ""))


def _clean_article_url(url: str) -> str:
    if not url:
        return ""

    parts = urlsplit(url)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"maca"}
    ]
    cleaned_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc.lower(), cleaned_path, urlencode(filtered_query), ""))


def _extract_timestamp(entry) -> float:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_value = getattr(entry, attr, None)
        if parsed_value:
            try:
                return float(calendar.timegm(parsed_value))
            except Exception:
                continue
    return 0.0


def _article_lookup_text(article: dict) -> str:
    return _normalize_whitespace(
        " ".join(
            [
                article.get("title", ""),
                article.get("description", ""),
                article.get("full_text", ""),
            ]
        )
    ).lower()


def _extract_entry_full_text(entry) -> str:
    parts = []

    for key in ("summary", "description"):
        value = entry.get(key, "")
        if value:
            parts.append(_strip_html(value))

    for content_item in entry.get("content", []) or []:
        if isinstance(content_item, dict):
            value = content_item.get("value", "")
            if value:
                parts.append(_strip_html(value))

    return _normalize_whitespace(" ".join(part for part in parts if part))


def _extract_search_keywords(search_query: str) -> list[str]:
    stopwords = {
        "der", "die", "das", "und", "oder", "mit", "ohne", "für", "fur", "von",
        "den", "dem", "des", "ein", "eine", "einer", "eines", "im", "in", "am",
        "an", "auf", "zu", "zum", "zur", "bei", "nach", "vor", "über", "uber",
        "the", "and", "for", "latest", "news",
    }
    tokens = re.findall(r"\w+", (search_query or "").lower())
    return [token for token in tokens if len(token) > 2 and token not in stopwords]


def _keyword_match_score(article: dict, search_query: str, keywords: list[str]) -> int:
    if not search_query:
        return 0

    haystack = _article_lookup_text(article)
    score = 0
    normalized_query = _normalize_whitespace(search_query).lower()
    if normalized_query and normalized_query in haystack:
        score += 12

    for keyword in keywords:
        if keyword in haystack:
            score += 3
            if keyword in article.get("title", "").lower():
                score += 2

    return score


def _parse_feed(url: str, source_name: str, max_items: int = 8) -> list[dict]:
    """Parse a single RSS feed and return cleaned articles."""
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_items]:
        title = _normalize_whitespace(entry.get("title", ""))
        desc = entry.get("summary", entry.get("description", ""))
        desc = _strip_html(desc)
        full_text = _extract_entry_full_text(entry)
        cleaned_url = _clean_article_url(entry.get("link", ""))
        if title.strip():
            timestamp = _extract_timestamp(entry)
            articles.append({
                "title": title,
                "description": desc,
                "full_text": full_text,
                "source": source_name,
                "url": cleaned_url,
                "published": entry.get("published", ""),
                "timestamp": timestamp,
            })
    return articles


class ArticleFinderOutput(BaseModel):
    selected_urls: List[str] = Field(description="Ordered list of the best candidate article URLs")


class ArticleEvaluationItem(BaseModel):
    url: str = Field(description="Candidate article URL")
    is_relevant: bool = Field(description="True if this article clearly matches the search query")
    reason: str = Field(description="Short explanation for the decision")


class ArticleEvaluatorOutput(BaseModel):
    evaluations: List[ArticleEvaluationItem] = Field(description="Relevance decisions for candidate articles")


def _build_article_candidate_block(articles: list[dict], search_query: str = "") -> str:
    keywords = _extract_search_keywords(search_query)
    lines = []
    for index, article in enumerate(articles, start=1):
        match_score = _keyword_match_score(article, search_query, keywords)
        lines.append(
            "\n".join(
                [
                    f"[{index}] URL: {article['url']}",
                    f"Source: {article['source']}",
                    f"Published: {article.get('published') or 'Unknown'}",
                    f"Timestamp: {int(article.get('timestamp', 0) or 0)}",
                    f"KeywordScore: {match_score}",
                    f"Title: {article['title']}",
                    f"Description: {article['description']}",
                ]
            )
        )
    return "\n\n".join(lines)


def _fallback_select_articles(candidate_articles: list[dict], top_n: int, search_query: str = "") -> list[dict]:
    keywords = _extract_search_keywords(search_query)
    if search_query:
        ranked = sorted(
            candidate_articles,
            key=lambda article: (
                _keyword_match_score(article, search_query, keywords),
                article.get("timestamp", 0),
            ),
            reverse=True,
        )
    else:
        ranked = sorted(candidate_articles, key=lambda article: article.get("timestamp", 0), reverse=True)
    return ranked[:top_n]


def _select_articles_with_finder_agent(candidate_articles: list[dict], top_n: int, search_query: str = "") -> list[dict]:
    if not candidate_articles:
        return []

    llm = _get_llm().with_structured_output(ArticleFinderOutput)
    shortlist_target = min(len(candidate_articles), max(top_n, ARTICLE_FINDER_SHORTLIST_SIZE))
    current_time = get_german_time().strftime("%Y-%m-%d %H:%M:%S")
    query_text = search_query.strip() or "No keyword filter. Focus on the latest distinct stories."
    prompt = f"""You are the Article Finder Agent for a German news briefing app.
Current time in Germany: {current_time}
Search query: {query_text}

Choose the strongest shortlist from the candidates below.

Rules:
- Return only URLs that already appear in the candidate list.
- Prefer articles published on or before the current time.
- Prefer the newest items.
- When a search query exists, strongly prioritize direct relevance to that query and close variants.
- Avoid near-duplicates unless they clearly add new information.
- Keep a healthy mix of sources when quality is similar.
- Return between {top_n} and {shortlist_target} URLs in best-first order.

Candidates:
{_build_article_candidate_block(candidate_articles, search_query)}
"""

    try:
        output = llm.invoke(prompt)
        chosen_urls = []
        seen = set()
        by_url = {article["url"]: article for article in candidate_articles}
        for url in output.selected_urls:
            cleaned = _clean_article_url(url)
            if cleaned in by_url and cleaned not in seen:
                chosen_urls.append(cleaned)
                seen.add(cleaned)
        selected = [by_url[url] for url in chosen_urls]
        if selected:
            return selected[:shortlist_target]
    except Exception:
        pass

    return _fallback_select_articles(candidate_articles, shortlist_target, search_query)


def _evaluate_articles_with_keyword_agent(selected_articles: list[dict], search_query: str, top_n: int) -> list[dict]:
    if not selected_articles:
        return []

    llm = _get_llm().with_structured_output(ArticleEvaluatorOutput)
    prompt = f"""You are the Evaluator Agent for article relevance.
Search query: {search_query}

Check whether each article is genuinely relevant to the search query.

Rules:
- Mark is_relevant true only for clear, direct matches.
- Reject weak or merely tangential mentions.
- Consider the title and description together.
- Return decisions for every URL listed below.

Candidates:
{_build_article_candidate_block(selected_articles, search_query)}
"""

    try:
        output = llm.invoke(prompt)
        approved_urls = {
            _clean_article_url(item.url)
            for item in output.evaluations
            if item.is_relevant
        }
        approved = [article for article in selected_articles if article["url"] in approved_urls]
        if approved:
            return approved[:top_n]
    except Exception:
        pass

    keywords = _extract_search_keywords(search_query)
    return [
        article
        for article in selected_articles
        if _keyword_match_score(article, search_query, keywords) > 0
    ][:top_n]


def curate_articles_with_agents(all_articles: list[dict], top_n: int = 5, search_query: str = "") -> list[dict]:
    if not all_articles:
        return []

    now_ts = get_german_time().timestamp()
    recent_articles = [
        article
        for article in all_articles
        if not article.get("timestamp") or article["timestamp"] <= now_ts + ALLOWED_FUTURE_SKEW_SECONDS
    ]
    if not recent_articles:
        recent_articles = list(all_articles)

    keywords = _extract_search_keywords(search_query)
    if search_query:
        ranked_candidates = sorted(
            recent_articles,
            key=lambda article: (
                _keyword_match_score(article, search_query, keywords),
                article.get("timestamp", 0),
            ),
            reverse=True,
        )
    else:
        ranked_candidates = sorted(recent_articles, key=lambda article: article.get("timestamp", 0), reverse=True)

    candidate_pool = ranked_candidates[:ARTICLE_AGENT_POOL_SIZE]
    if search_query.strip():
        candidate_pool = enrich_articles_for_keyword_search(candidate_pool, search_query=search_query)
        candidate_pool = sorted(
            candidate_pool,
            key=lambda article: (
                _keyword_match_score(article, search_query, keywords),
                article.get("timestamp", 0),
            ),
            reverse=True,
        )
    shortlisted = _select_articles_with_finder_agent(candidate_pool, top_n=top_n, search_query=search_query)

    if not search_query.strip():
        return shortlisted[:top_n]

    approved = _evaluate_articles_with_keyword_agent(shortlisted, search_query=search_query, top_n=top_n)
    if len(approved) >= top_n:
        return approved[:top_n]

    remaining_candidates = [article for article in candidate_pool if article["url"] not in {a["url"] for a in shortlisted}]
    if remaining_candidates:
        approved.extend(
            _evaluate_articles_with_keyword_agent(
                remaining_candidates,
                search_query=search_query,
                top_n=max(top_n - len(approved), 0),
            )
        )

    deduped = []
    seen_urls = set()
    for article in approved:
        if article["url"] and article["url"] not in seen_urls:
            deduped.append(article)
            seen_urls.add(article["url"])

    return deduped[:top_n]


def enrich_articles_for_keyword_search(candidate_articles: list[dict], search_query: str) -> list[dict]:
    if not search_query.strip() or not candidate_articles:
        return candidate_articles

    enriched_articles = [dict(article) for article in candidate_articles]
    indexes_to_fetch = [
        index
        for index, article in enumerate(enriched_articles)
        if article.get("url") and len(article.get("full_text", "")) < 240
    ][: min(len(enriched_articles), ARTICLE_FINDER_SHORTLIST_SIZE)]

    if not indexes_to_fetch:
        return enriched_articles

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(indexes_to_fetch))) as executor:
        future_map = {
            executor.submit(try_enrich_article, enriched_articles[index]["url"]): index
            for index in indexes_to_fetch
        }
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            try:
                full_text = future.result()
            except Exception:
                full_text = None
            if full_text:
                enriched_articles[index]["full_text"] = full_text

    return enriched_articles


def fetch_all_news(top_n: int = 5, selected_sources: list[str] = None, search_query: str = "") -> tuple[list[dict], str]:
    """
    Fetch top news from selected sources,
    then pick the top N articles combined with a fair mix.
    """
    if selected_sources is None:
        selected_sources = ["Deutsche Welle", "Tagesschau"]

    search_mode = bool(search_query.strip())
    dw_feed_urls = [DW_FEEDS[0]] if search_mode else DW_FEEDS
    ts_feed_urls = [TAGESSCHAU_FEEDS[0]] if search_mode else TAGESSCHAU_FEEDS

    # Fetch more items upfront to allow for search filtering
    def _fetch_concurrently(feed_urls, source_name):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_parse_feed, f, source_name, ARTICLES_PER_FEED) for f in feed_urls]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:
                    continue
        
        # Deduplicate
        seen = set()
        unique = []
        for a in results:
            dedupe_key = a["url"] or f"{a['source']}::{a['title'].lower()}"
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                unique.append(a)
        return sorted(unique, key=lambda article: article.get("timestamp", 0), reverse=True)

    dw_articles = _fetch_concurrently(dw_feed_urls, "Deutsche Welle") if "Deutsche Welle" in selected_sources else []
    ts_articles = _fetch_concurrently(ts_feed_urls, "Tagesschau") if "Tagesschau" in selected_sources else []

    # Interleave to get a balanced mix of both sources
    all_articles = []
    max_len = max(len(dw_articles), len(ts_articles), 1)
    for i in range(max_len):
        if i < len(dw_articles):
            all_articles.append(dw_articles[i])
        if i < len(ts_articles):
            all_articles.append(ts_articles[i])

    # Let the finder and evaluator agents finalize the latest relevant set
    all_articles = curate_articles_with_agents(all_articles, top_n=top_n, search_query=search_query)

    if search_mode and not all_articles:
        dw_articles = _fetch_concurrently(DW_FEEDS, "Deutsche Welle") if "Deutsche Welle" in selected_sources else []
        ts_articles = _fetch_concurrently(TAGESSCHAU_FEEDS, "Tagesschau") if "Tagesschau" in selected_sources else []

        fallback_articles = []
        max_len = max(len(dw_articles), len(ts_articles), 1)
        for i in range(max_len):
            if i < len(dw_articles):
                fallback_articles.append(dw_articles[i])
            if i < len(ts_articles):
                fallback_articles.append(ts_articles[i])

        all_articles = curate_articles_with_agents(fallback_articles, top_n=top_n, search_query=search_query)

    # Build a combined text block for the LLM
    text_parts = []
    for i, a in enumerate(all_articles, 1):
        text_parts.append(
            f"[{i}] ({a['source']}) {a['title']}\n{a['description']}\n{a.get('full_text', '')}"
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



# MULTI-AGENT SYSTEM (Journalist → Lexicographer → Grammarian)         

class AgentState(TypedDict):
    combined_text: str
    target_level: str
    headline: str
    summary: str
    bullet_points: List[str]
    vocabulary: List[VocabularyItem]
    grammar_spotlights: List[GrammarSpotlight]
    result: Optional[NewsBriefing]

class JournalistOutput(BaseModel):
    headline: str = Field(description="A short daily briefing headline")
    summary: str = Field(description="Comprehensive summary covering major stories (4-6 paragraphs)")
    bullet_points: List[str] = Field(description="5-7 key points across all stories")

class LexicographerOutput(BaseModel):
    vocabulary: List[VocabularyItem] = Field(description="Exactly 10 vocabulary words characteristic of the CEFR level, with English translation and German example")

class GrammarianOutput(BaseModel):
    grammar_spotlights: List[GrammarSpotlight] = Field(description="Exactly 5 grammar rule spotlights used in the summary, with examples from the text")

def _get_llm():
    # Try Streamlit Cloud secrets first, then local .env
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "Local: Add it to your .env file.\n"
            "Streamlit Cloud: Add it in App Settings → Secrets.\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)

def journalist_node(state: AgentState):
    llm = _get_llm().with_structured_output(JournalistOutput)
    prompt = f"""You are an expert German Journalist and Language Teacher. 
Write a unified news briefing based on the provided articles.
Target CEFR Level: {state['target_level']}

## Guidelines:
- Write a flowing, multi-paragraph summary (4-6 paragraphs) covering all major facts.
- Extract 5-7 key bullet points mapping the main events.
- B1: Use Perfekt, 'man', max 15 words/sentence. 
- B2: Use Passiv, complex connectors, max 25 words/sentence.
- C1: Use Nominalstil, Konjunktiv I, no length limit.

Articles:
{state['combined_text']}
"""
    output = llm.invoke(prompt)
    return {"headline": output.headline, "summary": output.summary, "bullet_points": output.bullet_points}

def lexicographer_node(state: AgentState):
    llm = _get_llm().with_structured_output(LexicographerOutput)
    prompt = f"""You are an expert German Lexicographer for CEFR Level {state['target_level']}.
Analyze the following news summary and extract exactly 10 important vocabulary words appropriate for {state['target_level']}.
For each word, provide the English translation and a German example sentence from a news context.

Summary:
{state['summary']}
"""
    output = llm.invoke(prompt)
    return {"vocabulary": output.vocabulary}

def grammarian_node(state: AgentState):
    llm = _get_llm().with_structured_output(GrammarianOutput)
    prompt = f"""You are an expert German Grammarian for CEFR Level {state['target_level']}.
Analyze the following news summary and highlight exactly 5 different grammar rules used in the text.
Provide the rule name, a clear explanation in German, and the exact sentence from the summary where it is used.
Make sure to explain rules suitable for {state['target_level']} (e.g. Perfekt/Nebensatz for B1, Passiv/Konjunktiv II for B2, Nominalstil/Konjunktiv I for C1).

Summary:
{state['summary']}
"""
    output = llm.invoke(prompt)
    return {"grammar_spotlights": output.grammar_spotlights}

def compiler_node(state: AgentState):
    result = NewsBriefing(
        headline=state["headline"],
        summary=state["summary"],
        bullet_points=state["bullet_points"],
        vocabulary=state["vocabulary"],
        grammar_spotlights=state["grammar_spotlights"]
    )
    return {"result": result}

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("journalist", journalist_node)
    graph.add_node("lexicographer", lexicographer_node)
    graph.add_node("grammarian", grammarian_node)
    graph.add_node("compiler", compiler_node)
    
    # Sequential Pipeline
    graph.add_edge(START, "journalist")
    graph.add_edge("journalist", "lexicographer")
    graph.add_edge("lexicographer", "grammarian")
    graph.add_edge("grammarian", "compiler")
    graph.add_edge("compiler", END)
    
    return graph.compile()

def generate_briefing(combined_text: str, level: str) -> NewsBriefing:
    """Run the Multi-Agent pipeline."""
    graph = build_graph()
    final = graph.invoke({
        "combined_text": combined_text,
        "target_level": level,
        "headline": "",
        "summary": "",
        "bullet_points": [],
        "vocabulary": [],
        "grammar_spotlights": [],
        "result": None
    })
    
    if final.get("result"):
        return final["result"]
        
    raise RuntimeError("Multi-Agent pipeline failed to produce a final briefing.")


# STREAMLIT UI                                                         

st.set_page_config(
    page_title="🇩🇪 German News Agent",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        color: var(--text-color);
        text-transform: uppercase;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Level badges - Theme aware */
    .level-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border: 1px solid var(--text-color);
        opacity: 0.9;
    }
    .level-B1 { background: transparent; color: var(--text-color); }
    .level-B2 { background: var(--secondary-background-color); color: var(--text-color); border: 1px solid transparent; }
    .level-C1 { background: var(--text-color); color: var(--background-color); }

    /* Glass card - Theme aware */
    .glass-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 4px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: border-color 0.2s ease;
    }
    .glass-card:hover { border-color: var(--primary-color); }
    .glass-card h3 {
        margin-top: 0;
        color: var(--text-color);
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Vocab table */
    .vocab-table {
        width: 100%;
        border-collapse: collapse;
    }
    .vocab-table th {
        text-align: left;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.8rem 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .vocab-table td {
        padding: 1rem;
        color: var(--text-color);
        opacity: 0.9;
        font-size: 0.95rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    .vocab-table tr td:first-child {
        font-weight: 700;
        opacity: 1;
    }

    /* Bullet list */
    .bullet-item {
        border-left: 3px solid var(--primary-color);
        padding: 1rem 1.5rem;
        margin-bottom: 0.8rem;
        background: var(--background-color);
        color: var(--text-color);
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Grammar card */
    .grammar-card {
        background: var(--background-color);
        color: var(--text-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 4px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        word-wrap: break-word;
    }
    .grammar-card h3 { 
        color: var(--text-color) !important; 
        margin-top: 0; 
        border-bottom: 1px solid rgba(128,128,128,0.2);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .grammar-rule { 
        color: var(--text-color) !important; 
        font-weight: 800; 
        font-size: 1.2rem; 
        text-transform: uppercase;
    }
    .grammar-explanation { 
        color: var(--text-color) !important; 
        opacity: 0.8;
        margin: 1rem 0; 
        line-height: 1.6; 
        flex-grow: 1;
    }
    .grammar-example {
        background: var(--secondary-background-color);
        border-radius: 2px;
        padding: 1.2rem;
        color: var(--text-color) !important;
        font-style: italic;
        font-size: 0.95rem;
        border-left: 4px solid var(--text-color);
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
        border: 1px solid rgba(128,128,128,0.3);
    }
    .source-dw { background: var(--text-color); color: var(--background-color); }
    .source-ts { background: transparent; color: var(--text-color); }

    .meta-bar {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1rem 1.5rem;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 4px;
        margin-bottom: 2rem;
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.85rem;
    }
    .meta-bar strong { color: var(--text-color); opacity: 1; }

    .graph-info {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        color: var(--text-color);
        padding: 1rem;
        font-size: 0.8rem;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
    }

    /* Remove stSidebar explicit background */

    /* Ensure primary buttons match Streamlit natively */
    div.stButton > button[kind="primary"] {
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }

    /* Restored visibility for navigation elements */
    #MainMenu {visibility: visible;}
    footer {visibility: visible;}
    header {visibility: visible;}
</style>
""", unsafe_allow_html=True)

#  Header 
st.markdown("""
<div class="main-header">
    <h1>🇩🇪 German News Agent</h1>
    <p>Your daily German news briefing from DW & Tagesschau — at your CEFR level</p>
</div>
""", unsafe_allow_html=True)

# Sidebar 
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
        st.info(f"Will auto-refresh every {refresh_minutes} min")

    st.markdown("---")
    st.markdown("### Sources")
    selected_sources = st.multiselect(
        "Select preferred news sources:",
        options=["Deutsche Welle", "Tagesschau"],
        default=["Deutsche Welle", "Tagesschau"],
    )
    if not selected_sources:
        st.warning("Please select at least one source.")



    st.markdown("---")
    st.markdown(
        "<div style='color:#8892b0; font-size:0.78rem; text-align:center;'>"
        "Powered by OpenAI · DW · Tagesschau</div>",
        unsafe_allow_html=True,
    )

# Session State 
if "briefing" not in st.session_state:
    st.session_state.briefing = None
if "articles" not in st.session_state:
    st.session_state.articles = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "fetch_time" not in st.session_state:
    st.session_state.fetch_time = None
if "click_count" not in st.session_state:
    st.session_state.click_count = 0
if "translated_summary" not in st.session_state:
    st.session_state.translated_summary = None

# Auto-refresh logic 
needs_auto_refresh = False
if auto_refresh and st.session_state.last_refresh:
    elapsed = (get_german_time() - st.session_state.last_refresh).total_seconds()
    if elapsed >= refresh_minutes * 60:
        needs_auto_refresh = True

# Main Action
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    search_query = st.text_input("🔍 Search topics (e.g. Politik, Wirtschaft) — Optional", value="")
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True) # Adds tiny space
    remaining_clicks = 5 - st.session_state.click_count
    
    if st.session_state.click_count < 5:
        get_summary = st.button(
            f"Get Latest Summary",
            use_container_width=True,
            type="primary",
            help="Fetches latest news from DW & Tagesschau and creates a combined briefing",
        )
    else:
        st.button("Limit Reached (5/5)", use_container_width=True, disabled=True)
        st.error("You have reached the limit of 5 summaries for this session.")
        get_summary = False

should_generate = (get_summary or needs_auto_refresh) and st.session_state.click_count < 5

if should_generate:
    if not selected_sources:
        st.error("Cannot fetch news: No sources were selected in the sidebar.")
        st.stop()

    # Increment counter
    if get_summary:
        st.session_state.click_count += 1
    # Reset translation on new fetch
    st.session_state.translated_summary = None
    # Step 1: Fetch
    with st.spinner(" Fetching feeds and running Finder, Evaluator, Journalist, Lexicographer, and Grammarian Agents..."):
        try:
            articles, combined_text = fetch_all_news(top_n=5, selected_sources=selected_sources, search_query=search_query)
            st.session_state.articles = articles
            st.session_state.fetch_time = get_german_time()
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            st.stop()

    if not articles:
        st.warning("No matching articles were found. Try again later or broaden the search keyword.")
        st.stop()

    # Step 2: Generate briefing
    with st.spinner(
        f" Creating {target_level}-level briefing from {len(articles)} articles "
        
    ):
        try:
            briefing = generate_briefing(combined_text, target_level)
            st.session_state.briefing = briefing
            st.session_state.last_refresh = get_german_time()
        except Exception as e:
            st.error(f"Error generating briefing: {e}")
            st.stop()

# Display Briefing 
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
        f"<div class='glass-card'><h2 style='color:var(--text-color); margin:0;'>"
        f"{briefing.headline}</h2></div>",
        unsafe_allow_html=True,
    )

    # Row 1: Summary + Key Points
    col_left, col_right = st.columns([3, 2])

    with col_left:
        formatted_summary = briefing.summary.replace("\n\n", "</p><p style='color:var(--text-color); opacity:0.9; line-height:1.8; margin-top:0.8rem;'>")
        formatted_summary = formatted_summary.replace("\n", "<br>")
        st.markdown(
            f"<div class='glass-card' style='margin-bottom: 1rem;'><h3>News Briefing</h3>"
            f"<p style='color:var(--text-color); opacity:0.9; line-height:1.8; font-size:1rem;'>"
            f"{formatted_summary}</p></div>",
            unsafe_allow_html=True,
        )

        # Translate Button & Display 
        if not st.session_state.translated_summary:
            if st.button("Translate to English", use_container_width=True):
                with st.spinner("Translating summary..."):
                    llm = _get_llm()
                    from langchain_core.messages import SystemMessage, HumanMessage
                    resp = llm.invoke([
                        SystemMessage(content="You are a professional translator. Translate the following German news summary into English. Maintain a precise and neutral journalistic tone."),
                        HumanMessage(content=briefing.summary)
                    ])
                    st.session_state.translated_summary = resp.content
                    st.rerun()
        else:
            eng_summary = st.session_state.translated_summary.replace("\n\n", "</p><p style='color:var(--text-color); opacity:0.8; line-height:1.8; margin-top:0.8rem;'>")
            eng_summary = eng_summary.replace("\n", "<br>")
            st.markdown(
                f"<div class='glass-card' style='background: var(--background-color); border-left: 4px solid var(--primary-color); padding: 1.5rem; margin-bottom: 1rem;'>"
                f"<h4 style='color: var(--primary-color); margin-top: 0; margin-bottom: 1rem;'>🇬🇧 English Translation</h4>"
                f"<p style='color:var(--text-color); opacity:0.8; line-height:1.8; font-size:0.95rem; margin: 0;'>"
                f"{eng_summary}</p></div>",
                unsafe_allow_html=True,
            )
            if st.button("Hide Translation", use_container_width=True):
                st.session_state.translated_summary = None
                st.rerun()

    with col_right:
        # Key Points
        kp_html = "<div class='glass-card'><h3>Key Points</h3>"
        for bp in briefing.bullet_points:
            kp_html += f"<div class='bullet-item'>{bp}</div>"
        kp_html += "</div>"
        st.markdown(kp_html, unsafe_allow_html=True)

        # Sources (compact)
        src_html = "<div class='glass-card'><h3>Sources</h3>"
        for a in articles[:10]:
            tag_class = "source-dw" if a["source"] == "Deutsche Welle" else "source-ts"
            src_html += (
                f"<div style='margin-bottom:0.5rem;'>"
                f"<span class='source-tag {tag_class}'>{a['source']}</span>"
                f"<a href='{a['url']}' target='_blank' style='color:var(--primary-color); "
                f"text-decoration:none; font-size:0.85rem;'>{a['title']}</a></div>"
            )
        src_html += "</div>"
        st.markdown(src_html, unsafe_allow_html=True)

    # Row 2: Vocabulary (full width) 
    st.markdown("---")
    vocab_rows = ""
    for v in briefing.vocabulary:
        vocab_rows += (
            f"<tr><td>{v.word}</td><td>{v.translation}</td>"
            f"<td style='font-style:italic; color:var(--text-color); opacity:0.7;'>{v.example}</td></tr>"
        )
    st.markdown(
        f"<div class='glass-card'><h3>Vocabulary — {target_level} Level ({len(briefing.vocabulary)} Words)</h3>"
        f"<table class='vocab-table'>"
        f"<tr><th>Wort</th><th>Translation</th><th>Beispielsatz</th></tr>"
        f"{vocab_rows}</table></div>",
        unsafe_allow_html=True,
    )

    # Row 3: Grammar Spotlights (5 cards in 2 rows) 
    st.markdown("---")

    # Row A: first 3 grammar spotlights
    row_a = briefing.grammar_spotlights[:3]
    cols_a = st.columns(3)
    for i, gs in enumerate(row_a):
        with cols_a[i]:
            st.markdown(
                f"<div class='grammar-card'>"
                f"<h3>Grammar {i+1}</h3>"
                f"<div class='grammar-rule'>{gs.rule_name}</div>"
                f"<div class='grammar-explanation'>{gs.explanation}</div>"
                f"<div class='grammar-example'>„{gs.example_from_text}“</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Row B: remaining grammar spotlights
    row_b = briefing.grammar_spotlights[3:]
    if row_b:
        cols_b = st.columns(len(row_b))
        for j, gs in enumerate(row_b):
            with cols_b[j]:
                st.markdown(
                    f"<div class='grammar-card'>"
                    f"<h3>Grammar {j+4}</h3>"
                    f"<div class='grammar-rule'>{gs.rule_name}</div>"
                    f"<div class='grammar-explanation'>{gs.explanation}</div>"
                    f"<div class='grammar-example'>„{gs.example_from_text}“</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;"> 🇩🇪 </div>
        <h2 style="color: var(--text-color); font-weight: 300; margin-bottom: 1rem;">
            Your Daily German News Briefing
        </h2>
        <p style="color: var(--text-color); opacity: 0.8; max-width: 550px; margin: 0 auto; line-height: 1.7;">
            Click <strong style="color: var(--text-color);">Get Latest Summary</strong> to fetch the latest
            news from <strong style="color: var(--primary-color);">Deutsche Welle</strong> and
            <strong style="color: var(--primary-color);">Tagesschau</strong>, combined into a single
            comprehensive briefing at your CEFR level with vocabulary and grammar highlights.
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem;">
            <span class="level-badge level-B1">B1 Intermediate</span>
            <span class="level-badge level-B2">B2 Upper-Intermediate</span>
            <span class="level-badge level-C1">C1 Advanced</span>
        </div>
        <div style="margin-top: 2rem; color: var(--text-color); opacity: 0.6; font-size: 0.85rem;">
             Enable auto-refresh in the sidebar for scheduled updates
        </div>
        <div style="margin-top: 0.5rem; color: var(--primary-color); font-size: 0.8rem;">
             LangGraph Multi-Agent Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh timer (triggers rerun at the right time)
if auto_refresh and st.session_state.last_refresh:
    elapsed = (get_german_time() - st.session_state.last_refresh).total_seconds()
    remaining = max(0, refresh_minutes * 60 - elapsed)
    if remaining > 0:
        time.sleep(min(remaining, 5))  # check every 5 seconds
        st.rerun()
