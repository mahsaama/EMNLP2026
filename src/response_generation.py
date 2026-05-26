import os
import json
import re
import ast
import hashlib
import logging
import warnings
from tqdm import tqdm
import pandas as pd
import requests
from urllib.parse import urlparse, unquote
import plotly.graph_objects as go
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import asyncio
import fitz
from playwright.async_api import async_playwright
from readability import Document
from rouge_score import rouge_scorer
import numpy as np
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluator_prompts import SYSTEM_PROMPT_RESP_SYNT, USER_PROMPT_RESP_SYNT, SYSTEM_PROMPT_CLAIM_EXTRACTION, USER_PROMPT_CLAIM_EXTRACTION


load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

pio.defaults.mathjax = None
from utils import *
from data_utils import *
from paper import with_paper_style, styler
from data_extraction import load_web_data_from_file, load_whole_data_from_file

CONF = "emnlp/response_generation"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DIRECT_API_DOMAINS = {"wikipedia.org"}
SKIP_REQUESTS_DOMAINS = {"politico.com", "reuters.com"}
REQUEST_TIMEOUT = int(os.getenv("ARTICLE_REQUEST_TIMEOUT", "12"))
WIKIPEDIA_TIMEOUT = int(os.getenv("ARTICLE_WIKIPEDIA_TIMEOUT", "15"))
PLAYWRIGHT_GOTO_TIMEOUT = int(os.getenv("ARTICLE_PLAYWRIGHT_GOTO_TIMEOUT", "20000"))
PLAYWRIGHT_NETWORKIDLE_TIMEOUT = int(
    os.getenv("ARTICLE_PLAYWRIGHT_NETWORKIDLE_TIMEOUT", "4000")
)
PLAYWRIGHT_FALLBACK_TIMEOUT = float(
    os.getenv("ARTICLE_PLAYWRIGHT_FALLBACK_TIMEOUT", "60")
)
URL_FETCH_TIMEOUT = float(os.getenv("ARTICLE_URL_FETCH_TIMEOUT", "60"))
URL_FETCH_CHECKPOINT_EVERY = int(os.getenv("ARTICLE_URL_CHECKPOINT_EVERY", "100"))
RESPONSE_URLS_CONTENT_PATH = (
    f"{OUTPUT_PATH}/metadata/response_and_sources_url_content.json"
)
RESPONSE_SOURCE_EFFECT_EVALUATIONS_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_effect_evaluations"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_JUDGE_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_bert"
)
EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES = {
    "Claude": {
        "bert": (
            f"{OUTPUT_PATH}/claude/metadata/"
            "response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"/{OUTPUT_PATH}/claude/metadata/"
            "response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
    "Grok": {
        "bert": (
            f"{OUTPUT_PATH}/grok/metadata/"
            "response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"{OUTPUT_PATH}/grok/metadata/"
            "response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
    "DeepSeek": {
        "bert": (
            f"{OUTPUT_PATH}/outputs/deepseek/metadata/"
            "response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"{OUTPUT_PATH}/outputs/deepseek/metadata/"
            "response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
}
EXTERNAL_PLATFORM_ORDER = ["OpenAI"] + list(
    EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys()
)

CITED_URL_VALIDITY_LABELS_PATH = (
    f"{OUTPUT_PATH}/metadata/cited_url_validity_labels.json"
)
BERT_NLI_MODEL_NAME = os.getenv("BERT_NLI_MODEL_NAME", "facebook/bart-large-mnli")
BERT_NLI_MAX_LENGTH = int(os.getenv("BERT_NLI_MAX_LENGTH", "512"))
NLI_JUDGE_CONTEXT_WINDOW_TOKENS = 128000
NLI_JUDGE_MAX_OUTPUT_TOKENS = 256
NLI_JUDGE_TOKEN_SAFETY_MARGIN = 2000
NLI_ESTIMATED_CHARS_PER_TOKEN = 3.0
CLAIM_EXTRACTION_MODEL = os.getenv("CLAIM_EXTRACTION_MODEL", "gpt-4o-mini")
CLAIM_EXTRACTION_MAX_INPUT_CHARS = int(
    os.getenv("CLAIM_EXTRACTION_MAX_INPUT_CHARS", "24000")
)
CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS = int(
    os.getenv("CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS", "1024")
)
CLAIM_EXTRACTION_CACHE_PATH = os.getenv(
    "CLAIM_EXTRACTION_CACHE_PATH",
    f"{OUTPUT_PATH}/metadata/response_source_claim_chunks_cache.json",
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def find_similarity(page_content, response):
    embeddings = model.encode([response, page_content])

    response_emb = embeddings[0]
    page_content_embs = embeddings[1:]

    scores = cosine_similarity(
        response_emb.reshape(1, -1), page_content_embs.reshape(1, -1)
    )[0]

    return float(scores.mean())

def extract_first_json_object(text):
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return {}
    return {}


def _coerce_claim_list(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["claims", "claim_list", "items", "sentences", "chunks"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _clean_claims(claims):
    cleaned_claims = []
    seen = set()
    for claim in claims:
        if isinstance(claim, dict):
            claim = (
                claim.get("claim")
                or claim.get("text")
                or claim.get("statement")
                or ""
            )
        claim = str(claim or "").strip(" -*\t\n")
        if len(claim) < 8 or not re.search(r"[A-Za-z]", claim):
            continue
        if claim in seen:
            continue
        seen.add(claim)
        cleaned_claims.append(claim)
    return cleaned_claims


def _claim_cache_key(text):
    text = str(text or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_claims_cache(cache_path=CLAIM_EXTRACTION_CACHE_PATH):
    if not cache_path or not os.path.exists(cache_path):
        return {}

    raw_cache = load_json(cache_path)
    if not isinstance(raw_cache, dict):
        logger.warning("Claim cache at %s is not a JSON object; ignoring.", cache_path)
        return {}

    normalized_cache = {}
    for key, value in raw_cache.items():
        cleaned_claims = _clean_claims(value if isinstance(value, list) else [])
        if cleaned_claims:
            normalized_cache[str(key)] = cleaned_claims
    return normalized_cache


def _save_claims_cache(claims_cache, cache_path=CLAIM_EXTRACTION_CACHE_PATH):
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    to_json(claims_cache, cache_path, indent=2)


def extract_claims_from_text(text):
    text = str(text or "").strip()
    if not text:
        return []

    prompt_text = text[:CLAIM_EXTRACTION_MAX_INPUT_CHARS]
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_EXTRACTION},
        {
            "role": "user",
            "content": USER_PROMPT_CLAIM_EXTRACTION.format(text=prompt_text),
        },
    ]

    response_text = ""
    try:
        response = client.chat.completions.create(
            model=CLAIM_EXTRACTION_MODEL,
            messages=msg,
            max_tokens=CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Claim extraction failed: %s", e)
        return []

    parsed_payload = None
    try:
        parsed_payload = json.loads(response_text)
    except Exception:
        parsed_payload = extract_first_json_object(response_text)

    claims = _coerce_claim_list(parsed_payload)
    return _clean_claims(claims)


def _estimate_token_count(text):
    if not text:
        return 0
    return int(np.ceil(len(text) / NLI_ESTIMATED_CHARS_PER_TOKEN))

def _trim_nli_source_to_context(source_text, response_text):
    source_text = str(source_text or "")
    response_text = str(response_text or "")
    if not source_text:
        return source_text

    base_prompt_text = (
        str(SYSTEM_PROMPT_RESP_SYNT or "")
        + USER_PROMPT_RESP_SYNT.format(response_text=response_text, source="")
    )
    base_tokens = _estimate_token_count(base_prompt_text)
    source_token_budget = (
        NLI_JUDGE_CONTEXT_WINDOW_TOKENS
        - NLI_JUDGE_MAX_OUTPUT_TOKENS
        - NLI_JUDGE_TOKEN_SAFETY_MARGIN
        - base_tokens
    )
    if source_token_budget <= 0:
        return ""

    source_tokens = _estimate_token_count(source_text)
    if source_tokens <= source_token_budget:
        return source_text

    max_source_chars = int(source_token_budget * NLI_ESTIMATED_CHARS_PER_TOKEN)
    if max_source_chars <= 0:
        return ""
    return source_text[:max_source_chars]

def compute_nli_scores(premise, hypothesis):
    """Score whether a source text entails, contradicts, or is neutral to the response."""
    premise = _trim_nli_source_to_context(premise, hypothesis).strip()
    hypothesis = str(hypothesis or "").strip()

    msg = [
        {"role": "system", "content": SYSTEM_PROMPT_RESP_SYNT},
        {
            "role": "user",
            "content": USER_PROMPT_RESP_SYNT.format(
                response_text=hypothesis,
                source=premise,
            ),
        },
    ]
    text = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            max_tokens=256,
            temperature=0.0,
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(e)
        text = ""

    json_response = extract_first_json_object(text)
    return json_response


def extract_response_and_sources(web_df):
    outer_pattern = r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    inner_pattern = r"\ue202[A-Za-z]+(\d+)[A-Za-z]+(\d+)(?=\ue202|\ue201)"

    def _dedupe_cited_items(items):
        def _item_richness(item):
            score = 0
            for value in item.values():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                score += 1
            return score

        unique_items = []
        key_to_index = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            dedupe_key = (
                item.get("url", ""),
                item.get("ref_index", None),
                item.get("turn_index", None),
            )
            existing_index = key_to_index.get(dedupe_key)
            if existing_index is None:
                key_to_index[dedupe_key] = len(unique_items)
                unique_items.append(item)
                continue

            if _item_richness(item) > _item_richness(unique_items[existing_index]):
                unique_items[existing_index] = item
        return unique_items

    web_df["srcs_retrieved"] = [{}] * len(web_df)
    web_df["srcs_safe_urls"] = [{}] * len(web_df)
    web_df["srcs_cited"] = [{}] * len(web_df)
    web_df["asistant_response"] = [""] * len(web_df)

    for i, row in tqdm(web_df.iterrows()):
        msgs = json.loads(row["turn_msgs"])
        parts = msgs[-1].get("content", {}).get("parts", [])
        asistant_response = " ".join([str(p) for p in parts])
        srcs_retrieved = []
        srcs_safe_urls = []
        srcs_cited = []
        for msg in msgs:
            # retrieved
            retrieved = msg.get("metadata", {}).get("search_result_groups", [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    if url:
                        d = urlparse(entry["url"]).netloc.replace("www.", "")
                        srcs_retrieved.append(
                            {
                                "url": url,
                                "domain": d,
                                "title": entry.get("title", ""),
                                "ref_index": (
                                    entry.get("ref_id", {}).get("ref_index", None)
                                    if entry.get("ref_id", {})
                                    else None
                                ),
                                "turn_index": (
                                    entry.get("ref_id", {}).get("turn_index", None)
                                    if entry.get("ref_id", {})
                                    else None
                                ),
                                "snippet": entry["snippet"],
                            }
                        )

            retrieved = msg.get("metadata", {}).get("image_results", [])
            for ri, r in enumerate(retrieved):
                d = urlparse(r["url"]).netloc.replace("www.", "")
                srcs_retrieved.append(
                    {
                        "url": r["url"],
                        "domain": d,
                        "title": r.get("title", ""),
                        "ref_index": ri,
                    }
                )

            # safe urls
            safe_urls = msg.get("metadata", {}).get("safe_urls", [])
            for r in safe_urls:
                if r:
                    url = r.removesuffix("?utm_source=chatgpt.com").removesuffix(
                        "&utm_source=chatgpt.com"
                    )
                    d = urlparse(url).netloc.replace("www.", "")
                    srcs_safe_urls.append({"url": url, "domain": d})

            # cited
            cited = msg.get("metadata", {}).get("content_references", [])
            for r in cited:
                matched_text = r.get("matched_text", "").strip()
                if matched_text:
                    outer = re.search(outer_pattern, matched_text)
                    if outer:
                        found_refs = re.findall(inner_pattern, outer.group(0))
                        cited_turns = []
                        cited_ranks = []
                        for fr in found_refs:
                            cited_turns.append(int(fr[0]))
                            cited_ranks.append(int(fr[1]))

                        url = r.get("url", "")
                        if url:
                            url = url.removesuffix(
                                "?utm_source=chatgpt.com"
                            ).removesuffix("&utm_source=chatgpt.com")
                            d = urlparse(url).netloc.replace("www.", "")
                            srcs_cited.append(
                                {
                                    "url": url,
                                    "domain": d,
                                    "title": r.get("title"),
                                    "snippet": r.get("snippet"),
                                    "ref_index": cited_ranks[0],
                                    "turn_index": cited_turns[0],
                                }
                            )

                        if "fallback_items" in r and r["fallback_items"]:
                            keys_to_check = ["images", "fallback_items"]
                        else:
                            keys_to_check = ["images", "items"]

                        for key in keys_to_check:
                            items = r.get(key, [])
                            refs = r.get("refs", [])
                            if items:
                                for ii, item in enumerate(items):
                                    url = (
                                        item.get("url", "")
                                        .removesuffix("?utm_source=chatgpt.com")
                                        .removesuffix("&utm_source=chatgpt.com")
                                    )
                                    d = urlparse(url).netloc.replace("www.", "")
                                    if item.get("refs", []):
                                        ref = item.get("refs", [])[0]
                                    else:
                                        ref = refs[ii] if ii < len(refs) else {}
                                    if url:
                                        srcs_cited.append(
                                            {
                                                "url": url,
                                                "domain": d,
                                                "title": item.get("title", ""),
                                                "snippet": item.get("snippet", ""),
                                                "ref_index": ref.get("ref_index", None),
                                                "turn_index": ref.get(
                                                    "turn_index", None
                                                ),
                                            }
                                        )

        web_df.at[i, "srcs_retrieved"] = srcs_retrieved
        web_df.at[i, "srcs_safe_urls"] = srcs_safe_urls
        web_df.at[i, "srcs_cited"] = _dedupe_cited_items(srcs_cited)
        web_df.at[i, "asistant_response"] = asistant_response

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)

    web_df.to_csv(
        f"{OUTPUT_PATH}/metadata/response_and_sources.csv",
        index=False,
    )
    web_df.to_pickle(f"{OUTPUT_PATH}/metadata/response_and_sources.pkl")


def clean_html_for_readability(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text


def extract_clean_text_from_html(html):
    html = clean_html_for_readability(html)
    if not html:
        return ""

    try:
        doc = Document(html)
        clean_html = doc.summary()
    except Exception:
        clean_html = html

    soup = BeautifulSoup(clean_html, "html.parser")
    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    if len(clean_text.strip()) < 200:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)

    return clean_text


def get_article_text(url):
    logger.info("Fetching URL with requests: %s", url)
    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    content_type = (response.headers.get("content-type") or "").lower()

    if (
        "application/pdf" in content_type
        or url.lower().endswith(".pdf")
        or "/bitstream/" in url.lower()
        or response.content[:4] == b"%PDF"
    ):
        logger.info("Detected PDF content from requests path: %s", url)
        return extract_text_from_pdf_bytes(response.content)

    response.encoding = response.encoding or response.apparent_encoding
    return extract_clean_text_from_html(response.text)


def get_article_text_wikipedia(url):
    logger.info("Fetching URL with Wikipedia API: %s", url)
    parsed = urlparse(url)
    title = unquote(parsed.path.removeprefix("/wiki/")).strip()
    if not title:
        raise ValueError(f"Could not parse Wikipedia title from URL: {url}")

    api_url = f"{parsed.scheme}://{parsed.netloc}/w/api.php"
    response = requests.get(
        api_url,
        headers=HEADERS,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        timeout=WIKIPEDIA_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "").strip()
        if extract:
            return extract
    raise ValueError(f"Wikipedia API returned no extract for {url}")


def get_domain(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def extract_text_from_pdf_bytes(pdf_bytes):
    if not pdf_bytes:
        return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.warning("Failed to open PDF bytes with PyMuPDF: %s", e)
        return ""

    try:
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    finally:
        doc.close()


async def fetch_url_content(url, browser=None, url_cache=None):
    if url_cache is not None and url in url_cache:
        logger.info("URL cache hit: %s", url)
        return url_cache[url]

    domain = get_domain(url)

    if any(domain.endswith(suffix) for suffix in DIRECT_API_DOMAINS):
        try:
            content = await asyncio.to_thread(get_article_text_wikipedia, url)
            logger.info("Wikipedia API path succeeded: %s", url)
            if url_cache is not None:
                url_cache[url] = content
            return content
        except Exception as e:
            logger.warning("Wikipedia API path failed for %s: %s", url, e)

    if not any(domain.endswith(suffix) for suffix in SKIP_REQUESTS_DOMAINS):
        try:
            content = await asyncio.to_thread(get_article_text, url)
            logger.info("Requests path succeeded: %s", url)
            if url_cache is not None:
                url_cache[url] = content
            return content
        except Exception as e:
            logger.warning("Requests path failed for %s: %s", url, e)
    else:
        logger.info("Skipping requests fast path for domain %s: %s", domain, url)

    try:
        content = await asyncio.wait_for(
            get_article_text_planB(url, browser=browser),
            timeout=PLAYWRIGHT_FALLBACK_TIMEOUT,
        )
        logger.info("Playwright path succeeded: %s", url)
        if url_cache is not None:
            url_cache[url] = content
        return content
    except asyncio.TimeoutError:
        logger.warning(
            "Playwright path timed out after %.1fs for %s",
            PLAYWRIGHT_FALLBACK_TIMEOUT,
            url,
        )
    except Exception as e:
        logger.warning("Playwright path failed for %s: %s", url, e)

    logger.warning("All extraction paths failed for %s", url)
    if url_cache is not None:
        url_cache[url] = ""
    return ""


COOKIE_BUTTON_TEXTS: List[str] = [
    "accept",
    "accept all",
    "agree",
    "agree to all",
    "allow all",
    "allow cookies",
    "consent",
    "continue",
    "i agree",
    "ok",
    "okay",
]

PAYWALL_BUTTON_TEXTS: List[str] = [
    "continue reading",
    "no thanks",
    "not now",
    "close",
    "dismiss",
    "maybe later",
]


async def accept_cookie_banners(page):
    # Try a few broad strategies because cookie walls vary heavily across sites.
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label*='Accept' i]",
        "button[title*='Accept' i]",
        "[id*='accept' i]",
        "[class*='accept' i]",
        "[data-testid*='accept' i]",
        "[data-test*='accept' i]",
    ]

    for frame in page.frames:
        for selector in selectors:
            try:
                locator = frame.locator(selector).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass

        for text in COOKIE_BUTTON_TEXTS:
            try:
                locator = frame.get_by_role(
                    "button", name=re.compile(rf"^{re.escape(text)}$", re.I)
                ).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass

            try:
                locator = frame.get_by_text(re.compile(rf"\b{re.escape(text)}\b", re.I)).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass


async def dismiss_paywall_overlays(page):
    selectors = [
        "[aria-label='Close']",
        "button[aria-label*='close' i]",
        "[data-testid*='close' i]",
        "[class*='close' i]",
        "[class*='modal' i]",
        "[class*='overlay' i]",
        "[class*='paywall' i]",
        "[id*='modal' i]",
        "[id*='overlay' i]",
        "[id*='paywall' i]",
    ]

    for frame in page.frames:
        for text in PAYWALL_BUTTON_TEXTS:
            try:
                locator = frame.get_by_role(
                    "button", name=re.compile(rf"\b{re.escape(text)}\b", re.I)
                ).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1000)
                    return
            except Exception:
                pass

        for selector in selectors:
            try:
                locator = frame.locator(selector).first
                if await locator.is_visible(timeout=1000):
                    await locator.evaluate(
                        """node => {
                            node.remove();
                            document.body.style.overflow = 'auto';
                            document.documentElement.style.overflow = 'auto';
                        }"""
                    )
                await page.wait_for_timeout(500)
            except Exception:
                pass

    try:
        await page.evaluate(
            """
            () => {
                const patterns = /(paywall|gateway|modal|overlay|subscribe)/i;
                for (const node of Array.from(document.querySelectorAll('div,section,aside'))) {
                    const attrs = [node.id || '', node.className || '', node.getAttribute('data-testid') || ''].join(' ');
                    if (patterns.test(attrs)) {
                        node.remove();
                    }
                }
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            }
            """
        )
    except Exception:
        pass


async def extract_text_from_live_dom(page):
    article_selectors = [
        "article",
        "main article",
        "[data-testid='ArticleBodyWrapper']",
        "[data-testid*='article-body' i]",
        "[class*='article-body' i]",
        "[class*='ArticleBody' i]",
        "main",
    ]

    for selector in article_selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible(timeout=1000):
                text = await locator.inner_text(timeout=3000)
                if text and len(text.strip()) > 300:
                    logger.info("Extracted content from live DOM selector %s", selector)
                    return text.strip()
        except Exception:
            pass

    return ""


async def download_pdf_text(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Capture the main response
        response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        if response is None:
            await browser.close()
            raise ValueError("No response")

        content_type = response.headers.get("content-type", "")

        if "application/pdf" not in content_type:
            await browser.close()
            raise ValueError(f"Blocked or not PDF. Content-Type: {content_type}")

        pdf_bytes = await response.body()
        await browser.close()

    return extract_text_from_pdf_bytes(pdf_bytes)


async def get_article_text_planB(url, browser=None):
    logger.info("Fetching URL with Playwright fallback: %s", url)
    if ".pdf" in url.lower() or "/bitstream/" in url.lower():
        return await download_pdf_text(url)

    if browser is None:
        async with async_playwright() as p:
            owned_browser = await p.chromium.launch(
                headless=True, args=["--disable-blink-features=AutomationControlled"]
            )
            try:
                return await get_article_text_planB(url, browser=owned_browser)
            finally:
                await owned_browser.close()

    context = await browser.new_context(
        user_agent=HEADERS["User-Agent"],
        locale="en-US",
        extra_http_headers=HEADERS,
        java_script_enabled=True,
        ignore_https_errors=True,
        viewport={"width": 1440, "height": 1600},
    )

    try:
        page = await context.new_page()
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            """
        )

        await page.goto(
            url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_GOTO_TIMEOUT
        )
        await page.wait_for_timeout(1000)

        try:
            await accept_cookie_banners(page)
        except Exception:
            pass

        try:
            await dismiss_paywall_overlays(page)
        except Exception:
            pass

        try:
            await page.wait_for_load_state(
                "networkidle", timeout=PLAYWRIGHT_NETWORKIDLE_TIMEOUT
            )
        except Exception:
            pass

        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)
        except Exception:
            pass

        live_text = await extract_text_from_live_dom(page)
        content = await page.content()
    finally:
        await context.close()

    if live_text:
        return live_text

    return extract_clean_text_from_html(content)



def _load_response_source_similarity_input():
    pkl_path = f"{OUTPUT_PATH}/metadata/response_and_sources.pkl"
    csv_path = f"{OUTPUT_PATH}/metadata/response_and_sources.csv"

    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        if not os.path.exists(csv_path):
            raise
        logger.warning(
            "Failed to load %s, falling back to %s: %s",
            pkl_path,
            csv_path,
            e,
        )
        df = pd.read_csv(csv_path)

        def _parse_source_list(value):
            if isinstance(value, list):
                return value
            if not isinstance(value, str) or not value.strip():
                return []
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
            return parsed if isinstance(parsed, list) else []

        for source_col in ["srcs_retrieved", "srcs_safe_urls", "srcs_cited"]:
            if source_col in df.columns:
                df[source_col] = df[source_col].apply(_parse_source_list)

    selected_topics = ["Science", "Health", "Politics & History"]
    random_state = 42
    image_url_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".tif",
        ".tiff",
        ".avif",
        ".heic",
        ".heif",
        ".jfif",
        ".pjpeg",
        ".pjp",
        ".mov"
    }

    def _is_image_url(url):
        if not url:
            return False
        lower_url = url.lower()
        return any(lower_url.endswith(ext) for ext in image_url_extensions)

    def _is_bing_tse_url(url):
        if not url:
            return False
        parsed = urlparse(url.lower())
        host = parsed.netloc or ""
        return (
            parsed.scheme in {"http", "https"}
            and host.startswith("tse")
            and host.endswith(".mm.bing.net")
        )

    def _row_has_cited_or_retrieved_image_url(row):
        for source_col in ["srcs_cited", "srcs_retrieved"]:
            sources = row.get(source_col, [])
            if not isinstance(sources, list):
                continue
            for src in sources:
                if not isinstance(src, dict):
                    continue
                source_url = src.get("url", "")
                if _is_image_url(source_url) or _is_bing_tse_url(source_url):
                    return True
        return False

    df = (
        df[
            (df["language"] == "en")
            & (df["topic"].isin(selected_topics))
        ]
        .copy()
    )
    if "srcs_cited" in df.columns and "srcs_retrieved" in df.columns:
        has_image_url_mask = df.apply(_row_has_cited_or_retrieved_image_url, axis=1)
        df = df.loc[~has_image_url_mask].copy()

    # print(df["topic"].value_counts())

    sampled_frames = []
    for topic in selected_topics:
        topic_df = df[df["topic"] == topic].copy()
        if topic_df.empty:
            continue
        sample_n = min(100, len(topic_df))
        sampled_frames.append(topic_df.sample(n=sample_n, random_state=random_state))

    if not sampled_frames:
        return df.iloc[0:0].reset_index(drop=True)

    sampled_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .sort_values(["topic", "conv_id", "turn_id"], kind="stable")
        .reset_index(drop=True)
    )
    return sampled_df


def _iter_response_source_urls(row):
    for source_col in ["srcs_retrieved", "srcs_safe_urls", "srcs_cited"]:
        sources = row.get(source_col, [])
        if not isinstance(sources, list):
            continue
        for src in sources:
            if not isinstance(src, dict):
                continue
            url = src.get("url", "")
            if url:
                yield url


def _load_urls_content(urls_content_path=RESPONSE_URLS_CONTENT_PATH, required=True):
    if not os.path.exists(urls_content_path):
        if required:
            raise FileNotFoundError(
                f"URL content cache not found: {urls_content_path}. "
                "Run asyncio.run(extract_urls_content()) first."
            )
        return {}

    urls_content = load_json(urls_content_path)
    if urls_content is None:
        return {}
    if not isinstance(urls_content, dict):
        raise ValueError(f"Expected a JSON object at {urls_content_path}")

    return {
        str(url): content if isinstance(content, str) else ""
        for url, content in urls_content.items()
    }


async def extract_urls_content(
    urls_content_path=RESPONSE_URLS_CONTENT_PATH,
    force_refresh=False,
):
    df = _load_response_source_similarity_input()

    num_urls = 0
    unique_urls = set()
    for i, row in df.iterrows():
        row_urls = list(_iter_response_source_urls(row))
        num_urls += len(row_urls)
        unique_urls.update(row_urls)

    print(num_urls)
    print(len(unique_urls))
    print(len(df))

    url_cache = (
        {}
        if force_refresh
        else _load_urls_content(urls_content_path=urls_content_path, required=False)
    )
    checkpoint_every = max(1, URL_FETCH_CHECKPOINT_EVERY)
    processed_urls = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            for url in tqdm(sorted(unique_urls)):
                if force_refresh or url not in url_cache:
                    processed_urls += 1
                    try:
                        url_cache[url] = await asyncio.wait_for(
                            fetch_url_content(
                                url, browser=browser, url_cache=url_cache
                            ),
                            timeout=URL_FETCH_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "URL extraction timed out after %.1fs: %s",
                            URL_FETCH_TIMEOUT,
                            url,
                        )
                        url_cache[url] = ""
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        try:
                            browser = await p.chromium.launch(
                                headless=True,
                                args=["--disable-blink-features=AutomationControlled"],
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to relaunch browser after timeout for %s: %s",
                                url,
                                e,
                            )
                            browser = None
                    if processed_urls % checkpoint_every == 0:
                        logger.info(
                            "Checkpointing URL content cache after %s processed URLs to %s",
                            processed_urls,
                            urls_content_path,
                        )
                        to_json(url_cache, urls_content_path, indent=2)
        finally:
            if browser is not None:
                await browser.close()

    logger.info(
        "Writing final URL content cache with %s entries to %s",
        len(url_cache),
        urls_content_path,
    )
    to_json(url_cache, urls_content_path, indent=2)


def response_source_similarity(
    urls_content_path=RESPONSE_URLS_CONTENT_PATH,
):
    df = _load_response_source_similarity_input()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    num_urls = 0
    for i, row in df.iterrows():
        num_urls += len(list(_iter_response_source_urls(row)))

    print(num_urls)
    print(len(df))
    df["retrieved_sources_similarity"] = [{}] * len(df)
    df["safe_sources_similarity"] = [{}] * len(df)
    df["cited_sources_similarity"] = [{}] * len(df)
    urls_content = _load_urls_content(urls_content_path=urls_content_path)
    missing_urls = set()

    for i, row in tqdm(df.iterrows()):
        srcs_retrieved = row["srcs_retrieved"]
        srcs_safe_urls = row["srcs_safe_urls"]
        srcs_cited = row["srcs_cited"]
        asistant_response = row["asistant_response"]
        row_url_payloads = {}

        def get_similarity_payload(url):
            if url in row_url_payloads:
                return row_url_payloads[url]
            if url not in urls_content:
                missing_urls.add(url)
            content = urls_content.get(url, "")
            score = find_similarity(content, asistant_response)
            scores = scorer.score(content, asistant_response)
            nli_judge_response = compute_nli_scores(content, asistant_response)
            payload = {
                "similarity_score": score,
                "rouge_score": scores,
                "nli_judge": nli_judge_response,
                "content": content,
            }
            row_url_payloads[url] = payload
            return payload

        retrieved_urls_content = {}
        for src in srcs_retrieved:
            url = src["url"]
            retrieved_urls_content[url] = get_similarity_payload(url)

        safe_urls_content = {}
        for src in srcs_safe_urls:
            url = src["url"]
            safe_urls_content[url] = get_similarity_payload(url)

        cited_urls_content = {}
        for src in srcs_cited:
            url = src["url"]
            cited_urls_content[url] = get_similarity_payload(url)

        df.at[i, "retrieved_sources_similarity"] = retrieved_urls_content
        df.at[i, "safe_sources_similarity"] = safe_urls_content
        df.at[i, "cited_sources_similarity"] = cited_urls_content

    if missing_urls:
        logger.warning(
            "%s URLs were missing from %s and were scored with empty content",
            len(missing_urls),
            urls_content_path,
        )

    df.drop(
        columns=[
            "srcs_retrieved",
            "srcs_safe_urls",
            "srcs_cited",
            "thoughts",
            "openai_models",
            "user_msg_history",
            "interactions",
            "thinking",
        ],
        inplace=True,
    )
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.csv",
        index=False,
    )
    df.to_pickle(f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.pkl")
    json_df = df.copy()
    for col in json_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        json_df[col] = json_df[col].astype(str)
    to_json(
        json_df.to_dict(orient="records"),
        f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.json",
    )

def _load_response_source_similarity_frames():
    """Build long-form source-level scores plus cited-only response-level coverage metrics."""
    df = pd.read_pickle(f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.pkl")

    per_source_rows = []
    source_cols = [
        ("retrieved_sources_similarity", "Retrieved"),
        ("safe_sources_similarity", "Safe"),
        ("cited_sources_similarity", "Cited"),
    ]
    for _, row in df.iterrows():
        for source_col, source_type in source_cols:
            source_similarity = row.get(source_col, {})
            if not isinstance(source_similarity, dict):
                continue
            for url, payload in source_similarity.items():
                rouge_payload = payload.get("rouge_score", {})
                rouge_1 = 0.0
                rouge_2 = 0.0
                rouge_l = 0.0
                if isinstance(rouge_payload, dict):
                    def _rouge_precision(score_obj):
                        if hasattr(score_obj, "precision"):
                            return score_obj.precision
                        if isinstance(score_obj, dict):
                            return score_obj.get("precision", 0.0)
                        return 0.0

                    rouge_1 = _rouge_precision(rouge_payload.get("rouge1"))
                    rouge_2 = _rouge_precision(rouge_payload.get("rouge2"))
                    rouge_l = _rouge_precision(rouge_payload.get("rougeL"))
                elif hasattr(rouge_payload, "precision"):
                    rouge_l = rouge_payload.precision
                sim = payload.get("similarity_score", 0.0)
                nli_judge = payload.get("nli_judge", {}) or {}
                nli_label = str(nli_judge.get("label", "")).strip().lower()
                nli_score = int(
                    nli_judge.get("confidence", nli_judge.get("score", 0)) or 0
                )
                per_source_rows.append(
                    {
                        "user_id": row.get("user_id"),
                        "conv_id": row.get("conv_id"),
                        "turn_id": row.get("turn_id"),
                        "topic": row.get("topic"),
                        "time": pd.to_datetime(row.get("time"), errors="coerce"),
                        "url": url,
                        "source_type": source_type,
                        "response_text": row.get("asistant_response", ""),
                        "source_content": payload.get("content", ""),
                        "similarity_score": sim,
                        "rouge1_precision": rouge_1,
                        "rouge2_precision": rouge_2,
                        "rougeL_precision": rouge_l,
                        "nli_entailment": int(nli_label == "entailment") * nli_score,
                        "nli_neutral": int(nli_label == "neutral") * nli_score,
                        "nli_contradiction": int(nli_label == "contradiction") * nli_score,
                        "nli_score": nli_score,
                        "nli_label": nli_label,
                        "contradiction_reason": nli_judge.get("reasoning", ""),
                    }
                )

    per_source_df = pd.DataFrame(per_source_rows)
    if len(per_source_df) == 0:
        return per_source_df, pd.DataFrame()

    per_source_df["month"] = per_source_df["time"].dt.to_period("M").dt.to_timestamp()

    return per_source_df

def plot_response_source_quality_summary():
    df = _load_response_source_similarity_frames()
    if len(df) == 0:
        return

    row_key_cols = [
        col
        for col in ["user_id", "conv_id", "turn_id", "topic", "time", "response_text"]
        if col in df.columns
    ]

    def _make_source_types_exclusive(source_df):
        exclusive_rows = []
        for _, row_df in source_df.groupby(row_key_cols, dropna=False, sort=False):
            cited_urls = set(
                row_df.loc[row_df["source_type"] == "Cited", "url"]
                .fillna("")
                .astype(str)
            )
            safe_urls = set(
                row_df.loc[row_df["source_type"] == "Safe", "url"]
                .fillna("")
                .astype(str)
            )
            url_keys = row_df["url"].fillna("").astype(str)
            keep_mask = (
                (row_df["source_type"] == "Cited")
                | (
                    (row_df["source_type"] == "Safe")
                    & ~url_keys.isin(cited_urls)
                )
                | (
                    (row_df["source_type"] == "Retrieved")
                    & ~url_keys.isin(safe_urls | cited_urls)
                )
            )
            exclusive_rows.append(row_df.loc[keep_mask])

        if not exclusive_rows:
            return source_df.iloc[0:0].copy()
        return pd.concat(exclusive_rows, ignore_index=True)

    df = _make_source_types_exclusive(df)
    if len(df) == 0:
        return

    source_order = ["Retrieved", "Safe", "Cited"]
    color_map = {
        "Retrieved": "#636EFA",
        "Cited": "#EF553B",
        "Safe": "#00CC96",
    }

    def _plot_metric_group(
        metrics,
        file_name,
        yaxis_title,
        yaxis_range=None,
        tickformat=None,
        nli_label_filter=None,
        count_annotations=None,
    ):
        fig = go.Figure()
        for metric_col, metric_label in metrics:
            for source_type in source_order:
                subset = df[df["source_type"] == source_type]
                if nli_label_filter is not None:
                    subset = subset[subset["nli_label"] == nli_label_filter.get(metric_col, "")]
                if len(subset) == 0:
                    continue
                fig.add_trace(
                    go.Box(
                        x=[metric_label] * len(subset),
                        y=subset[metric_col],
                        name=source_type,
                        legendgroup=source_type,
                        offsetgroup=source_type,
                        marker_color=color_map[source_type],
                        boxmean=True,
                        showlegend=(metric_col == metrics[0][0]),
                    )
                )
            fig.add_vline(
                x=metric_label,
                line_width=0,
            )

        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title=yaxis_title,
            boxmode="group",
        )
        if yaxis_range is not None:
            fig.update_yaxes(range=yaxis_range)
        if tickformat is not None:
            fig.update_yaxes(tickformat=tickformat)
        if count_annotations:
            for metric_col, metric_label in metrics:
                label_counts = count_annotations.get(metric_col, {})
                annotation_text = "<br>".join(
                    [
                        f"R={label_counts.get('Retrieved', 0)}",
                        f"S={label_counts.get('Safe', 0)}",
                        f"C={label_counts.get('Cited', 0)}",
                    ]
                )
                fig.add_annotation(
                    x=metric_label,
                    y=1.08,
                    xref="x",
                    yref="paper",
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="center",
                )
            fig.update_layout(margin=dict(t=120))
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    rouge_metrics = [
        ("rouge1_precision", "Rouge-1 Precision"),
        ("rouge2_precision", "Rouge-2 Precision"),
        ("rougeL_precision", "Rouge-L Precision"),
    ]
    _plot_metric_group(
        rouge_metrics,
        "response_source_quality_rouge_summary",
        "Rouge Precision",
        yaxis_range=[0, 1],
        tickformat=".0%",
    )

    _plot_metric_group(
        [("similarity_score", "Similarity")],
        "response_source_quality_similarity_summary",
        "Similarity",
        yaxis_range=[0, 1],
        tickformat=".0%",
    )

    def _plot_nli_label_distribution():
        nli_labels = [
            ("entailment", "Entailment"),
            ("neutral", "Neutral"),
            ("contradiction", "Contradiction"),
        ]
        valid_labels = {label for label, _display in nli_labels}
        group_keys = ["user_id", "conv_id", "turn_id", "source_type"]

        nli_rate_rows = []
        filtered_df = df[df["nli_label"].isin(valid_labels)].copy()
        for group_values, group_df in filtered_df.groupby(group_keys):
            total = len(group_df)
            if total == 0:
                continue
            row_payload = dict(zip(group_keys, group_values))
            for nli_label, label_display in nli_labels:
                count = int((group_df["nli_label"] == nli_label).sum())
                row_payload[f"{nli_label}_rate"] = count / total
            nli_rate_rows.append(row_payload)

        rate_df = pd.DataFrame(nli_rate_rows)
        if len(rate_df) == 0:
            return

        fig = go.Figure()
        for source_type in source_order:
            source_df = rate_df[rate_df["source_type"] == source_type].copy()
            if len(source_df) == 0:
                continue
            for nli_label, label_display in nli_labels:
                rate_col = f"{nli_label}_rate"
                fig.add_trace(
                    go.Box(
                        x=[label_display] * len(source_df),
                        y=source_df[rate_col],
                        name=source_type,
                        legendgroup=source_type,
                        offsetgroup=source_type,
                        marker_color=color_map[source_type],
                        boxmean=True,
                        showlegend=(nli_label == nli_labels[0][0]),
                        hovertemplate=(
                            f"{source_type}<br>{label_display}: "
                            "%{y:.1%}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            xaxis_title="NLI Label",
            yaxis_title="Rate Per Sample",
            boxmode="group",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        file_name = "response_source_quality_nli_summary"
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    _plot_nli_label_distribution()

    contradiction_samples = df[df["nli_label"] == "contradiction"].copy()
    for col in contradiction_samples.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        contradiction_samples[col] = contradiction_samples[col].astype(str)
    to_json(
        contradiction_samples.to_dict(orient="records"),
        f"{OUTPUT_PATH}/{CONF}/response_source_contradiction_samples.json",
    )

    valid_nli_labels = {"entailment", "neutral", "contradiction"}
    sample_with_all_labels = None
    grouped_df = df[df["nli_label"].isin(valid_nli_labels)].copy()
    for group_keys, group in grouped_df.groupby(
        ["user_id", "conv_id", "turn_id", "source_type"], dropna=False
    ):
        present_labels = set(group["nli_label"].tolist())
        if valid_nli_labels.issubset(present_labels):
            first_row = group.iloc[0]
            sample_with_all_labels = {
                "user_id": first_row.get("user_id"),
                "conv_id": first_row.get("conv_id"),
                "turn_id": first_row.get("turn_id"),
                "source_type": first_row.get("source_type"),
                "topic": first_row.get("topic"),
                "time": str(first_row.get("time")),
                "response_text": first_row.get("response_text", ""),
                "sources": [],
            }
            for _, source_row in group.iterrows():
                sample_with_all_labels["sources"].append(
                    {
                        "url": source_row.get("url", ""),
                        "nli_label": source_row.get("nli_label", ""),
                        "nli_score": source_row.get("nli_score", None),
                        "reasoning": source_row.get("contradiction_reason", ""),
                        "source_content": source_row.get("source_content", ""),
                    }
                )
            break

    if sample_with_all_labels is not None:
        def _to_json_safe(value):
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if pd.isna(value):
                return None
            return value

        sample_with_all_labels = {
            key: (
                [_to_json_safe(v) for v in value]
                if isinstance(value, list)
                else (
                    {
                        inner_key: (
                            [
                                {
                                    source_key: _to_json_safe(source_value)
                                    for source_key, source_value in source_item.items()
                                }
                                for source_item in inner_value
                            ]
                            if inner_key == "sources" and isinstance(inner_value, list)
                            else _to_json_safe(inner_value)
                        )
                        for inner_key, inner_value in value.items()
                    }
                    if isinstance(value, dict)
                    else _to_json_safe(value)
                )
            )
            for key, value in sample_with_all_labels.items()
        }
        to_json(
            sample_with_all_labels,
            f"{OUTPUT_PATH}/{CONF}/response_source_all_nli_labels_example.json",
        )

def _load_response_source_nli_sentence_based(output_base=None):
    output_base = output_base or RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE
    pkl_path = f"{output_base}.pkl"
    csv_path = f"{output_base}.csv"
    json_path = f"{output_base}.json"

    if os.path.exists(pkl_path):
        try:
            return pd.read_pickle(pkl_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", pkl_path, e)

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    if os.path.exists(json_path):
        records = load_json(json_path)
        if isinstance(records, list):
            return pd.DataFrame(records)

    raise FileNotFoundError(
        f"Sentence-level response source NLI results not found at {output_base}.*. "
        "Run response_source_nli_sentence_based() first."
    )


def _normalize_chunking_method(chunking_method):
    chunking_method_map = {
        "citation_marker": "citation_marker",
        "marker": "citation_marker",
        "citation": "citation_marker",
        "chunk": "citation_marker",
        "chunk_based": "citation_marker",
        "claim": "claim",
        "claim_based": "claim",
        "sentence": "claim",
        "sentence_based": "claim",
    }
    chunking_method_key = str(chunking_method or "").strip().lower()
    if chunking_method_key not in chunking_method_map:
        raise ValueError(
            "chunking_method must be one of {'citation_marker', 'claim'}"
        )
    return chunking_method_map[chunking_method_key]


def _normalize_claim_selection_mode(claim_selection_mode):
    claim_selection_mode_map = {
        "all": "all",
        "all_claims": "all",
        "all_claims_in_chunk": "all",
        "latest_preceding": "latest_preceding",
        "latest_preceding_claim": "latest_preceding",
        "latest_before_marker": "latest_preceding",
        "immediate_predecessor": "latest_preceding",
    }
    mode_key = str(claim_selection_mode or "").strip().lower()
    if mode_key not in claim_selection_mode_map:
        raise ValueError(
            "claim_selection_mode must be one of {'all', 'latest_preceding'}"
        )
    return claim_selection_mode_map[mode_key]


def _response_source_nli_output_base(
    nli_method,
    chunking_method,
    claim_selection_mode="all",
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    method_base = (
        RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE
        if nli_method == "bert"
        else RESPONSE_SOURCE_NLI_SENTENCE_BASED_JUDGE_BASE
    )
    if chunking_method == "citation_marker":
        return method_base
    if claim_selection_mode == "all":
        return f"{method_base}_{chunking_method}"
    return f"{method_base}_{chunking_method}_{claim_selection_mode}"


def response_source_nli_sentence_based(
    nli_method="bert",
    judge_entailment_min_score=1,
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
    claim_cache_path=CLAIM_EXTRACTION_CACHE_PATH,
):
    """Attribute response chunks using either judge NLI or BERT NLI."""
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    output_base = _response_source_nli_output_base(
        nli_method,
        chunking_method,
        claim_selection_mode=claim_selection_mode,
    )
    persisted_claims_cache = _load_claims_cache(cache_path=claim_cache_path)
    claims_cache_dirty = False
    new_claim_cache_entries = 0

    df = _load_response_source_similarity_input()
    urls_content_by_clean_url = {}

    urls_content = _load_urls_content(
        urls_content_path=RESPONSE_URLS_CONTENT_PATH,
        required=False,
    )
    print(len(urls_content.keys()))
    for url, content in urls_content.items():
        clean_url = str(url).removesuffix("?utm_source=chatgpt.com").removesuffix(
            "&utm_source=chatgpt.com"
        )
        urls_content_by_clean_url[clean_url] = content
        urls_content_by_clean_url[clean_url.rstrip("/")] = content

    citation_marker_pattern = re.compile(
        r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    )
    citation_ref_pattern = re.compile(
        r"\ue202[A-Za-z]+(\d+)[A-Za-z]+(\d+)(?=\ue202|\ue201)"
    )

    def _safe_int(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _clean_url(url):
        if not isinstance(url, str):
            return ""
        return url.strip().removesuffix("?utm_source=chatgpt.com").removesuffix(
            "&utm_source=chatgpt.com"
        )

    def _as_source_list(value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
        return parsed if isinstance(parsed, list) else []

    claims_cache = {}

    def _extract_claims(text):
        nonlocal claims_cache_dirty
        nonlocal new_claim_cache_entries
        text = str(text or "").strip()
        if not text:
            return []
        if text in claims_cache:
            return claims_cache[text]

        cache_key = _claim_cache_key(text)
        cached_claims = persisted_claims_cache.get(cache_key, []) or persisted_claims_cache.get(text, [])
        if cached_claims:
            claims_cache[text] = cached_claims
            return cached_claims

        claims = extract_claims_from_text(text)
        if not claims:
            claims = [text]

        claims_cache[text] = claims
        persisted_claims_cache[cache_key] = claims
        claims_cache_dirty = True
        new_claim_cache_entries += 1
        if new_claim_cache_entries % 25 == 0:
            _save_claims_cache(persisted_claims_cache, cache_path=claim_cache_path)
        return claims

    def _split_sentences(text):
        if not text:
            return []
        if chunking_method == "claim":
            sentence_parts = _extract_claims(text)
        else:
            sentence_parts = re.split(
                r"(?<=[.!?])\s+(?=[`'*_\"(\[]*[A-Z0-9])",
                text,
            )

        sentences = []
        for sentence in sentence_parts:
            sentence = sentence.strip(" -*\t\n")
            if len(sentence) < 8 or not re.search(r"[A-Za-z]", sentence):
                continue
            sentences.append(sentence)
        return sentences

    def _citation_refs(marker_text):
        refs = []
        for turn_index, ref_index in citation_ref_pattern.findall(marker_text or ""):
            refs.append((int(turn_index), int(ref_index)))
        return refs

    def _append_response_chunks(rows, sentences, citation_refs, citation_markers):
        citation_refs = list(citation_refs or [])
        citation_markers = list(citation_markers or [])
        if citation_refs and chunking_method == "citation_marker":
            if not sentences:
                return
            rows.append(
                {
                    "sentence": " ".join(sentences),
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                    "citation_refs": list(citation_refs),
                    "citation_markers": list(citation_markers),
                }
            )
            return

        for sentence in sentences:
            rows.append(
                {
                    "sentence": sentence,
                    "sentences": [sentence],
                    "sentence_count": 1,
                    "citation_refs": list(citation_refs),
                    "citation_markers": list(citation_markers),
                }
            )

    def _extract_response_sentences(response_text):
        rows = []
        response_text = response_text or ""
        marker_matches = list(citation_marker_pattern.finditer(response_text))

        if (
            chunking_method == "claim"
            and claim_selection_mode == "latest_preceding"
        ):
            previous_end = 0
            last_refs_key = None
            for marker_match in marker_matches:
                marker_text = marker_match.group(0)
                marker_refs = _citation_refs(marker_text)
                refs_key = tuple(marker_refs)
                predecessor_claims = _split_sentences(
                    response_text[previous_end:marker_match.start()]
                )

                if not marker_refs:
                    previous_end = marker_match.end()
                    last_refs_key = None
                    continue

                if predecessor_claims:
                    latest_claim = predecessor_claims[-1]
                    # For repeated markers with the same refs, keep only the latest
                    # preceding claim in the current marker chunk.
                    if rows and refs_key == last_refs_key:
                        rows[-1]["sentence"] = latest_claim
                        rows[-1]["sentences"] = [latest_claim]
                        rows[-1]["sentence_count"] = 1
                        rows[-1]["citation_markers"].append(marker_text)
                    else:
                        _append_response_chunks(
                            rows,
                            [latest_claim],
                            marker_refs,
                            [marker_text],
                        )
                elif rows and refs_key == last_refs_key:
                    # Keep marker metadata even when no new predecessor claim appears.
                    rows[-1]["citation_markers"].append(marker_text)

                if marker_refs:
                    last_refs_key = refs_key

                previous_end = marker_match.end()

            return rows

        previous_end = 0
        for marker_match in marker_matches:
            marker_text = marker_match.group(0)
            marker_refs = _citation_refs(marker_text)
            raw_chunk = response_text[previous_end:marker_match.start()]
            chunk_sentences = _split_sentences(raw_chunk)

            if not chunk_sentences and rows and marker_refs:
                rows[-1]["citation_refs"].extend(
                    ref for ref in marker_refs if ref not in rows[-1]["citation_refs"]
                )
                rows[-1]["citation_markers"].append(marker_text)
            else:
                _append_response_chunks(
                    rows,
                    chunk_sentences,
                    marker_refs,
                    [marker_text],
                )

            previous_end = marker_match.end()

        tail_sentences = _split_sentences(response_text[previous_end:])
        _append_response_chunks(rows, tail_sentences, [], [])

        if not rows:
            _append_response_chunks(rows, _split_sentences(response_text), [], [])

        return rows

    def _source_records(row):
        records = []
        for source_col, source_type in [
            ("srcs_cited", "Cited"),
            ("srcs_retrieved", "Retrieved"),
        ]:
            for src in _as_source_list(row.get(source_col, [])):
                if not isinstance(src, dict):
                    continue
                url = _clean_url(src.get("url", ""))
                if not url:
                    continue
                turn_index = _safe_int(src.get("turn_index"))
                ref_index = _safe_int(src.get("ref_index"))
                records.append(
                    {
                        "url": url,
                        "source_type": source_type,
                        "turn_index": turn_index,
                        "ref_index": ref_index,
                        "ref_key": (
                            (turn_index, ref_index)
                            if turn_index is not None and ref_index is not None
                            else None
                        ),
                        "domain": src.get("domain", ""),
                        "title": src.get("title", ""),
                    }
                )
        return records

    def _source_content(url):
        url = _clean_url(url)
        return str(
            urls_content_by_clean_url.get(
                url,
                urls_content_by_clean_url.get(url.rstrip("/"), ""),
            )
            or ""
        )

    def _load_bert_nli_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(BERT_NLI_MODEL_NAME)
            model.eval()

            return {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
            }
        except Exception as e:
            logger.warning("Could not initialize BERT NLI model %s: %s", BERT_NLI_MODEL_NAME, e)
            return None

    bert_nli_model = _load_bert_nli_model()

    def _bert_nli_scores(source_text, sentence):
        if bert_nli_model is None:
            return {"label": "", "confidence": 0.0, "reasoning": ""}

        source_text = str(source_text or "").strip()
        sentence = str(sentence or "").strip()
        if not source_text or not sentence:
            return {"label": "", "confidence": 0.0, "reasoning": ""}

        torch = bert_nli_model["torch"]
        tokenizer = bert_nli_model["tokenizer"]
        model = bert_nli_model["model"]

        try:
            encoded = tokenizer(
                source_text,
                sentence,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            with torch.no_grad():
                logits = model(**encoded).logits[0]
                # label_mapping = ['entailment', 'neutral', 'contradiction']
                label_mapping = ['contradiction', 'neutral', 'entailment']
                probs = torch.softmax(logits, dim=-1)
                # print(probs)
                label_id = int(torch.argmax(probs).item())
                confidence = float(probs[label_id].item())

            label = label_mapping[label_id]
            # print(label)
            payload = {
                "label": label,
                "confidence": confidence,
                "reasoning": "",
            }
        except Exception as e:
            logger.warning("BERT NLI scoring failed: %s", e)
            payload = {"label": "", "confidence": 0.0, "reasoning": ""}

        return payload

    def _nli_label(payload):
        payload = payload if isinstance(payload, dict) else {}
        return str(payload.get("label", "")).strip().lower()

    def _nli_score(payload):
        payload = payload if isinstance(payload, dict) else {}
        try:
            return float(payload.get("confidence", payload.get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0

    def _nli_reasoning(payload):
        payload = payload if isinstance(payload, dict) else {}
        return payload.get("reasoning", payload.get("reason", ""))

    def _score_candidate(source, sentence, source_relation, source_group):
        source_text = _source_content(source["url"])
        if nli_method == "judge":
            if source_text.strip() and sentence.strip():
                nli_judge = compute_nli_scores(source_text, sentence)
            else:
                nli_judge = {"label": "", "confidence": 0.0, "reasoning": ""}
            bert_nli = {"label": "", "confidence": 0.0, "reasoning": ""}
        else:
            nli_judge = {"label": "", "confidence": 0.0, "reasoning": ""}
            bert_nli = _bert_nli_scores(source_text, sentence)

        judge_label = _nli_label(nli_judge)
        judge_score = _nli_score(nli_judge)
        judge_entailed = (
            judge_label == "entailment"
            and judge_score >= judge_entailment_min_score
        )
        bert_label = _nli_label(bert_nli)
        bert_confidence = _nli_score(bert_nli)
        bert_entailed = bert_label == "entailment"
        attribution_entailed = judge_entailed if nli_method == "judge" else bert_entailed
        source_bucket = {
            "cited_marker": "Marked Citations",
            "other_cited": "Other Citations",
            "retrieved": "Retrieved Sources",
        }.get(source_relation, source_group)

        return {
            "nli_method": nli_method,
            "url": source["url"],
            "domain": source.get("domain", ""),
            "title": source.get("title", ""),
            "source_type": source["source_type"],
            "source_relation": source_relation,
            "source_group": source_group,
            "source_bucket": source_bucket,
            "source_content_chars": len(source_text),
            "judge_nli_label": judge_label,
            "judge_nli_score": judge_score,
            "judge_nli_reasoning": _nli_reasoning(nli_judge),
            "judge_entailed": judge_entailed,
            "bert_nli_label": bert_label,
            "bert_nli_confidence": bert_confidence,
            "bert_nli_reasoning": _nli_reasoning(bert_nli),
            "bert_entailed": bert_entailed,
            "attribution_entailed": attribution_entailed,
            "entailed": attribution_entailed,
        }

    def _candidate_sources(source_records, citation_refs):
        candidates = []
        seen_urls = set()
        citation_refs = set(citation_refs or [])
        cited_sources = [
            source
            for source in source_records
            if source["source_type"] == "Cited"
        ]
        retrieved_sources = [
            source
            for source in source_records
            if source["source_type"] == "Retrieved"
        ]
        cited_by_url = {
            source["url"]: source
            for source in cited_sources
            if source.get("url")
        }
        cited_urls = set(cited_by_url)

        def _append_sources(sources, source_relation, source_group):
            for source in sources:
                url = source["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                candidate = source.copy()
                candidate["source_relation"] = source_relation
                candidate["source_group"] = source_group
                candidates.append(candidate)

        marker_sources = [
            source
            for source in cited_sources
            if source["ref_key"] in citation_refs
        ]
        marked_retrieved_sources = [
            source
            for source in retrieved_sources
            if source["ref_key"] in citation_refs
        ]
        marker_sources.extend(
            cited_by_url[source["url"]]
            for source in marked_retrieved_sources
            if source["url"] in cited_by_url
        )
        _append_sources(marker_sources, "cited_marker", "Cited Sources")

        other_cited_sources = [
            source
            for source in cited_sources
            if source["url"] not in seen_urls
        ]
        _append_sources(other_cited_sources, "other_cited", "Cited Sources")

        retrieved_sources = [
            source
            for source in retrieved_sources
            if source["url"] not in cited_urls
            and source["url"] not in seen_urls
        ]
        _append_sources(retrieved_sources, "retrieved", "Retrieved Sources")

        return candidates

    def _json_safe(value):
        if isinstance(value, dict):
            return {key: _json_safe(inner_value) for key, inner_value in value.items()}
        if isinstance(value, list):
            return [_json_safe(inner_value) for inner_value in value]
        if isinstance(value, tuple):
            return [_json_safe(inner_value) for inner_value in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            return str(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def _base_output_row(row, sample_index, sentence_index, response_text, sentence_payload):
        sentence = sentence_payload["sentence"]
        return {
            "sample": sample_index,
            "nli_method": nli_method,
            "chunking_method": chunking_method,
            "claim_selection_mode": (
                claim_selection_mode if chunking_method == "claim" else ""
            ),
            "user_id": row.get("user_id"),
            "conv_id": row.get("conv_id"),
            "turn_id": row.get("turn_id"),
            "topic": row.get("topic"),
            "language": row.get("language"),
            "time": _json_safe(row.get("time")),
            "response_text": response_text,
            "response_chunk_index": sentence_index,
            "response_chunk_text": sentence,
            "citation_refs": sentence_payload["citation_refs"],
            "citation_markers": sentence_payload["citation_markers"],
        }

    def _judge_checked_source(check):
        return {
            "url": check["url"],
            "domain": check["domain"],
            "title": check["title"],
            "source_type": check["source_type"],
            "source_bucket": check["source_bucket"],
            "source_content_chars": check["source_content_chars"],
            "judge_nli_label": check["judge_nli_label"],
            "judge_nli_score": check["judge_nli_score"],
            "judge_nli_reasoning": check["judge_nli_reasoning"],
            "judge_entailed": check["judge_entailed"],
            "bert_nli_label": check["bert_nli_label"],
            "bert_nli_confidence": check["bert_nli_confidence"],
            "bert_nli_reasoning": check["bert_nli_reasoning"],
            "bert_entailed": check["bert_entailed"],
        }

    rows = []
    for sample_index, row in tqdm(df.iterrows(), total=len(df)):
        response_text = str(row.get("asistant_response", "") or "")
        response_sentences = _extract_response_sentences(response_text)
        sources = _source_records(row)

        for sentence_index, sentence_payload in enumerate(response_sentences):
            sentence = sentence_payload["sentence"]
            citation_refs = sentence_payload["citation_refs"]
            candidates = _candidate_sources(sources, citation_refs)
            checked_sources = []
            entailed_check = None
            base_row = _base_output_row(
                row,
                sample_index,
                sentence_index,
                response_text,
                sentence_payload,
            )

            def _evaluate_candidates(candidate_group, stop_on_entailment=False):
                checks = []
                for candidate in candidate_group:
                    check = _score_candidate(
                        candidate,
                        sentence,
                        candidate["source_relation"],
                        candidate["source_group"],
                    )
                    checks.append(check)
                    if stop_on_entailment and check["attribution_entailed"]:
                        break
                return checks

            def _first_entailed(checks):
                for check in checks:
                    if check["attribution_entailed"]:
                        return check
                return None

            marker_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "cited_marker"
            ]
            other_cited_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "other_cited"
            ]
            retrieved_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "retrieved"
            ]

            marker_checks = _evaluate_candidates(marker_candidates)
            checked_sources.extend(marker_checks)
            entailed_check = _first_entailed(marker_checks)

            if entailed_check is None:
                other_cited_checks = _evaluate_candidates(
                    other_cited_candidates,
                    stop_on_entailment=True,
                )
                checked_sources.extend(other_cited_checks)
                entailed_check = _first_entailed(other_cited_checks)

            if entailed_check is None:
                retrieved_checks = _evaluate_candidates(
                    retrieved_candidates,
                    stop_on_entailment=True,
                )
                checked_sources.extend(retrieved_checks)
                entailed_check = _first_entailed(retrieved_checks)

            marker_cited_urls = [
                candidate["url"]
                for candidate in marker_candidates
            ]

            if entailed_check is None:
                entailed_check = {
                    "nli_method": nli_method,
                    "url": "",
                    "domain": "",
                    "title": "",
                    "source_type": "Unknown",
                    "source_relation": "Unknown",
                    "source_group": "Unknown",
                    "source_bucket": "Unexplained",
                    "source_content_chars": 0,
                    "judge_nli_label": "",
                    "judge_nli_score": 0,
                    "judge_nli_reasoning": "",
                    "judge_entailed": False,
                    "bert_nli_label": "",
                    "bert_nli_confidence": 0.0,
                    "bert_nli_reasoning": "",
                    "bert_entailed": False,
                    "attribution_entailed": False,
                    "entailed": False,
                }

            rows.append(
                {
                    **base_row,
                    "marker_cited_urls": marker_cited_urls,
                    "checked_source_count": len(checked_sources),
                    "entailment_source_bucket": entailed_check["source_bucket"],
                    "entailment_source_type": entailed_check["source_type"],
                    "entailed_url": entailed_check["url"],
                    "entailed_domain": entailed_check["domain"],
                    "entailed_title": entailed_check["title"],
                    "judge_nli_label": entailed_check["judge_nli_label"],
                    "judge_nli_score": entailed_check["judge_nli_score"],
                    "judge_nli_reasoning": entailed_check["judge_nli_reasoning"],
                    "judge_entailed": entailed_check["judge_entailed"],
                    "bert_nli_label": entailed_check["bert_nli_label"],
                    "bert_nli_confidence": entailed_check["bert_nli_confidence"],
                    "bert_nli_reasoning": entailed_check["bert_nli_reasoning"],
                    "bert_entailed": entailed_check["bert_entailed"],
                    "attribution_entailed": entailed_check["attribution_entailed"],
                    "entailed": entailed_check["entailed"],
                    "Unknown": (
                        entailed_check["source_group"] == "Unknown"
                    ),
                    "checked_sources": [
                        _judge_checked_source(check)
                        for check in checked_sources
                    ],
                }
            )

    if claims_cache_dirty:
        _save_claims_cache(persisted_claims_cache, cache_path=claim_cache_path)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    result_df.to_csv(f"{output_base}.csv", index=False)
    result_df.to_pickle(f"{output_base}.pkl")

    json_records = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in result_df.to_dict(orient="records")
    ]
    to_json(json_records, f"{output_base}.json")

    return result_df


def plot_response_source_nli_sentence_based(
    output_base=None,
    file_name=None,
    nli_method="bert",
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)

    modes_to_plot = (
        ["all", "latest_preceding"] if chunking_method == "claim" else [claim_selection_mode]
    )

    if file_name is None:
        file_name = f"response_source_nli_sentence_based_{nli_method}_summary"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"

    source_order = [
        "Associated Citations",
        "Other Citations",
        "Retrieved Sources",
        "Parametric Knowledge",
    ]
    color_map = {
        "Associated Citations": "#EF553B",
        "Other Citations": "#AB63FA",
        "Retrieved Sources": "#636EFA",
        "Parametric Knowledge": "#7F7F7F",
    }
    mode_label_map = {
        "all": "All Claims",
        "latest_preceding": "Latest Claim Before Citation",
    }
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    summary_frames = []

    claim_all_df = None
    claim_latest_df = None
    external_claim_mode_dfs = {}
    if chunking_method == "claim":
        claim_all_output_base = (
            output_base
            if output_base is not None
            else _response_source_nli_output_base(
                nli_method,
                chunking_method,
                claim_selection_mode="all",
            )
        )
        try:
            claim_all_df = _load_response_source_nli_sentence_based(
                output_base=claim_all_output_base
            )
        except FileNotFoundError as e:
            logger.warning("Could not load claim-based metadata for plotting: %s", e)
            return pd.DataFrame()

        def _citation_refs_key(value):
            if isinstance(value, (list, tuple)):
                parsed = value
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    return tuple()
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return tuple()
            else:
                return tuple()

            if not isinstance(parsed, (list, tuple)):
                return tuple()

            refs = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    refs.append(f"{str(item[0])}::{str(item[1])}")
                elif item is not None:
                    refs.append(str(item))
            return tuple(refs)

        def _prepare_claim_mode_dfs(df):
            df = df.copy()
            if "chunking_method" in df.columns:
                df = df[
                    df["chunking_method"].fillna("citation_marker") == "claim"
                ].copy()
            if len(df) == 0:
                return df, df

            sort_cols = [
                col
                for col in ["sample", "response_chunk_index"]
                if col in df.columns
            ]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="stable").copy()

            if "citation_refs" not in df.columns:
                return df, df.iloc[0:0].copy()

            refs_key_series = df["citation_refs"].apply(_citation_refs_key)
            nonempty_refs_mask = refs_key_series.apply(bool)
            if not bool(nonempty_refs_mask.any()):
                return df, df.iloc[0:0].copy()

            sample_series = (
                df["sample"]
                if "sample" in df.columns
                else pd.Series([0] * len(df), index=df.index)
            )
            prev_sample = sample_series.shift(1)
            prev_refs = refs_key_series.shift(1)
            same_as_prev = (
                nonempty_refs_mask
                & (sample_series == prev_sample)
                & (refs_key_series == prev_refs)
            )
            run_id = (~same_as_prev).cumsum()
            latest_df = df.loc[nonempty_refs_mask].copy()
            latest_df["_run_id"] = run_id[nonempty_refs_mask].values
            latest_df = (
                latest_df.groupby("_run_id", sort=False)
                .tail(1)
                .drop(columns=["_run_id"])
                .copy()
            )
            if sort_cols:
                latest_df = latest_df.sort_values(sort_cols, kind="stable").copy()

            return df, latest_df

        claim_all_df, claim_latest_df = _prepare_claim_mode_dfs(claim_all_df)
        if len(claim_all_df) == 0:
            return pd.DataFrame()

        # External files are treated as all-claims sources; derive latest mode here.
        for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
            platform_output_base = (
                EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES
                .get(platform_label, {})
                .get(nli_method)
            )
            if not platform_output_base:
                continue

            candidate_output_bases = [platform_output_base]
            if platform_output_base.endswith("_latest_preceding"):
                candidate_output_bases.append(
                    platform_output_base.removesuffix("_latest_preceding")
                )

            platform_raw_df = None
            last_error = None
            for candidate_output_base in candidate_output_bases:
                try:
                    platform_raw_df = _load_response_source_nli_sentence_based(
                        output_base=candidate_output_base
                    )
                    break
                except FileNotFoundError as e:
                    last_error = e
                    continue

            if platform_raw_df is None:
                logger.warning(
                    "Skipping %s claim plot data: %s",
                    platform_label,
                    last_error or "no matching metadata file found",
                )
                continue
            platform_all_df, platform_latest_df = _prepare_claim_mode_dfs(platform_raw_df)
            if len(platform_all_df) == 0:
                continue
            external_claim_mode_dfs[platform_label] = {
                "all": platform_all_df,
                "latest_preceding": platform_latest_df,
            }

    for mode in modes_to_plot:
        platform_sentence_dfs = {}
        if chunking_method == "claim":
            platform_sentence_dfs["OpenAI"] = (
                claim_all_df.copy()
                if mode == "all"
                else claim_latest_df.copy()
            )

            for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
                mode_dfs = external_claim_mode_dfs.get(platform_label)
                if not mode_dfs:
                    continue
                platform_df = mode_dfs.get(mode)
                if platform_df is None or len(platform_df) == 0:
                    continue
                platform_sentence_dfs[platform_label] = platform_df.copy()
        else:
            mode_output_base = (
                output_base
                if output_base is not None and len(modes_to_plot) == 1
                else _response_source_nli_output_base(
                    nli_method,
                    chunking_method,
                    claim_selection_mode=mode,
                )
            )
            try:
                sentence_df = _load_response_source_nli_sentence_based(
                    output_base=mode_output_base
                )
            except FileNotFoundError as e:
                logger.warning(
                    "Skipping sentence-based NLI summary for claim_selection_mode=%s: %s",
                    mode,
                    e,
                )
                continue

            if "chunking_method" in sentence_df.columns:
                sentence_df = sentence_df[
                    sentence_df["chunking_method"].fillna("citation_marker")
                    == chunking_method
                ].copy()
            platform_sentence_dfs["OpenAI"] = sentence_df

        mode_summary_frames = []
        for platform_label, sentence_df in platform_sentence_dfs.items():
            if len(sentence_df) == 0:
                continue

            source_buckets = (
                sentence_df["entailment_source_bucket"]
                .fillna("Parametric Knowledge")
                .replace(
                    {
                        "": "Parametric Knowledge",
                        "Unexplained": "Parametric Knowledge",
                        "Unknown": "Parametric Knowledge",
                        "unknown": "Parametric Knowledge",
                        "Marked Citations": "Associated Citations",
                    }
                )
            )
            sentence_weights = pd.Series([1] * len(sentence_df), index=sentence_df.index)
            total_sentences = float(sentence_weights.sum())
            if total_sentences <= 0:
                continue

            counts = (
                pd.DataFrame(
                    {
                        "entailment_source_bucket": source_buckets,
                        "sentence_weight": sentence_weights,
                    }
                )
                .groupby("entailment_source_bucket")["sentence_weight"]
                .sum()
            )
            platform_summary_df = pd.DataFrame(
                {
                    "entailment_source_bucket": source_order,
                    "sentence_count": [
                        int(counts.get(source_bucket, 0))
                        for source_bucket in source_order
                    ],
                }
            )
            platform_summary_df["sentence_rate"] = (
                platform_summary_df["sentence_count"] / total_sentences
            )
            platform_summary_df["total_sentence_count"] = int(total_sentences)
            platform_summary_df["claim_selection_mode"] = mode
            platform_summary_df["platform"] = platform_label
            mode_summary_frames.append(platform_summary_df)

        if not mode_summary_frames:
            continue
        mode_summary_df = pd.concat(mode_summary_frames, ignore_index=True)

        mode_file_name = (
            f"{file_name}_{mode}" if chunking_method == "claim" else file_name
        )
        mode_title = (
            mode_label_map.get(mode, mode) if chunking_method == "claim" else ""
        )
        mode_summary_df.to_csv(
            f"{output_dir}/{mode_file_name}.csv",
            index=False,
        )

        platform_order = EXTERNAL_PLATFORM_ORDER.copy()
        present_platforms = mode_summary_df["platform"].astype(str).unique().tolist()
        platform_order = [
            platform
            for platform in platform_order
            if platform in present_platforms
        ] + [
            platform
            for platform in present_platforms
            if platform not in platform_order
        ]
        platform_total_by_label = {}
        for platform_label in platform_order:
            platform_rows = mode_summary_df[
                mode_summary_df["platform"] == platform_label
            ]
            platform_total_by_label[platform_label] = int(
                platform_rows["total_sentence_count"].iloc[0]
            )

        fig = go.Figure()
        for source_bucket in source_order:
            rates = []
            counts = []
            totals = []
            for platform_label in platform_order:
                bucket_rows = mode_summary_df[
                    (mode_summary_df["platform"] == platform_label)
                    & (mode_summary_df["entailment_source_bucket"] == source_bucket)
                ]
                if len(bucket_rows) == 0:
                    rates.append(0.0)
                    counts.append(0)
                    totals.append(platform_total_by_label.get(platform_label, 0))
                else:
                    rates.append(float(bucket_rows["sentence_rate"].iloc[0]))
                    counts.append(int(bucket_rows["sentence_count"].iloc[0]))
                    totals.append(int(bucket_rows["total_sentence_count"].iloc[0]))
            fig.add_trace(
                go.Bar(
                    x=platform_order,
                    y=rates,
                    name=source_bucket,
                    marker_color=color_map[source_bucket],
                    text=[
                        f"{rate:.1%}"
                        if rate > 0
                        else ""
                        for rate in rates
                    ],
                    textposition="inside",
                    textfont=dict(color="white"),
                    customdata=np.column_stack([counts, totals]),
                    hovertemplate=(
                        "Platform: %{x}<br>"
                        "Source bucket: %{fullData.name}<br>"
                        "Sentence rate: %{y:.1%}<br>"
                        "Count: %{customdata[0]} / %{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            barmode="stack",
            xaxis_title="Platform",
            yaxis_title="Rate of Response Claims",
            legend_title="Source",
            title=mode_title,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=platform_order)
        fig.update_yaxes(range=[0, 1], tickformat=".0%")

        fig.write_html(f"{output_dir}/{mode_file_name}.html")
        try:
            paper_fig = with_paper_style(
                fig,
                config=styler(22, 16),
                legend_pos=(0.9, 1.2),
            )
            paper_fig.write_image(f"{output_dir}/{mode_file_name}.pdf", format="pdf")
        except Exception as e:
            logger.warning("Could not write sentence-based NLI PDF: %s", e)

        summary_frames.append(mode_summary_df.assign(plot_file_name=mode_file_name))

    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


def plot_response_source_nli_sentence_based_judge(
    output_base=None,
    file_name=None,
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
):
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    if output_base is None and chunking_method != "claim":
        output_base = _response_source_nli_output_base(
            "judge",
            chunking_method,
            claim_selection_mode=claim_selection_mode,
        )
    if file_name is None:
        file_name = "response_source_nli_sentence_based_judge_summary"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"
    return plot_response_source_nli_sentence_based(
        output_base=output_base,
        file_name=file_name,
        nli_method="judge",
        chunking_method=chunking_method,
        claim_selection_mode=claim_selection_mode,
    )


def plot_response_source_nli_entailment_score_boxplot(
    nli_method="bert",
    chunking_method="citation_marker",
    output_base=None,
    file_name=None,
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")

    chunking_method = _normalize_chunking_method(chunking_method)
    output_base = output_base or _response_source_nli_output_base(
        nli_method,
        chunking_method,
    )
    chunking_label = "chunk" if chunking_method == "citation_marker" else "claim"
    if file_name is None:
        file_name = (
            f"response_source_nli_{nli_method}_{chunking_label}_"
            "entailment_score_boxplot"
        )

    sentence_df = _load_response_source_nli_sentence_based(output_base=output_base)
    if "chunking_method" in sentence_df.columns:
        sentence_df = sentence_df[
            sentence_df["chunking_method"].fillna("citation_marker") == chunking_method
        ].copy()
    if len(sentence_df) == 0:
        return pd.DataFrame()

    source_order = [
        "Marked Citations",
        "Other Citations",
        "Retrieved Sources",
        "Unexplained",
    ]
    color_map = {
        "Marked Citations": "#EF553B",
        "Other Citations": "#AB63FA",
        "Retrieved Sources": "#636EFA",
        "Unexplained": "#00CC96",
    }

    source_buckets = (
        sentence_df["entailment_source_bucket"]
        .fillna("Unexplained")
        .replace(
            {
                "": "Unexplained",
                "Unknown": "Unexplained",
                "unknown": "Unexplained",
            }
        )
    )
    score_col = "bert_nli_confidence" if nli_method == "bert" else "judge_nli_score"
    score_series = pd.to_numeric(sentence_df.get(score_col), errors="coerce")
    plot_df = pd.DataFrame(
        {
            "entailment_source_bucket": source_buckets,
            "entailment_score": score_series,
        }
    )
    plot_df = plot_df[plot_df["entailment_source_bucket"].isin(source_order)].copy()
    plot_df = plot_df.dropna(subset=["entailment_score"]).copy()
    if len(plot_df) == 0:
        return pd.DataFrame()

    summary_df = (
        plot_df.groupby("entailment_source_bucket")["entailment_score"]
        .agg(["count", "mean", "median"])
        .rename(
            columns={
                "count": "score_count",
                "mean": "score_mean",
                "median": "score_median",
            }
        )
        .reindex(source_order)
        .fillna(0)
        .reset_index()
    )

    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(f"{output_dir}/{file_name}.csv", index=False)

    fig = go.Figure()
    for source_bucket in source_order:
        subset = plot_df[plot_df["entailment_source_bucket"] == source_bucket].copy()
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Box(
                x=[source_bucket] * len(subset),
                y=subset["entailment_score"],
                name=source_bucket,
                legendgroup=source_bucket,
                marker_color=color_map[source_bucket],
                boxmean=True,
                showlegend=False,
                hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
            )
        )

    score_label = (
        "BERT Entailment Confidence" if nli_method == "bert" else "Judge Entailment Score"
    )
    fig.update_layout(
        xaxis_title="Source Bucket",
        yaxis_title=score_label,
        boxmode="group",
        showlegend=False,
    )
    if nli_method == "bert":
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(range=[0, 5])

    fig.write_html(f"{output_dir}/{file_name}.html")
    try:
        paper_fig = with_paper_style(fig, config=styler(18, 18))
        paper_fig.update_xaxes(tickfont=dict(size=14))
        paper_fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    except Exception as e:
        logger.warning("Could not write entailment score boxplot PDF: %s", e)

    return summary_df


def plot_response_source_nli_entailment_score_boxplots_all():
    combinations = [
        ("bert", "citation_marker"),
        ("bert", "claim"),
        ("judge", "citation_marker"),
        ("judge", "claim"),
    ]
    summary_frames = []

    for nli_method, chunking_method in combinations:
        chunking_label = "chunk" if chunking_method == "citation_marker" else "claim"
        file_name = (
            f"response_source_nli_{nli_method}_{chunking_label}_"
            "entailment_score_boxplot"
        )
        try:
            summary_df = plot_response_source_nli_entailment_score_boxplot(
                nli_method=nli_method,
                chunking_method=chunking_method,
                file_name=file_name,
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s/%s plot: %s", nli_method, chunking_method, e)
            continue

        if len(summary_df) == 0:
            continue
        summary_df = summary_df.copy()
        summary_df["nli_method"] = nli_method
        summary_df["chunking_method"] = chunking_method
        summary_frames.append(summary_df)

    if not summary_frames:
        return pd.DataFrame()

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    combined_summary_df.to_csv(
        f"{output_dir}/response_source_nli_entailment_score_boxplot_all_summary.csv",
        index=False,
    )
    return combined_summary_df


if __name__ == "__main__":
    # web_df = load_web_data_from_file(fmt="pkl")
    # print(f"Loaded web data: {len(web_df)}")
    # extract_response_and_sources(web_df)
    # df = _load_response_source_similarity_input()
    # print(len(df))
    # print(df["topic"].value_counts())
    # asyncio.run(extract_urls_content(force_refresh=True))
    # response_source_similarity()
    # plot_response_source_quality_summary()

    # response_source_nli_sentence_based(nli_method="judge", chunking_method="citation_marker")
    # plot_response_source_nli_sentence_based_judge(chunking_method="citation_marker")
    
    # response_source_nli_sentence_based(nli_method="bert", chunking_method="citation_marker")
    # plot_response_source_nli_sentence_based(chunking_method="citation_marker")

    # response_source_nli_sentence_based(nli_method="bert", chunking_method="claim")
    plot_response_source_nli_sentence_based(
        chunking_method="claim",
    )
    # response_source_nli_sentence_based(nli_method="judge", chunking_method="claim")
    plot_response_source_nli_sentence_based_judge(
        chunking_method="claim",
    )

    # plot_response_source_nli_entailment_score_boxplot(nli_method="bert", chunking_method="citation_marker")
    # plot_response_source_nli_entailment_score_boxplot(nli_method="bert", chunking_method="claim")
    # plot_response_source_nli_entailment_score_boxplot(nli_method="judge", chunking_method="citation_marker")
    # plot_response_source_nli_entailment_score_boxplot(nli_method="judge", chunking_method="claim")
