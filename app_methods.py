import io
import re
import time
from collections import Counter
from heapq import nlargest

import docx2txt
import matplotlib.pyplot as plt
import neattext as nt
import neattext.functions as nfx
import pandas as pd
import requests
import spacy
import streamlit as st
import textstat
import yake
from better_profanity import profanity
from bs4 import BeautifulSoup
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from matplotlib import use as matplotlib_use
from PyPDF2 import PdfReader
from spacy import displacy
from textblob import TextBlob
from wordcloud import WordCloud

matplotlib_use("Agg")

# Brand palette + clean chart style (matches the UI theme)
PRIMARY, TEAL, AMBER, PINK = "#6C5CE7", "#00B8A9", "#FFB84C", "#F6416C"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    "figure.figsize": (6, 3.6), "font.size": 10,
    "text.color": "#374151", "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280", "ytick.color": "#6B7280",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#D1D5DB", "axes.grid": True, "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})

LANG_NAMES = {"en": "English", "mr": "Marathi", "hi": "Hindi", "gu": "Gujarati",
              "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
              "fr": "French", "de": "German", "es": "Spanish"}


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once, not on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


nlp = load_nlp()


# --------------------------------------------------------------------------
# Core NLP functions
# --------------------------------------------------------------------------
def text_analyzer(my_text):
    """Token-level attributes as a DataFrame."""
    doc = nlp(my_text)
    data = [(t.text, t.shape_, t.pos_, t.tag_, t.lemma_, t.is_alpha, t.is_stop)
            for t in doc]
    return pd.DataFrame(data, columns=["Token", "Shape", "PoS", "Tag",
                                       "Lemma", "Is_Alpha", "Is_Stopword"])


def get_entities(my_text):
    """Named entities as a DataFrame."""
    doc = nlp(my_text)
    return pd.DataFrame([(e.text, e.label_, spacy.explain(e.label_))
                         for e in doc.ents],
                        columns=["Entity", "Label", "Meaning"])


def render_entities(my_text):
    """Entities highlighted inline (spaCy displacy) as HTML."""
    return displacy.render(nlp(my_text), style="ent")


def get_most_common_tokens(my_text, num=5):
    """Most common words (lowercased, punctuation removed)."""
    cleaned = nfx.remove_puncts(my_text.lower())
    words = [w for w in cleaned.split() if w.strip()]
    return dict(Counter(words).most_common(num))


def get_ngrams(my_text, n=2, top=10):
    """Most common n-grams (bigrams / trigrams) after stopword removal."""
    cleaned = nfx.remove_puncts(nfx.remove_stopwords(my_text.lower()))
    words = cleaned.split()
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return dict(Counter(grams).most_common(top))


def get_sentiment(my_text):
    return TextBlob(my_text).sentiment


def sentiment_label(polarity):
    if polarity > 0.1:
        return "Positive 😊"
    if polarity < -0.1:
        return "Negative 😠"
    return "Neutral 😐"


def sentence_sentiments(my_text):
    """Polarity for each sentence, for the sentiment-flow chart."""
    doc = nlp(my_text)
    rows = [TextBlob(s.text).sentiment.polarity
            for s in doc.sents if s.text.strip()]
    return pd.DataFrame({"Polarity": rows},
                        index=range(1, len(rows) + 1))


def summarize(my_text, n_sent=3):
    """Simple extractive summarizer (word-frequency based)."""
    doc = nlp(my_text)
    freq = Counter(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)
    if not freq:
        return ""
    mx = max(freq.values())
    scores = {}
    for sent in doc.sents:
        score = sum(freq[t.lemma_.lower()] / mx
                    for t in sent if t.lemma_.lower() in freq)
        scores[sent] = score
    best = nlargest(n_sent, scores, key=scores.get)
    # keep the original sentence order
    best = sorted(best, key=lambda s: s.start)
    return " ".join(s.text.strip() for s in best)


def get_readability(my_text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(my_text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(my_text),
        "Gunning Fog": textstat.gunning_fog(my_text),
        "Sentences": textstat.sentence_count(my_text),
        "Avg. words / sentence": round(
            textstat.lexicon_count(my_text) / max(textstat.sentence_count(my_text), 1), 1),
    }


# --------------------------------------------------------------------------
# Language detection + translation (Marathi / Hindi -> English)
# --------------------------------------------------------------------------
def detect_language(my_text):
    try:
        return detect(my_text)
    except LangDetectException:
        return "unknown"


@st.cache_data(show_spinner=False)
def translate_to_english(my_text, chunk_size=4000):
    """Translate in chunks because the translator has a ~5000 char limit."""
    sentences = re.split(r"(?<=[.!?।])\s+", my_text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > chunk_size:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    translator = GoogleTranslator(source="auto", target="en")
    return " ".join(translator.translate(c) for c in chunks if c.strip())


def prepare_text(raw_text, auto_translate=True):
    """Detect language; translate to English if needed (spaCy model is English)."""
    lang = detect_language(raw_text)
    if lang not in ("en", "unknown") and auto_translate:
        try:
            with st.spinner(f"Translating from {LANG_NAMES.get(lang, lang)} to English..."):
                translated = translate_to_english(raw_text)
            st.info(f"Detected **{LANG_NAMES.get(lang, lang)}** → translated to English "
                    f"for analysis. (Needs internet connection.)")
            return translated
        except Exception as e:
            st.warning(f"Translation failed ({e}). Analyzing the original text.")
    elif lang not in ("en", "unknown"):
        st.warning(f"Detected {LANG_NAMES.get(lang, lang)}. Results may be poor because "
                   f"the model is English-only. Enable auto-translate in the sidebar.")
    return raw_text


# --------------------------------------------------------------------------
# Input readers
# --------------------------------------------------------------------------
def read_pdf(file):
    reader = PdfReader(file)
    return " ".join((page.extract_text() or "") for page in reader.pages)


@st.cache_data(show_spinner=False, ttl=600)
def read_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))


def handle_uploaded_file(text_file):
    if text_file.type == "application/pdf":
        return read_pdf(text_file)
    if text_file.type == "text/plain":
        return text_file.read().decode("utf-8", errors="ignore")
    return docx2txt.process(text_file)


# --------------------------------------------------------------------------
# Extra simple features
# --------------------------------------------------------------------------
def get_yake_keywords(my_text, top=10):
    """Keyword / keyphrase extraction with YAKE (lower score = more important)."""
    extractor = yake.KeywordExtractor(lan="en", n=3, top=top)
    kws = extractor.extract_keywords(my_text)
    return pd.DataFrame(kws, columns=["Keyword / Phrase", "Score"]).sort_values("Score")


def get_reading_time(my_text):
    """Reading time (200 wpm) and speaking time (130 wpm) in minutes."""
    words = len(my_text.split())
    return {"Words": words,
            "Reading time (min)": round(words / 200, 1),
            "Speaking time (min)": round(words / 130, 1)}


def extract_patterns(my_text):
    """Find emails, URLs, hashtags, @mentions and phone numbers using regex."""
    return {
        "Emails": re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", my_text),
        "URLs": re.findall(r"https?://\S+|www\.\S+", my_text),
        "Hashtags": re.findall(r"#\w+", my_text),
        "Mentions": re.findall(r"(?<![\w.])@\w+", my_text),
        "Phone numbers": re.findall(r"(?:\+?\d{1,3}[\s-]?)?\d{10}\b", my_text),
    }


def clean_text(my_text, urls=True, emails=True, emojis=True, numbers=False, puncts=False,
               stopwords=False):
    """Clean text with neattext, based on the selected options."""
    if urls:
        my_text = nfx.remove_urls(my_text)
    if emails:
        my_text = nfx.remove_emails(my_text)
    if emojis:
        my_text = nfx.remove_emojis(my_text)
    if numbers:
        my_text = nfx.remove_numbers(my_text)
    if puncts:
        my_text = nfx.remove_puncts(my_text)
    if stopwords:
        my_text = nfx.remove_stopwords(my_text)
    return re.sub(r"\s+", " ", my_text).strip()


def check_profanity(my_text):
    """Return (list of bad words found, censored text)."""
    profanity.load_censor_words()
    found = sorted({w for w in re.findall(r"\w+", my_text.lower())
                    if profanity.contains_profanity(w)})
    return found, profanity.censor(my_text)


def text_similarity(text_a, text_b):
    """Cosine similarity (TF-IDF) between two texts, in percent."""
    vec = TfidfVectorizer(stop_words="english").fit_transform([text_a, text_b])
    return round(float(cosine_similarity(vec[0], vec[1])[0][0]) * 100, 2)


def batch_sentiment(df, column):
    """Add Polarity + Sentiment columns for every row in a CSV column."""
    df = df.copy()
    df[column] = df[column].astype(str)
    df["Polarity"] = df[column].apply(lambda t: round(TextBlob(t).sentiment.polarity, 3))
    df["Sentiment"] = df["Polarity"].apply(sentiment_label)
    return df


def text_to_speech(my_text, lang="en"):
    """Convert text to MP3 bytes with gTTS (needs internet)."""
    if lang != "en":
        my_text = GoogleTranslator(source="auto", target=lang).translate(my_text[:4500])
    buf = io.BytesIO()
    gTTS(text=my_text[:4500], lang=lang).write_to_fp(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_wordcloud(my_text):
    if not my_text.strip():
        st.warning("Not enough words for a word cloud.")
        return
    wc = WordCloud(width=900, height=420, background_color=None, mode="RGBA",
                   colormap="cool", prefer_horizontal=0.9).generate(my_text)
    fig = plt.figure(figsize=(9, 4.2))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def plot_bar(data, title="", color=PRIMARY, horizontal=False):
    if not data:
        st.warning("Insufficient data.")
        return
    keys, vals = list(data.keys()), list(data.values())
    fig, ax = plt.subplots()
    if horizontal:
        bars = ax.barh(keys[::-1], vals[::-1], color=color, height=0.65)
        ax.grid(axis="y", visible=False)
    else:
        bars = ax.bar(keys, vals, color=color, width=0.65)
        ax.grid(axis="x", visible=False)
        plt.xticks(rotation=45, ha="right")
    ax.bar_label(bars, padding=3, fontsize=9, color="#374151")
    if title:
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_donut(counts):
    """Donut chart for sentiment counts (labels may contain emojis, which we strip)."""
    colors = {"Positive": TEAL, "Neutral": AMBER, "Negative": PINK}
    labels = [str(l).split()[0] for l in counts.index]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.78,
           colors=[colors.get(l, PRIMARY) for l in labels],
           wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
           textprops={"fontsize": 10})
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main analysis UI
# --------------------------------------------------------------------------
def analyze_text(raw_text, num_most_common=5, n_sentences=3):
    if not raw_text or not raw_text.strip():
        st.warning("Please provide some text to analyze.")
        return
    if len(raw_text.split()) < 3:
        st.warning("Text is too short for meaningful analysis.")
        return

    processed_text = nfx.remove_stopwords(raw_text)
    token_df = text_analyzer(raw_text)

    tab_overview, tab_entities, tab_sentiment, tab_visuals, tab_tokens, tab_extras = st.tabs(
        ["📄 Overview", "🏷️ Entities", "😊 Sentiment", "📊 Visualizations",
         "🔤 Tokens & Download", "🧰 Extras"])

    # ---------------- Overview ----------------
    with tab_overview:
        with st.expander("Original Text"):
            st.write(raw_text)

        st.markdown("### 📝 Summary")
        summary = summarize(raw_text, n_sentences)
        st.success(summary if summary else "Could not generate a summary.")

        st.markdown("### 📚 Readability")
        read = get_readability(raw_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reading Ease", read["Flesch Reading Ease"])
        c2.metric("Grade Level", read["Flesch-Kincaid Grade"])
        c3.metric("Avg words/sentence", read["Avg. words / sentence"])
        st.caption("Reading Ease: higher = easier (60-70 is plain English). "
                   "Grade Level = US school grade needed to understand it.")

        st.markdown("### 🔢 Word Stats")
        st.json(nt.TextFrame(raw_text).word_stats())

        st.markdown("### ⏱️ Reading Time")
        rt = get_reading_time(raw_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", rt["Words"])
        c2.metric("Reading time (min)", rt["Reading time (min)"])
        c3.metric("Speaking time (min)", rt["Speaking time (min)"])

        st.markdown("### 🔑 Top Keywords")
        st.write(get_most_common_tokens(processed_text, num_most_common))

        st.markdown("### 🧠 Key Phrases (YAKE)")
        try:
            st.dataframe(get_yake_keywords(raw_text, 10), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not extract key phrases: {e}")

    # ---------------- Entities ----------------
    with tab_entities:
        st.markdown("### Highlighted Entities")
        html = render_entities(raw_text)
        st.components.v1.html(html, height=400, scrolling=True)
        with st.expander("Entity table"):
            ent_df = get_entities(raw_text)
            if ent_df.empty:
                st.info("No named entities found.")
            else:
                st.dataframe(ent_df, use_container_width=True)
                st.bar_chart(ent_df["Label"].value_counts(), color=PRIMARY)

    # ---------------- Sentiment ----------------
    with tab_sentiment:
        sent = get_sentiment(raw_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall", sentiment_label(sent.polarity))
        c2.metric("Polarity (-1 to 1)", round(sent.polarity, 3))
        c3.metric("Subjectivity (0 to 1)", round(sent.subjectivity, 3))
        st.progress(min(max((sent.polarity + 1) / 2, 0.0), 1.0))

        st.markdown("### Sentiment flow (sentence by sentence)")
        flow = sentence_sentiments(raw_text)
        if len(flow) > 1:
            st.line_chart(flow, color=PRIMARY)
        else:
            st.info("Need more than one sentence for a sentiment flow chart.")

    # ---------------- Visualizations ----------------
    with tab_visuals:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top words**")
            plot_bar(get_most_common_tokens(processed_text, num_most_common))
            st.markdown("**Part of Speech**")
            pos_counts = token_df["PoS"].value_counts().to_dict()
            plot_bar(pos_counts, color=TEAL)
        with col2:
            st.markdown("**Top bigrams**")
            plot_bar(get_ngrams(raw_text, 2, 8), color=AMBER, horizontal=True)
            st.markdown("**Top trigrams**")
            plot_bar(get_ngrams(raw_text, 3, 8), color=PINK, horizontal=True)

        st.markdown("**Word Cloud**")
        plot_wordcloud(processed_text)

    # ---------------- Tokens & Download ----------------
    with tab_tokens:
        st.dataframe(token_df, use_container_width=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        st.download_button("⬇️ Download token analysis (CSV)",
                           token_df.to_csv(index=False),
                           file_name=f"nlp_result_{timestr}.csv",
                           mime="text/csv")

    # ---------------- Extras ----------------
    with tab_extras:
        st.markdown("### 🔍 Extracted Items")
        found_any = False
        for label, items in extract_patterns(raw_text).items():
            if items:
                found_any = True
                st.write(f"**{label}:** {', '.join(sorted(set(items)))}")
        if not found_any:
            st.info("No emails, URLs, hashtags, mentions or phone numbers found.")

        st.markdown("### 🚫 Profanity Check")
        bad_words, censored = check_profanity(raw_text)
        if bad_words:
            st.error(f"Found {len(bad_words)} offensive word(s): {', '.join(bad_words)}")
            with st.expander("Censored version"):
                st.write(censored)
        else:
            st.success("No offensive words found ✅")

        st.markdown("### 🔊 Listen to Summary")
        lang_choice = st.selectbox("Language", ["en", "hi", "mr"], key="tts_lang",
                                   format_func=lambda c: LANG_NAMES[c])
        if st.button("Generate audio", key="tts_btn"):
            summary_text = summarize(raw_text, n_sentences)
            if not summary_text:
                st.warning("Nothing to read.")
            else:
                try:
                    with st.spinner("Generating audio..."):
                        st.audio(text_to_speech(summary_text, lang_choice), format="audio/mp3")
                except Exception as e:
                    st.error(f"Audio failed ({e}). Internet connection is required.")