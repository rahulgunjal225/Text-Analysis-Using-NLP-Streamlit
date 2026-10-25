import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Text Analysis Tool",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

from app_methods import *  # noqa: E402  (must come after set_page_config)


def main():
    st.title("📊 NLP Application With Streamlit")

    menu = ["Text Analysis Dashboard",
            "File-Based NLP Analysis",
            "URL-Based Analysis",
            "Text Similarity Checker",
            "Batch CSV Sentiment",
            "Text Cleaner",
            "About This Application"]
    choice = st.sidebar.selectbox("Menu", menu)

    # ---- Shared sidebar settings ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    num_most_common = st.sidebar.number_input("Most Common Tokens", 1, 20, 5)
    n_sentences = st.sidebar.slider("Summary length (sentences)", 1, 10, 3)
    auto_translate = st.sidebar.checkbox(
        "Auto-translate Marathi/Hindi/other → English", value=True,
        help="The spaCy model is English-only, so other languages are translated first. "
             "Requires internet.")

    # ---------------- Text input ----------------
    if choice == "Text Analysis Dashboard":
        st.subheader("Analyze Text")
        raw_text = st.text_area("Enter Text Here...", height=250)

        if st.button("Analyze"):
            st.session_state["text_analyzed"] = True

        if st.session_state.get("text_analyzed"):
            if not raw_text.strip():
                st.warning("Please enter some text first.")
            else:
                text = prepare_text(raw_text, auto_translate)
                analyze_text(text, num_most_common, n_sentences)

    # ---------------- File input ----------------
    elif choice == "File-Based NLP Analysis":
        st.subheader("Upload and Analyze Text Files")
        text_file = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"])

        if text_file is not None:
            try:
                raw_text = handle_uploaded_file(text_file)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                return
            if not raw_text.strip():
                st.warning("No text found in the file (scanned PDFs need OCR).")
            else:
                text = prepare_text(raw_text, auto_translate)
                analyze_text(text, num_most_common, n_sentences)

    # ---------------- URL input ----------------
    elif choice == "URL-Based Analysis":
        st.subheader("Analyze a Web Article")
        url = st.text_input("Paste an article URL (https://...)")

        if st.button("Fetch & Analyze"):
            st.session_state["url_analyzed"] = True

        if st.session_state.get("url_analyzed"):
            if not url.startswith(("http://", "https://")):
                st.warning("Please enter a valid URL starting with http:// or https://")
            else:
                try:
                    with st.spinner("Fetching article..."):
                        raw_text = read_url(url)
                except Exception as e:
                    st.error(f"Could not fetch the page: {e}")
                    return
                if not raw_text.strip():
                    st.warning("No readable text found on that page.")
                else:
                    text = prepare_text(raw_text, auto_translate)
                    analyze_text(text, num_most_common, n_sentences)

    # ---------------- Similarity ----------------
    elif choice == "Text Similarity Checker":
        st.subheader("Compare Two Texts (Plagiarism Check)")
        col1, col2 = st.columns(2)
        text_a = col1.text_area("Text A", height=250)
        text_b = col2.text_area("Text B", height=250)

        if st.button("Compare"):
            if not text_a.strip() or not text_b.strip():
                st.warning("Please enter both texts.")
            else:
                score = text_similarity(text_a, text_b)
                st.metric("Similarity", f"{score}%")
                st.progress(min(score / 100, 1.0))
                if score >= 70:
                    st.error("High similarity - possible plagiarism.")
                elif score >= 40:
                    st.warning("Moderate similarity.")
                else:
                    st.success("Low similarity - the texts are mostly different.")

    # ---------------- Batch CSV ----------------
    elif choice == "Batch CSV Sentiment":
        st.subheader("Sentiment Analysis on a CSV File")
        st.caption("Upload a CSV (for example product reviews) and pick the text column.")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Could not read the CSV: {e}")
                return
            column = st.selectbox("Text column", df.columns)
            if len(df) > 10:
                max_rows = st.slider("Rows to analyze", 10, min(len(df), 2000),
                                     min(len(df), 500))
            else:
                max_rows = len(df)
            result = batch_sentiment(df.head(max_rows), column)

            counts = result["Sentiment"].value_counts()
            c1, c2 = st.columns(2)
            c1.metric("Average polarity", round(result["Polarity"].mean(), 3))
            c2.metric("Rows analyzed", len(result))

            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(result, use_container_width=True)
            st.download_button("⬇️ Download results (CSV)", result.to_csv(index=False),
                               file_name="batch_sentiment.csv", mime="text/csv")

    # ---------------- Text cleaner ----------------
    elif choice == "Text Cleaner":
        st.subheader("Clean Your Text")
        raw_text = st.text_area("Enter text to clean...", height=200)
        c1, c2, c3 = st.columns(3)
        urls = c1.checkbox("Remove URLs", True)
        emails = c1.checkbox("Remove emails", True)
        emojis = c2.checkbox("Remove emojis", True)
        numbers = c2.checkbox("Remove numbers", False)
        puncts = c3.checkbox("Remove punctuation", False)
        stopwords = c3.checkbox("Remove stopwords", False)

        if st.button("Clean"):
            cleaned = clean_text(raw_text, urls, emails, emojis, numbers, puncts, stopwords)
            st.text_area("Cleaned text", cleaned, height=200)
            st.caption(f"{len(raw_text.split())} words → {len(cleaned.split())} words")
            st.download_button("⬇️ Download cleaned text", cleaned, file_name="cleaned.txt")

    # ---------------- About ----------------
    else:
        st.write("""
        An interactive NLP application for analyzing text using spaCy, TextBlob and Streamlit.
        """)

        st.subheader("Features")
        st.markdown("""
        - **Text / File / URL input** - paste text, upload PDF, DOCX, TXT or analyze a web article
        - **Multilingual input** - Marathi, Hindi and other languages are detected and translated to English
        - **Summarization** - extractive summary of the text
        - **Named Entity Recognition** - highlighted inline plus a table
        - **Sentiment analysis** - overall label, polarity, subjectivity and sentence-by-sentence flow
        - **Readability scores** - Flesch Reading Ease, grade level
        - **Visualizations** - top words, bigrams, trigrams, POS distribution, word cloud
        - **Token table** - POS, tag, lemma, stopword info, downloadable as CSV
        - **Key phrases (YAKE)**, reading/speaking time, email/URL/hashtag extractor
        - **Profanity check** and **text-to-speech** of the summary (English/Hindi/Marathi)
        - **Text similarity checker**, **batch CSV sentiment** and **text cleaner**
        """)


        st.subheader("My Contributions")
        st.markdown("""
        - Bug fixes (tokenization, PDF reading, caching, input validation)
        - Added summarization, readability, n-grams, highlighted NER
        - Added URL input and Marathi/Hindi translation support
        - Redesigned UI with tabs and metrics
        - Added YAKE keywords, profanity check, text-to-speech, similarity checker,
          batch CSV sentiment and text cleaner
        """)

        st.subheader("About the Developer")
        st.write("**Rahul Gunjal** - BCOE, BE(AIML)  \n"
                 "GitHub: https://github.com/rahulgunjal225  \n"
                 "Email: rahulgunjal225@gmail.com")


if __name__ == "__main__":
    main()