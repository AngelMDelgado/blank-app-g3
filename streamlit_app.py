import io
import pandas as pd
import streamlit as st
import nltk

###############################################################################
# 📄 Text Pre-processor for Classification                                    #
# This app tokenizes text into sentences and creates a rolling context window.#
###############################################################################

# --- NLTK Setup: Download the 'punkt' tokenizer if not present ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    st.spinner("Downloading language model for sentence tokenization...")
    nltk.download('punkt')

# --- Page Configuration ---
st.set_page_config(page_title="Text Pre-processor", layout="wide")
st.title("📄 Text Pre-processor for Classification")
st.markdown("""
This app pre-processes text for classification tasks.
1.  **Sentence Tokenizer**: It breaks down text from a specified column into individual sentences. Each sentence becomes a 'Statement'.
2.  **Rolling Context Window**: It creates a 'Context' for each statement, which includes the statement itself and a specified number of preceding sentences.
""")

# --- Sidebar: Upload & Configuration ---
with st.sidebar:
    st.header("🗂️ 1. Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a text column.",
        type=["csv"]
    )

    st.markdown("---")
    st.header("🔧 2. Configure Pre-processing")

    # User specifies the name of the column containing the text
    input_col = st.text_input(
        "Name of the column with text to process",
        "text",
        help="Enter the exact column name from your CSV that contains the social media posts or sales transcripts."
    )

    # User specifies the size of the rolling context window
    window_size = st.number_input(
        "Context Window Size (preceding sentences)",
        min_value=0,
        max_value=20,
        value=3,
        help="The number of *previous* sentences to include in the context along with the current one. '0' means the context is just the statement itself."
    )

###############################################################################
# Helper – Pre-processing Function
###############################################################################

def preprocess_text(df: pd.DataFrame, text_column: str, window: int) -> pd.DataFrame:
    """
    Tokenizes text in a column into sentences and creates a rolling context
    window for each sentence.
    """
    processed_rows = []

    # Iterate over each row in the original dataframe
    for index, row in df.iterrows():
        # Ensure the value is a string
        text = str(row.get(text_column, ''))
        if not text.strip():
            continue

        # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)

        # For each sentence, create a new row with its statement and context
        for i, sentence in enumerate(sentences):
            # Determine the start of the context window
            start_index = max(0, i - window)
            # Slice the list to get sentences for the context
            context_sentences = sentences[start_index : i + 1]
            # Join them into a single string
            context = " ".join(context_sentences)

            # Create a new dictionary for the row, preserving original data
            new_row = row.to_dict()
            new_row["Statement"] = sentence
            new_row["Context"] = context
            processed_rows.append(new_row)

    if not processed_rows:
        return pd.DataFrame(columns=list(df.columns) + ["Statement", "Context"])

    # Create a new dataframe from the list of processed rows
    result_df = pd.DataFrame(processed_rows)

    # Optional: Remove the original, unprocessed text column to avoid redundancy
    if text_column in result_df.columns:
        result_df = result_df.drop(columns=[text_column])

    return result_df

###############################################################################
# 🚀 Main – Run Pre-processing & Display Results
###############################################################################

def run_preprocessor(file_buffer: io.BytesIO, text_col_name: str, window_sz: int):
    """Main function to run the Streamlit app flow."""
    df = pd.read_csv(file_buffer)

    if text_col_name not in df.columns:
        st.error(f"❌ The uploaded CSV must contain a column named '{text_col_name}'.")
        st.info(f"Available columns are: {', '.join(df.columns)}")
        return

    # Pre-process the dataframe
    with st.spinner("Pre-processing text... This may take a moment."):
        processed_df = preprocess_text(df, text_column=text_col_name, window=window_sz)

    st.success("✅ Pre-processing complete!")

    # Display a preview of the processed data
    st.subheader("🔍 Preview of Processed Data")
    st.dataframe(processed_df.head(20), use_container_width=True)

    # Allow user to download the final CSV
    csv_bytes = processed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Processed CSV",
        data=csv_bytes,
        file_name="preprocessed_output.csv",
        mime="text/csv",
    )

###############################################################################
# 🏁 App Execution
###############################################################################
if uploaded_file is not None:
    if not input_col:
        st.warning("Please enter the name of the column containing the text to process.")
    else:
        try:
            run_preprocessor(uploaded_file, text_col_name=input_col, window_sz=window_size)
        except Exception as e:
            st.exception(e)
else:
    st.info("👆 Upload a CSV file in the sidebar to get started.")
