import io
import re
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import streamlit as st

###############################################################################
# Streamlit – Text Classification Tool                     #
###############################################################################
st.set_page_config(page_title="Marketing & Brand Analysis Tool", layout="wide")
st.title("📈 Text Classification Tool")
st.markdown("*Analyze marketing keywords and brand language effectiveness*")

# ---------------------------------------------------------------------------
# 🛠️ Sidebar – Upload & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂️ 1. Analysis Mode")
    analysis_mode = st.selectbox(
        "Choose analysis type:",
        ["Marketing Keyword Classification", "Brand Language Analysis", "Combined Analysis"]
    )
    
    st.markdown("---")
    st.header("📁 2. Upload Your Files")
    
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        uploaded_file = st.file_uploader(
            "CSV file with a 'Statement' column", 
            type=["csv"], 
            key="marketing_file"
        )
    
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        brand_file = st.file_uploader(
            "Brand analysis CSV (with 'Caption' column)", 
            type=["csv"], 
            key="brand_file"
        )
        
        engagement_file = st.file_uploader(
            "Engagement data CSV (optional)", 
            type=["csv"], 
            key="engagement_file"
        )

    st.markdown("---")
    st.header("🔧 3. Configure Analysis")

    # Marketing Keyword Dictionaries
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        st.subheader("Marketing Keywords")
        
        # Default marketing keyword dictionaries
        default_dicts: Dict[str, Set[str]] = {
            "urgency_marketing": {
                "limited", "limited time", "limited run", "limited edition", "order now",
                "last chance", "hurry", "while supplies last", "before they're gone",
                "selling out", "selling fast", "act now", "don't wait", "today only",
                "expires soon", "final hours", "almost gone",
            },
            "exclusive_marketing": {
                "exclusive", "exclusively", "exclusive offer", "exclusive deal",
                "members only", "vip", "special access", "invitation only",
                "premium", "privileged", "limited access", "select customers",
                "insider", "private sale", "early access",
            },
        }

        # Load edited or new dictionaries
        current_dicts: Dict[str, Set[str]] = {}

        for label, keywords in default_dicts.items():
            kw_text = "\n".join(sorted(keywords))
            new_kw_text = st.text_area(
                f"Keywords for **{label}** (one per line)", 
                kw_text, 
                key=f"marketing_{label}"
            )
            kw_set = {kw.strip().lower() for kw in new_kw_text.split("\n") if kw.strip()}
            if kw_set:
                current_dicts[label] = kw_set

        # Add new marketing category
        new_label = st.text_input("New marketing category name")
        new_kw_input = st.text_area("Keywords for new category (one per line)")
        if new_label and new_kw_input:
            new_kw_set = {kw.strip().lower() for kw in new_kw_input.split("\n") if kw.strip()}
            if new_kw_set:
                current_dicts[new_label.strip().lower()] = new_kw_set

        one_hot = st.checkbox("Add one‑hot encoded columns", value=True)

    # Brand Language Keywords
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        st.subheader("Brand Language Keywords")
        
        default_brand_keywords = [
            "personalized", "custom", "tailored", "human", "experience",
            "connection", "care", "thoughtful", "responsive", "concierge"
        ]
        
        brand_kw_text = "\n".join(default_brand_keywords)
        brand_keywords_input = st.text_area(
            "Brand keywords (one per line)", 
            brand_kw_text,
            key="brand_keywords"
        )
        brand_keywords = [kw.strip().lower() for kw in brand_keywords_input.split("\n") if kw.strip()]

###############################################################################
# Helper Functions
###############################################################################

def classify_statement(text: str, dictionaries: Dict[str, Set[str]]) -> List[str]:
    """Return list of dictionary names whose keywords appear in text."""
    text_lower = text.lower()
    matched: List[str] = []
    for label, keywords in dictionaries.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(label)
    return matched

def count_brand_matches(caption, keywords):
    """Count matched brand words per statement."""
    # Lowercase and remove punctuation using regex
    words = re.findall(r'\b\w+\b', str(caption).lower())
    matched = sum(1 for word in words if any(k in word for k in keywords))  # partial match
    return pd.Series([len(words), matched])

def analyze_brand_language(df, keywords, id_col='ID', caption_col='Caption'):
    """Analyze brand language in captions."""
    # Count matches
    df[['total_words', 'matched_words']] = df[caption_col].apply(
        lambda x: count_brand_matches(x, keywords)
    )
    
    # Calculate match percentage
    df['match_pct'] = df['matched_words'] / df['total_words'].replace(0, 1)  # Avoid division by zero
    
    # Group by ID and aggregate
    df_grouped = df.groupby(id_col).agg({
        'matched_words': 'sum',
        'total_words': 'sum'
    }).reset_index()
    
    # Recalculate match_pct
    df_grouped['match_pct'] = df_grouped['matched_words'] / df_grouped['total_words'].replace(0, 1)
    
    return df, df_grouped

###############################################################################
# Analysis Functions
###############################################################################

def run_marketing_classifier(file_buffer: io.BytesIO, dictionaries: Dict[str, Set[str]]):
    """Run marketing keyword classification."""
    df = pd.read_csv(file_buffer)

    if "Statement" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'Statement'.")
        return None

    with st.spinner("Classifying marketing keywords…"):
        df["labels"] = df["Statement"].astype(str).apply(classify_statement, dictionaries=dictionaries)
        if one_hot:
            for label in dictionaries:
                df[label] = df["labels"].apply(lambda cats, lbl=label: lbl in cats)

    return df

def run_brand_analysis(file_buffer: io.BytesIO, keywords: List[str], engagement_buffer=None):
    """Run brand language analysis."""
    df = pd.read_csv(file_buffer)
    
    required_cols = ['Caption']
    id_col = 'ID' if 'ID' in df.columns else df.columns[0]  # Use first column if no ID
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ The uploaded CSV must contain: {required_cols}")
        return None, None, None
    
    with st.spinner("Analyzing brand language…"):
        df_detailed, df_grouped = analyze_brand_language(df, keywords, id_col)
    
    # Merge with engagement data if provided
    df_final = df_grouped.copy()
    correlations = {}
    
    if engagement_buffer is not None:
        try:
            df_engage = pd.read_csv(engagement_buffer)
            
            # Try to merge - look for common columns
            merge_col = None
            for col in ['shortcode', 'ID', 'id']:
                if col in df_engage.columns:
                    merge_col = col
                    break
            
            if merge_col:
                df_final = pd.merge(df_grouped, df_engage, left_on=id_col, right_on=merge_col, how='inner')
                
                # Calculate correlations
                numeric_cols = df_final.select_dtypes(include=['float64', 'int64']).columns
                engagement_cols = [col for col in numeric_cols if 'like' in col.lower() or 'comment' in col.lower() or 'engagement' in col.lower()]
                
                for col in engagement_cols:
                    if col in df_final.columns and df_final[col].notna().sum() > 1:
                        corr = df_final['match_pct'].corr(df_final[col])
                        if pd.notna(corr):
                            correlations[col] = corr
            
        except Exception as e:
            st.warning(f"Could not merge engagement data: {str(e)}")
    
    return df_detailed, df_final, correlations

###############################################################################
# Visualization Functions
###############################################################################

def create_brand_visualizations(df_final, correlations):
    """Create visualizations for brand analysis."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of match percentages using Streamlit histogram
        st.subheader("📊 Distribution of Brand Match %")
        st.histogram_chart(df_final['match_pct'])
    
    with col2:
        # Top posts by brand match percentage
        st.subheader("🏆 Top 10 Posts by Brand Match %")
        top_posts = df_final.nlargest(10, 'match_pct').reset_index()
        
        # Create a simple bar chart using streamlit
        chart_data = pd.DataFrame({
            'Post Index': [f"Post {i+1}" for i in range(len(top_posts))],
            'Match %': top_posts['match_pct'].values
        })
        st.bar_chart(chart_data.set_index('Post Index'))
    
    # Correlation analysis if engagement data available
    if correlations:
        st.subheader("📊 Brand Language vs Engagement Correlations")
        
        # Display correlations as metrics
        cols = st.columns(len(correlations))
        for i, (metric, corr) in enumerate(correlations.items()):
            with cols[i % len(cols)]:
                # Determine correlation strength
                if abs(corr) > 0.7:
                    strength = "Very Strong"
                elif abs(corr) > 0.5:
                    strength = "Strong"
                elif abs(corr) > 0.3:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                st.metric(
                    label=metric.replace('_', ' ').title(),
                    value=f"{corr:.3f}",
                    delta=f"{strength} correlation"
                )
        
        # Create correlation bar chart
        if len(correlations) > 1:
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Metric', 'Correlation'])
            st.bar_chart(corr_df.set_index('Metric'))

###############################################################################
# Main App Logic
###############################################################################

def main():
    if analysis_mode == "Marketing Keyword Classification":
        if uploaded_file is not None:
            try:
                result_df = run_marketing_classifier(uploaded_file, current_dicts)
                if result_df is not None:
                    st.success("✅ Marketing classification complete!")
                    
                    # Preview
                    st.subheader("🔍 Preview (first 10 rows)")
                    st.dataframe(result_df.head(10), use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Classification Summary")
                    for label in current_dicts.keys():
                        if label in result_df.columns:
                            count = result_df[label].sum()
                            pct = (count / len(result_df)) * 100
                            st.metric(f"{label.replace('_', ' ').title()}", f"{count} ({pct:.1f}%)")
                    
                    # Download
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Marketing Classification Results",
                        data=csv_bytes,
                        file_name="marketing_classified.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.exception(e)
        else:
            st.info("👆 Upload a CSV file to get started with marketing classification.")
    
    elif analysis_mode == "Brand Language Analysis":
        if brand_file is not None:
            try:
                df_detailed, df_final, correlations = run_brand_analysis(
                    brand_file, brand_keywords, engagement_file
                )
                
                if df_detailed is not None:
                    st.success("✅ Brand language analysis complete!")
                    
                    # Key metrics
                    st.subheader("📊 Brand Language Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_match_pct = df_final['match_pct'].mean()
                        st.metric("Average Match %", f"{avg_match_pct:.2%}")
                    
                    with col2:
                        total_posts = len(df_final)
                        st.metric("Total Posts", total_posts)
                    
                    with col3:
                        total_words = df_final['total_words'].sum()
                        st.metric("Total Words", f"{total_words:,}")
                    
                    with col4:
                        total_matches = df_final['matched_words'].sum()
                        st.metric("Brand Keywords Found", total_matches)
                    
                    # Visualizations
                    create_brand_visualizations(df_final, correlations)
                    
                    # Correlation results
                    if correlations:
                        st.subheader("🔗 Correlation Analysis")
                        for metric, corr in correlations.items():
                            st.write(f"**{metric}**: {corr:.3f}")
                    
                    # Preview data
                    st.subheader("🔍 Preview Results")
                    st.dataframe(df_final.head(10), use_container_width=True)
                    
                    # Download
                    csv_bytes = df_final.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Brand Analysis Results",
                        data=csv_bytes,
                        file_name="brand_analysis_results.csv",
                        mime="text/csv",
                    )
                    
            except Exception as e:
                st.exception(e)
        else:
            st.info("👆 Upload a brand analysis CSV file to get started.")
    
    elif analysis_mode == "Combined Analysis":
        if uploaded_file is not None and brand_file is not None:
            try:
                # Run both analyses
                marketing_df = run_marketing_classifier(uploaded_file, current_dicts)
                brand_detailed, brand_final, correlations = run_brand_analysis(
                    brand_file, brand_keywords, engagement_file
                )
                
                if marketing_df is not None and brand_detailed is not None:
                    st.success("✅ Combined analysis complete!")
                    
                    # Create tabs for results
                    tab1, tab2, tab3 = st.tabs(["Marketing Analysis", "Brand Analysis", "Combined Insights"])
                    
                    with tab1:
                        st.subheader("📊 Marketing Classification Summary")
                        for label in current_dicts.keys():
                            if label in marketing_df.columns:
                                count = marketing_df[label].sum()
                                pct = (count / len(marketing_df)) * 100
                                st.metric(f"{label.replace('_', ' ').title()}", f"{count} ({pct:.1f}%)")
                        
                        st.dataframe(marketing_df.head(10), use_container_width=True)
                    
                    with tab2:
                        # Brand analysis metrics and visualizations
                        avg_match_pct = brand_final['match_pct'].mean()
                        st.metric("Average Brand Match %", f"{avg_match_pct:.2%}")
                        
                        create_brand_visualizations(brand_final, correlations)
                        st.dataframe(brand_final.head(10), use_container_width=True)
                    
                    with tab3:
                        st.subheader("🎯 Combined Insights")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Marketing Keywords Found:**")
                            for label in current_dicts.keys():
                                if label in marketing_df.columns:
                                    count = marketing_df[label].sum()
                                    st.write(f"- {label.replace('_', ' ').title()}: {count}")
                        
                        with col2:
                            st.write("**Brand Language Performance:**")
                            st.write(f"- Average match rate: {avg_match_pct:.2%}")
                            st.write(f"- Total brand keywords: {brand_final['matched_words'].sum()}")
                            if correlations:
                                best_corr = max(correlations.items(), key=lambda x: abs(x[1]))
                                st.write(f"- Strongest correlation: {best_corr[0]} ({best_corr[1]:.3f})")
                    
                    # Combined download
                    with io.BytesIO() as buffer:
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            marketing_df.to_excel(writer, sheet_name='Marketing_Analysis', index=False)
                            brand_final.to_excel(writer, sheet_name='Brand_Analysis', index=False)
                        
                        st.download_button(
                            label="📥 Download Combined Analysis (Excel)",
                            data=buffer.getvalue(),
                            file_name="combined_analysis_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
            except Exception as e:
                st.exception(e)
        else:
            st.info("👆 Upload both CSV files to run combined analysis.")

if __name__ == "__main__":
    main()
