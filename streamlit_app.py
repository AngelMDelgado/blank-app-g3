import io
import re
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import streamlit as st

###############################################################################
# Streamlit – Marketing & Brand Analysis Tool                                #
###############################################################################
st.set_page_config(page_title="Marketing & Brand Analysis Tool", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-container {
        background-color: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .upload-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px dashed #d1d5db;
        text-align: center;
        margin-bottom: 1rem;
    }
    .dict-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Marketing & Brand Analysis Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your dataset and analyze text using customizable dictionaries</div>', unsafe_allow_html=True)

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
    words = re.findall(r'\b\w+\b', str(caption).lower())
    matched = sum(1 for word in words if any(k in word for k in keywords))
    return pd.Series([len(words), matched])

def analyze_brand_language(df, keywords, id_col='ID', caption_col='Caption'):
    """Analyze brand language in captions."""
    df[['total_words', 'matched_words']] = df[caption_col].apply(
        lambda x: count_brand_matches(x, keywords)
    )
    df['match_pct'] = df['matched_words'] / df['total_words'].replace(0, 1)
    
    df_grouped = df.groupby(id_col).agg({
        'matched_words': 'sum',
        'total_words': 'sum'
    }).reset_index()
    df_grouped['match_pct'] = df_grouped['matched_words'] / df_grouped['total_words'].replace(0, 1)
    
    return df, df_grouped

###############################################################################
# Main Layout
###############################################################################

# Create two main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📁 Upload Dataset")
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Choose analysis type:",
        ["Marketing Keyword Classification", "Brand Language Analysis", "Combined Analysis"],
        help="Select the type of analysis you want to perform"
    )
    
    # File upload section
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        uploaded_file = st.file_uploader(
            "📄 Marketing Data CSV (with 'Statement' column)",
            type=["csv"],
            key="marketing_file",
            help="Upload a CSV file containing statements to classify"
        )
    
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        brand_file = st.file_uploader(
            "📄 Brand Analysis CSV (with 'Caption' column)",
            type=["csv"],
            key="brand_file",
            help="Upload a CSV file containing captions to analyze"
        )
        
        engagement_file = st.file_uploader(
            "📄 Engagement Data CSV (optional)",
            type=["csv"],
            key="engagement_file",
            help="Optional: Upload engagement metrics for correlation analysis"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 📚 Current Dictionaries")
    
    # Default dictionaries
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
    
    # Display current dictionaries
    current_dicts: Dict[str, Set[str]] = {}
    
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        for label, keywords in default_dicts.items():
            with st.expander(f"📋 {label.replace('_', ' ').title()} ({len(keywords)} terms)", expanded=False):
                st.write("**Current terms:**")
                st.write(", ".join(sorted(keywords)))
                current_dicts[label] = keywords
    
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        default_brand_keywords = [
            "personalized", "custom", "tailored", "human", "experience",
            "connection", "care", "thoughtful", "responsive", "concierge"
        ]
        
        with st.expander(f"🎯 Brand Keywords ({len(default_brand_keywords)} terms)", expanded=False):
            st.write("**Current terms:**")
            st.write(", ".join(default_brand_keywords))
        
        brand_keywords = default_brand_keywords

# Configuration section
st.markdown("---")
st.markdown("## ⚙️ Edit Dictionaries")

config_col1, config_col2 = st.columns([1, 1])

with config_col1:
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        st.markdown("### Marketing Keywords")
        
        # Allow editing of dictionaries
        edited_dicts: Dict[str, Set[str]] = {}
        
        for label, keywords in default_dicts.items():
            st.markdown(f"**{label.replace('_', ' ').title()}**")
            kw_text = "\n".join(sorted(keywords))
            new_kw_text = st.text_area(
                f"Edit terms (one per line)",
                kw_text,
                key=f"marketing_{label}",
                height=100
            )
            kw_set = {kw.strip().lower() for kw in new_kw_text.split("\n") if kw.strip()}
            if kw_set:
                edited_dicts[label] = kw_set
        
        current_dicts = edited_dicts
        
        # Add new category
        st.markdown("**Add New Marketing Category**")
        new_label = st.text_input("Category name", placeholder="e.g., price_marketing")
        new_kw_input = st.text_area("Keywords (one per line)", placeholder="sale\ndiscount\nspecial price", height=80)
        
        if new_label and new_kw_input:
            new_kw_set = {kw.strip().lower() for kw in new_kw_input.split("\n") if kw.strip()}
            if new_kw_set:
                current_dicts[new_label.strip().lower()] = new_kw_set
                st.success(f"✅ Added '{new_label}' with {len(new_kw_set)} terms")

with config_col2:
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        st.markdown("### Brand Language Keywords")
        
        brand_kw_text = "\n".join(default_brand_keywords)
        brand_keywords_input = st.text_area(
            "Edit brand keywords (one per line)",
            brand_kw_text,
            key="brand_keywords",
            height=200
        )
        brand_keywords = [kw.strip().lower() for kw in brand_keywords_input.split("\n") if kw.strip()]
    
    # Options
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        one_hot = st.checkbox("📊 Add one-hot encoded columns", value=True, 
                             help="Create binary columns for each category")

###############################################################################
# Analysis Functions
###############################################################################

def run_marketing_classifier(file_buffer: io.BytesIO, dictionaries: Dict[str, Set[str]]):
    """Run marketing keyword classification."""
    df = pd.read_csv(file_buffer)
    
    if "Statement" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'Statement'.")
        return None
    
    with st.spinner("🔄 Classifying marketing keywords..."):
        df["labels"] = df["Statement"].astype(str).apply(classify_statement, dictionaries=dictionaries)
        if one_hot:
            for label in dictionaries:
                df[label] = df["labels"].apply(lambda cats, lbl=label: lbl in cats)
    
    return df

def run_brand_analysis(file_buffer: io.BytesIO, keywords: List[str], engagement_buffer=None):
    """Run brand language analysis."""
    df = pd.read_csv(file_buffer)
    
    required_cols = ['Caption']
    id_col = 'ID' if 'ID' in df.columns else df.columns[0]
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ The uploaded CSV must contain: {required_cols}")
        return None, None, None
    
    with st.spinner("🔄 Analyzing brand language..."):
        df_detailed, df_grouped = analyze_brand_language(df, keywords, id_col)
    
    df_final = df_grouped.copy()
    correlations = {}
    
    if engagement_buffer is not None:
        try:
            df_engage = pd.read_csv(engagement_buffer)
            
            merge_col = None
            for col in ['shortcode', 'ID', 'id']:
                if col in df_engage.columns:
                    merge_col = col
                    break
            
            if merge_col:
                df_final = pd.merge(df_grouped, df_engage, left_on=id_col, right_on=merge_col, how='inner')
                
                numeric_cols = df_final.select_dtypes(include=['float64', 'int64']).columns
                engagement_cols = [col for col in numeric_cols if 'like' in col.lower() or 'comment' in col.lower() or 'engagement' in col.lower()]
                
                for col in engagement_cols:
                    if col in df_final.columns and df_final[col].notna().sum() > 1:
                        corr = df_final['match_pct'].corr(df_final[col])
                        if pd.notna(corr):
                            correlations[col] = corr
        
        except Exception as e:
            st.warning(f"⚠️ Could not merge engagement data: {str(e)}")
    
    return df_detailed, df_final, correlations

###############################################################################
# Results Display
###############################################################################

def display_results():
    """Display analysis results based on mode."""
    
    if analysis_mode == "Marketing Keyword Classification":
        if uploaded_file is not None:
            try:
                result_df = run_marketing_classifier(uploaded_file, current_dicts)
                if result_df is not None:
                    st.success("✅ Marketing classification complete!")
                    
                    # Key metrics
                    st.markdown("## 📊 Classification Results")
                    
                    metrics_cols = st.columns(len(current_dicts) + 1)
                    
                    with metrics_cols[0]:
                        st.metric("Total Statements", len(result_df))
                    
                    for i, label in enumerate(current_dicts.keys(), 1):
                        if label in result_df.columns:
                            count = result_df[label].sum()
                            pct = (count / len(result_df)) * 100
                            with metrics_cols[i]:
                                st.metric(f"{label.replace('_', ' ').title()}", f"{count}", f"{pct:.1f}%")
                    
                    # Preview
                    st.markdown("### 🔍 Preview Results")
                    st.dataframe(result_df.head(10), use_container_width=True)
                    
                    # Download
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Classification Results",
                        data=csv_bytes,
                        file_name="marketing_classified.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
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
                    st.markdown("## 📊 Brand Analysis Results")
                    
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
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.markdown("### 📈 Match % Distribution")
                        st.bar_chart(df_final['match_pct'])
                    
                    with viz_col2:
                        st.markdown("### 🏆 Top Posts by Match %")
                        top_posts = df_final.nlargest(10, 'match_pct').reset_index()
                        chart_data = pd.DataFrame({
                            'Post': [f"Post {i+1}" for i in range(len(top_posts))],
                            'Match %': top_posts['match_pct'].values
                        })
                        st.bar_chart(chart_data.set_index('Post'))
                    
                    # Correlations
                    if correlations:
                        st.markdown("### 🔗 Engagement Correlations")
                        corr_cols = st.columns(len(correlations))
                        for i, (metric, corr) in enumerate(correlations.items()):
                            with corr_cols[i]:
                                strength = "Very Strong" if abs(corr) > 0.7 else "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                                st.metric(
                                    metric.replace('_', ' ').title(),
                                    f"{corr:.3f}",
                                    f"{strength}"
                                )
                    
                    # Preview
                    st.markdown("### 🔍 Preview Results")
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
                st.error(f"❌ Error processing file: {str(e)}")
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
                    tab1, tab2, tab3 = st.tabs(["📈 Marketing Analysis", "🎯 Brand Analysis", "💡 Combined Insights"])
                    
                    with tab1:
                        st.markdown("### Marketing Classification Summary")
                        
                        metrics_cols = st.columns(len(current_dicts))
                        for i, label in enumerate(current_dicts.keys()):
                            if label in marketing_df.columns:
                                count = marketing_df[label].sum()
                                pct = (count / len(marketing_df)) * 100
                                with metrics_cols[i]:
                                    st.metric(f"{label.replace('_', ' ').title()}", f"{count}", f"{pct:.1f}%")
                        
                        st.dataframe(marketing_df.head(10), use_container_width=True)
                    
                    with tab2:
                        st.markdown("### Brand Language Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_match_pct = brand_final['match_pct'].mean()
                            st.metric("Average Match %", f"{avg_match_pct:.2%}")
                        with col2:
                            st.metric("Total Posts", len(brand_final))
                        with col3:
                            st.metric("Total Words", f"{brand_final['total_words'].sum():,}")
                        with col4:
                            st.metric("Brand Keywords", brand_final['matched_words'].sum())
                        
                        st.dataframe(brand_final.head(10), use_container_width=True)
                    
                    with tab3:
                        st.markdown("### 🎯 Key Insights")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            st.markdown("**Marketing Performance:**")
                            for label in current_dicts.keys():
                                if label in marketing_df.columns:
                                    count = marketing_df[label].sum()
                                    st.write(f"• {label.replace('_', ' ').title()}: {count} instances")
                        
                        with insight_col2:
                            st.markdown("**Brand Language Effectiveness:**")
                            avg_match = brand_final['match_pct'].mean()
                            st.write(f"• Average brand alignment: {avg_match:.2%}")
                            st.write(f"• Total brand keywords used: {brand_final['matched_words'].sum()}")
                            if correlations:
                                best_corr = max(correlations.items(), key=lambda x: abs(x[1]))
                                st.write(f"• Strongest engagement correlation: {best_corr[0]} ({best_corr[1]:.3f})")
                    
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
                st.error(f"❌ Error processing files: {str(e)}")
        else:
            st.info("👆 Upload both CSV files to run combined analysis.")

# Run analysis
st.markdown("---")
display_results()
