import io
import re
import json
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import streamlit as st

###############################################################################
# Streamlit – Text Classification Tool                                       #
###############################################################################
st.set_page_config(page_title="Text Classification Tool", layout="wide")

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
    .upload-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px dashed #d1d5db;
        text-align: center;
        margin-bottom: 1rem;
    }
    .dict-preview {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 📊 Text Classification Tool")
st.markdown("Upload your dataset and classify text using customizable dictionaries")

# ---------------------------------------------------------------------------
# 🛠️ Sidebar – Dictionary Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Dictionary Management")
    
    # Initialize session state for dictionaries
    if 'dictionaries' not in st.session_state:
        st.session_state.dictionaries = {
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
            "brand_language": {
                "personalized", "custom", "tailored", "human", "experience",
                "connection", "care", "thoughtful", "responsive", "concierge"
            }
        }
    
    # Edit Dictionaries Section
    st.subheader("Edit Dictionaries")
    
    # Select dictionary to edit
    st.write("Select dictionary to edit:")
    selected_dict = st.selectbox(
        "Dictionary",
        options=list(st.session_state.dictionaries.keys()),
        key="dict_selector"
    )
    
    if selected_dict:
        st.write(f"**{selected_dict}** terms:")
        
        # Display current terms for editing
        current_terms = st.session_state.dictionaries[selected_dict]
        terms_text = "\n".join(sorted(current_terms))
        
        st.write("Edit terms (one per line):")
        new_terms_text = st.text_area(
            "Terms",
            value=terms_text,
            height=150,
            key=f"edit_{selected_dict}",
            label_visibility="collapsed"
        )
        
        if st.button(f"Update {selected_dict}", key=f"update_{selected_dict}"):
            new_terms = {term.strip().lower() for term in new_terms_text.split('\n') if term.strip()}
            st.session_state.dictionaries[selected_dict] = new_terms
            st.success(f"✅ Updated {selected_dict} with {len(new_terms)} terms")
            st.rerun()
    
    st.markdown("---")
    
    # Add New Dictionary Section
    st.subheader("Add New Dictionary")
    
    new_dict_name = st.text_input("Dictionary name:", key="new_dict_name")
    new_dict_terms = st.text_area(
        "Terms (one per line):",
        height=100,
        key="new_dict_terms"
    )
    
    if st.button("Add Dictionary", key="add_dict_btn"):
        if new_dict_name and new_dict_terms:
            new_terms = {term.strip().lower() for term in new_dict_terms.split('\n') if term.strip()}
            if new_terms:
                dict_key = new_dict_name.strip().lower().replace(' ', '_')
                st.session_state.dictionaries[dict_key] = new_terms
                st.success(f"✅ Added dictionary '{dict_key}' with {len(new_terms)} terms")
                st.rerun()
        else:
            st.error("Please provide both dictionary name and terms")
    
    st.markdown("---")
    
    # Remove Dictionary Section
    st.write("Remove dictionary:")
    dict_to_remove = st.selectbox(
        "Select dictionary to remove",
        options=[""] + list(st.session_state.dictionaries.keys()),
        key="remove_dict_selector"
    )
    
    if st.button("Remove Dictionary", key="remove_dict_btn"):
        if dict_to_remove and dict_to_remove in st.session_state.dictionaries:
            del st.session_state.dictionaries[dict_to_remove]
            st.success(f"✅ Removed dictionary '{dict_to_remove}'")
            st.rerun()
        else:
            st.error("Please select a dictionary to remove")
    
    st.markdown("---")
    
    # Export/Import Section
    st.subheader("Export/Import")
    
    # Export Dictionaries
    if st.button("Export Dictionaries", key="export_btn"):
        # Convert sets to lists for JSON serialization
        export_data = {k: list(v) for k, v in st.session_state.dictionaries.items()}
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Dictionaries JSON",
            data=json_str,
            file_name="dictionaries.json",
            mime="application/json"
        )
    
    # Import Dictionaries
    st.write("Import dictionaries (JSON):")
    
    uploaded_dict_file = st.file_uploader(
        "Choose JSON file",
        type=["json"],
        key="import_dict_file",
        help="Upload a JSON file with dictionary data"
    )
    
    if uploaded_dict_file is not None:
        try:
            dict_data = json.load(uploaded_dict_file)
            
            if st.button("Import Dictionaries", key="import_btn"):
                # Convert lists back to sets
                imported_dicts = {k: set(v) if isinstance(v, list) else set(v) for k, v in dict_data.items()}
                st.session_state.dictionaries.update(imported_dicts)
                st.success(f"✅ Imported {len(imported_dicts)} dictionaries")
                st.rerun()
                
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Error importing file: {str(e)}")
    
    st.markdown("---")
    
    # Analysis Options
    st.subheader("Analysis Options")
    analysis_mode = st.selectbox(
        "Analysis type:",
        ["Marketing Keyword Classification", "Brand Language Analysis", "Combined Analysis"]
    )
    
    one_hot = st.checkbox("Add one-hot encoded columns", value=True)

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
# Main Content Area
###############################################################################

# Create two main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📁 Upload Dataset")
    
    # File upload section
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.write("Choose a CSV file")
    st.write("Limit 200MB per file • CSV")
    
    if analysis_mode in ["Marketing Keyword Classification", "Combined Analysis"]:
        uploaded_file = st.file_uploader(
            "Marketing Data CSV (with 'Statement' column)",
            type=["csv"],
            key="marketing_file",
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    if analysis_mode in ["Brand Language Analysis", "Combined Analysis"]:
        brand_file = st.file_uploader(
            "Brand Analysis CSV (with 'Caption' column)",
            type=["csv"],
            key="brand_file",
            label_visibility="collapsed"
        )
        if brand_file:
            st.success(f"✅ Uploaded: {brand_file.name}")
        
        engagement_file = st.file_uploader(
            "Engagement Data CSV (optional)",
            type=["csv"],
            key="engagement_file",
            label_visibility="collapsed"
        )
        if engagement_file:
            st.success(f"✅ Uploaded: {engagement_file.name}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 📚 Current Dictionaries")
    
    # Display current dictionaries
    for dict_name, terms in st.session_state.dictionaries.items():
        with st.expander(f"📋 {dict_name} ({len(terms)} terms)", expanded=False):
            # Display terms in a more compact format
            terms_display = ", ".join(sorted(list(terms)[:10]))  # Show first 10 terms
            if len(terms) > 10:
                terms_display += f"... and {len(terms) - 10} more"
            st.write(terms_display)

###############################################################################
# Analysis Functions
###############################################################################

def run_marketing_classifier(file_buffer: io.BytesIO, dictionaries: Dict[str, Set[str]]):
    """Run marketing keyword classification."""
    df = pd.read_csv(file_buffer)
    
    if "Statement" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'Statement'.")
        return None
    
    with st.spinner("🔄 Classifying statements..."):
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
        if 'uploaded_file' in locals() and uploaded_file is not None:
            try:
                result_df = run_marketing_classifier(uploaded_file, st.session_state.dictionaries)
                if result_df is not None:
                    st.success("✅ Classification complete!")
                    
                    # Key metrics
                    st.markdown("## 📊 Classification Results")
                    
                    metrics_cols = st.columns(len(st.session_state.dictionaries) + 1)
                    
                    with metrics_cols[0]:
                        st.metric("Total Statements", len(result_df))
                    
                    for i, label in enumerate(st.session_state.dictionaries.keys(), 1):
                        if label in result_df.columns:
                            count = result_df[label].sum()
                            pct = (count / len(result_df)) * 100
                            with metrics_cols[i % len(metrics_cols)]:
                                st.metric(f"{label.replace('_', ' ').title()}", f"{count}", f"{pct:.1f}%")
                    
                    # Preview
                    st.markdown("### 🔍 Preview (first 10 rows)")
                    st.dataframe(result_df.head(10), use_container_width=True)
                    
                    # Download
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Classification Results",
                        data=csv_bytes,
                        file_name="classified_output.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        else:
            st.info("👆 Upload a CSV file to get started.")
    
    elif analysis_mode == "Brand Language Analysis":
        if 'brand_file' in locals() and brand_file is not None:
            try:
                # Get brand keywords from dictionaries
                brand_keywords = []
                if 'brand_language' in st.session_state.dictionaries:
                    brand_keywords = list(st.session_state.dictionaries['brand_language'])
                
                df_detailed, df_final, correlations = run_brand_analysis(
                    brand_file, brand_keywords, engagement_file if 'engagement_file' in locals() else None
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
        files_ready = ('uploaded_file' in locals() and uploaded_file is not None and 
                      'brand_file' in locals() and brand_file is not None)
        
        if files_ready:
            try:
                # Run both analyses
                marketing_df = run_marketing_classifier(uploaded_file, st.session_state.dictionaries)
                
                brand_keywords = []
                if 'brand_language' in st.session_state.dictionaries:
                    brand_keywords = list(st.session_state.dictionaries['brand_language'])
                
                brand_detailed, brand_final, correlations = run_brand_analysis(
                    brand_file, brand_keywords, engagement_file if 'engagement_file' in locals() else None
                )
                
                if marketing_df is not None and brand_detailed is not None:
                    st.success("✅ Combined analysis complete!")
                    
                    # Create tabs for results
                    tab1, tab2, tab3 = st.tabs(["📈 Marketing Analysis", "🎯 Brand Analysis", "💡 Combined Insights"])
                    
                    with tab1:
                        st.markdown("### Marketing Classification Summary")
                        
                        metrics_cols = st.columns(len(st.session_state.dictionaries))
                        for i, label in enumerate(st.session_state.dictionaries.keys()):
                            if label in marketing_df.columns:
                                count = marketing_df[label].sum()
                                pct = (count / len(marketing_df)) * 100
                                with metrics_cols[i % len(metrics_cols)]:
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
                            for label in st.session_state.dictionaries.keys():
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
