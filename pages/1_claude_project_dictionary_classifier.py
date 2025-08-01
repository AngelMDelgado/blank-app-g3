# 📘 PA6 Method 2 – Classifier Word Metrics App (Colab Version)
# Team Project Notebook – Personalized Style

# ---------------------------------------------
# SECTION 1: Upload Required Files
# ---------------------------------------------
from google.colab import files
import pandas as pd
import re

print("📂 Please upload 'ig_posts_transformed.csv'")
uploaded = files.upload()

# Read the uploaded CSV
df = pd.read_csv('ig_posts_transformed.csv')
df.head()

# ---------------------------------------------
# SECTION 2: Define Keyword Dictionary
# ---------------------------------------------
# These keywords represent the 'Personalized' brand language
keywords = [
"personalized", "custom", "tailored", "human", "experience",
    "connection", "care", "thoughtful", "responsive", "concierge"
]


# ---------------------------------------------
# SECTION 3: Count Matched Words Per Statement
# ---------------------------------------------
def count_matches(caption):
    # Lowercase and remove punctuation using regex
    words = re.findall(r'\b\w+\b', str(caption).lower())
    matched = sum(1 for word in words if any(k in word for k in keywords))  # partial match
    return pd.Series([len(words), matched])

# Apply the function
df[['total_words', 'matched_words']] = df['Caption'].apply(count_matches)

# Calculate match percentage
df['match_pct'] = df['matched_words'] / df['total_words']

# Preview results
print("\n📊 Sample keyword matches:")
display(df[['ID', 'Caption', 'matched_words', 'total_words', 'match_pct']].head())

# ---------------------------------------------
# SECTION 4: Aggregate to IG Post Level
# ---------------------------------------------
# Group by ID and sum matched/total words
df_grouped = df.groupby('ID').agg({
    'matched_words': 'sum',
    'total_words': 'sum'
}).reset_index()

# Recalculate match_pct
df_grouped['match_pct'] = df_grouped['matched_words'] / df_grouped['total_words']

print("\n📦 Aggregated to post level:")
display(df_grouped.head())

# ---------------------------------------------
# SECTION 5: (Optional) Upload & Join Engagement Data
# ---------------------------------------------
print("📂 Please upload 'ig_posts_shi_new.csv'")
uploaded2 = files.upload()

# Read the engagement data
df_engage = pd.read_csv('ig_posts_shi_new.csv')

# Merge using ID (from word metrics) and shortcode (from engagement)
df_merged = pd.merge(df_grouped, df_engage, left_on='ID', right_on='shortcode')

# Select relevant columns for output
final_df = df_merged[['ID', 'match_pct', 'number_likes', 'number_comments']]

print("\n✅ Final merged table:")
display(final_df.head())

# ---------------------------------------------
# SECTION 6: Correlation Analysis
# ---------------------------------------------
likes_corr = final_df['match_pct'].corr(final_df['number_likes'])
comments_corr = final_df['match_pct'].corr(final_df['number_comments'])

print(f"\n📈 Correlation between match_pct and number_likes: {likes_corr:.2f}")
print(f"💬 Correlation between match_pct and number_comments: {comments_corr:.2f}")

# ---------------------------------------------
# SECTION 7: Export Results for Submission
# ---------------------------------------------
# Save results as a CSV
output_filename = 'IG_joined_result_colab.csv'
final_df.to_csv(output_filename, index=False)

# Download file
files.download(output_filename)
