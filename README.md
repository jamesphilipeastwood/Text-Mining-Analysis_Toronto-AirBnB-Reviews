# Coming Clean: What Toronto Airbnb Reviews Reveal

- **Course:** OPIM 5671 Data Mining and Time Series Forecasting (University of Connecticut)
- **Instructor:** Jaeung Sim
- **Team Seven:** Matthew Koh, Preshitha Sureshkumar, James Eastwood, Rama Akach, Kaan Ates Boyaci, Luke Bauer
- **Main Analysis Notebook:** [OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final Notebook.ipynb](OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final%20Notebook.ipynb)

## Project Overview

Aggregate star ratings on Airbnb suffer from severe upward compression and survivorship bias, making mediocre or declining properties look nearly identical to standout stays. Standard review keyword search also struggles when positive words appear inside negated complaints.

To look past inflated summary scores, this project links individual timestamped guest reviews to property metadata and applies domain-aware text mining across Toronto Airbnb listings. Following Team Seven's proposal and course guidelines, the notebook investigates three questions:

- **Themes and Dissatisfaction:** What main themes appear in Toronto Airbnb guest reviews, and where does expressed dissatisfaction concentrate?
- **Longitudinal Property Trajectory:** When reviews are linked chronologically to specific listings over time, how do sentiment and complaints evolve, and can deteriorating units be detected before their cumulative star averages fall?
- **The Michelin Star Host Question:** How important is personal guest-host interaction (measured via host-name mentions, communication ratings, and host portfolio scale) relative to physical cleanliness and expectation-alignment dimensions?

## Dataset and Reproducibility

- **Source:** Inside Airbnb Toronto dataset on Kaggle (`vladimirkondratev/inside-airbnb-toronto-sep-2026`), downloaded automatically inside the notebook through `kagglehub`.
- **Reproducible Execution:** The v4 notebook pins the dataset version, fixes random seeds, and uses deterministic hashing for sampling and listing-grouped splitting so results reproduce cleanly in both Google Colab and local Python environments.
- **Property Linkage and Anonymization:** Each review row is merged with its listing's property attributes, host attributes, and multi-category rating scores by listing ID, while reviewer names are dropped immediately upon merging to protect guest privacy.

## Analytical Pipeline

- **Cleaning and English Filtering:** Strips HTML line-break artifacts before tokenization and applies a fast, deterministic rule-based English filter so lexicon and topic models are not distorted by foreign-language fragments or machine-translation noise.
- **Stratified Sampling and Leakage-Free Validation:** Constructs a stratified working sample that intentionally includes lower-rated and complaint-flagged stays so topic modeling captures rich complaint vocabulary, then splits data into training, validation, and held-out test partitions grouped strictly by listing ID so zero properties overlap across splits.
- **Domain-Aware Text Preprocessing:**
  - Normalizes host first names and common nicknames into a generic host token (excluding ordinary English words that overlap with first names) so host references can be tracked without fragmenting topics by individual names.
  - Binds sentiment-flipping negations to the following lemmatized word (such as `not_clean`, `not_work`, and `not_recommend`), separating genuine complaints from double-negative praise.
  - Applies WordNet lemmatization and removes generic stopwords and domain boilerplate terms.
- **Text Representations and Lexicon Features:** Builds Bag-of-Words count matrices, explicit TF/IDF/TF-IDF representations fitted strictly on training reviews, Truncated SVD components, TextBlob sentiment polarity on unstripped review text, review verbosity metrics, and normalized NRC emotion rates across anger, fear, anticipation, trust, surprise, sadness, joy, and disgust.
- **LDA Topic Modeling:** Trains Latent Dirichlet Allocation on word counts across candidate topic counts, selects the three-topic model using UMass coherence and qualitative inspection before test evaluation, labels topics programmatically using anchor words, and cross-analyzes themes against dissatisfaction flags, emotion density, host mentions, and distinctive TF-IDF terms.
- **Longitudinal and Host-Scale Checks:** Tracks city-wide annual complaint trends alongside within-property comparisons between each well-reviewed listing's early stays and recent stays, and evaluates host relevance across host-name mentions, standardized multiple regression of sub-ratings, and single-unit versus multi-unit commercial operators.
- **Predictive Modeling:** Evaluates progressively richer linear regression models—from a training-mean baseline through sentiment, emotions and length, host/negation indicators plus LDA topic weights, and a full hybrid model with Truncated SVD components—selecting the best specification on validation error before unlocking the held-out test set.

## Key Findings

### Review Themes and Where Dissatisfaction Concentrates

- **Topic 1 — Enthusiastic Praise and Host Appreciation:** Captures warm guest gratitude, overall comfort, smooth check-in, and helpful hosts; represents the largest share of training reviews and shows low dissatisfaction alongside the highest personal host-name mention rate and the highest cleanliness and overall listing ratings.
- **Topic 2 — Practical Stay Quality:** Focuses on walkability, parking, transit convenience, restaurants, and downtown access; mostly positive with low dissatisfaction and clean average scores.
- **Topic 3 — Unit Operations and Problems:** Centers on beds, bathrooms, appliances working properly, physical upkeep, and nightly logistics; accounts for the overwhelming concentration of guest dissatisfaction, the longest complaint narratives, lowest mean sentiment polarity, elevated anger and disgust emotion density, lowest host-name mention rate, and the lowest cleanliness and overall ratings.

### Property Reviews Changing Over Time

- Across Toronto as a whole, the share of reviews carrying explicit complaint cues shows a gentle upward drift in recent years.
- Splitting individual listings chronologically into their early half versus recent half reveals sharp divergences in both improvement and decline. Inspecting raw review text confirms that properties suffering recent waves of cleanliness, odor, maintenance, or noise complaints often continue displaying respectable cumulative star ratings on the platform.

### The Michelin Star Host Answer

- Reviews that address the host by name or nickname tend to be more detailed, carry noticeably lower dissatisfaction rates, and come from listings with higher overall, communication, and cleanliness ratings.
- Standardized regression across listing sub-scores indicates that expectation-alignment dimensions (**Value** and **Accuracy**) hold the largest weights predicting overall listing ratings, while **Host Communication** and **Cleanliness** follow closely as virtually tied secondary drivers. Check-in and location contribute smaller marginal effects once other dimensions are controlled for.
- Guests staying with single-unit hosts mention their host personally far more often and award higher communication, cleanliness, and overall ratings than guests booking properties managed by commercial multi-unit operators. Personal hospitality helps cushion minor friction, but it cannot compensate for inaccurate descriptions, unfair pricing, or a poorly cleaned unit.

### Predictive Rating Modeling Insights

- Combining sentiment polarity, NRC emotion rates, review length, host-mention and negation indicators, LDA topic weights, and Truncated SVD components achieves the lowest validation error and outperforms the development-mean baseline on unseen test listings.
- Predictive gains remain modest because Inside Airbnb provides each listing's cumulative lifetime average rating rather than individual review scores, and strong score compression pulls linear predictions toward the high market average when a glowing review belongs to an otherwise lower-rated listing.

## Limitations and Methodological Notes

- **Observational Associations:** Cross-tabulations and regressions show descriptive and predictive relationships rather than causal treatment effects.
- **Aggregate Rating Granularity and Bias:** Because scores exist only at the listing level and delisted properties drop out of snapshots, review-level dissatisfaction tags and longitudinal half-period comparisons are needed to detect problems hidden behind high average scores.
- **Sampling and Measurement Caveats:** Oversampling lower-rated and complaint-flagged stays intentionally elevates working-sample dissatisfaction rates above city-wide averages, and because dissatisfaction indicators incorporate polarity and complaint keywords, comparative patterns—rather than raw sample rates—provide the reliable signal.
- **English-Only Lexical Scope:** Rule-based filtering focuses analysis on English reviews to avoid machine-translation distortions, though lexicon methods remain imperfect on sarcasm and understated complaints.

## Actionable Recommendations

- **For Airbnb:** Introduce a rolling recent-review health signal built from polarity and negation-bound complaint terms to flag deteriorating units early, and make review search negation-aware so negative phrases do not match positive keyword queries.
- **For Hosts:** Audit photos, descriptions, and pricing for complete accuracy, follow strict pre-arrival checklists covering bathrooms, odors, beds, climate control, and appliances, and pair physical readiness with personal, name-based communication early in the stay.
- **For Guests:** Always sort property reviews by most recent and scan the latest stays for unit-operation complaints or impersonal commercial management signs instead of relying on lifetime star averages alone.

## How to Run

Open [OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final Notebook.ipynb](OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final%20Notebook.ipynb) in Google Colab, JupyterLab, or VS Code and run all cells from top to bottom. On startup, the setup cells automatically check and install required packages (`kagglehub`, `nltk`, `textblob`, `nrclex`), fetch required NLTK lexical corpora, and download the Inside Airbnb Toronto dataset from Kaggle.

## Repository Structure

- [OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final Notebook.ipynb](OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_Final%20Notebook.ipynb) — Final v4 analysis notebook with deterministic sampling, listing-grouped validation splits, and complete findings.
- [Previous Notebook Versions/](Previous%20Notebook%20Versions/) — Earlier notebook versions preserved for reference (`OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb` and `OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews_v2.ipynb`).
- [OPIM 5671 - Team Project 2.docx.pdf](OPIM%205671%20-%20Team%20Project%202.docx.pdf) — Team Seven project proposal and analytical plan.
- [Documents/](Documents/) — Course materials including [OPIM-5671 Team Project #2 Guideline.pdf](Documents/OPIM-5671%20Team%20Project%20%232%20Guideline.pdf), syllabus, and lecture decks.
- [Reference Notebooks/](Reference%20Notebooks/) — Course reference notebooks used to align methods and validation standards.
