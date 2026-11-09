# Coming Clean — What Toronto Airbnb Reviews Reveal

**OPIM-5671 Data Mining and Time Series Forecasting (Fall 2026) — Team Project #2**  
**Instructor:** Prof. Jaeung Sim (University of Connecticut)  
**Team Seven:** Matthew Koh, Preshitha Sureshkumar, James Eastwood, Rama Akach, Kaan Ates Boyaci, Luke Bauer

---

## 1. Project Overview & Research Questions

Airbnb property star ratings exhibit severe upward compression (in Toronto, the median property rating is **4.87 / 5.00** and the mean is **4.79 / 5.00**). Furthermore, Airbnb's website relies on basic single-word tag filtering for reviews, which fails to capture multi-word negation (`"not clean"`, `"didn't work"`), thematic mixtures, or property deterioration over time.

This project applies machine text-mining techniques to **729,202 Toronto Airbnb guest reviews** (`June 2010 – September 2026`) linked across **17,764 reviewed properties** (`22,050` total listings) from [Inside Airbnb: Toronto (Sep 2026)](https://www.kaggle.com/datasets/vladimirkondratev/inside-airbnb-toronto-sep-2026) to answer three core questions:

1. **Primary Research Question (Themes & Dissatisfaction):** *What themes appear in Toronto Airbnb guest reviews, and which are most common in reviews expressing dissatisfaction?*
2. **Longitudinal Property Trajectory Question (Reviews Changing Over Time):** *When linking individual guest reviews chronologically to specific properties (`listing_id`) from 2010 to 2026, how do review sentiment and complaint themes evolve over time to expose deteriorating properties versus turnaround properties that static cumulative ratings mask?*
3. **The "Michelin Star" Host Interaction Question:** *How relevant is a guest's personal interaction with the host (measured via normalized `host_name` mentions in text and communication ratings) to the overall rating and sentiment of an Airbnb listing compared to physical property attributes (cleanliness, accuracy, location, and value)?*

---

## 2. Intended Decision-Makers & Practical Value

| Stakeholder | Core Decision / Problem | Actionable Insights Delivered |
|---|---|---|
| **Airbnb (Platform Quality & Search)** | *Which listings are deteriorating despite high lifetime ratings? How can review search filters be improved?* | Deploy rolling longitudinal text-sentiment and negation-token alerts (`not_clean`, `not_work`, `smell`, `stain`) per `listing_id` to flag decaying properties before their lifetime star average drops, and upgrade single-word review search to handle negated phrases. |
| **Hosts (Property Operations)** | *Is my listing performing well or poorly in guest text, and what should I prioritize fixing?* | Protect the non-negotiable **hygiene & accuracy floor** (`cleanliness` $\beta = 0.258$, `accuracy` $\beta = 0.312$) to prevent severe rating penalties, while leveraging personal host communication (`host_name_token` mentions reduce dissatisfaction by more than half, from **21.0% to 9.8%**) as a **hospitality multiplier**. |
| **Customers / Guests (Booking Strategy)** | *Where are the genuinely reliable listings, and what red flags hide behind a 4.7+ star average?* | Sort reviews chronologically ("Most Recent") and scan for `Topic 2` operational failure markers (`bed`, `bathroom`, `door`, `smell`, `dirty`, `not_work`) that cumulative star scores conceal. |

---

## 3. Dataset & Schema Linkage

The dataset is downloaded automatically via `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vladimirkondratev/inside-airbnb-toronto-sep-2026")
print("Path to dataset files:", path)
```

- **Primary Corpus (`reviews.csv`):** `729,202` guest review records (`listing_id`, `id`, `date`, `reviewer_id`, `reviewer_name`, `comments`) spanning `2010-06-07` to `2026-09-18`.
- **Property & Host Metadata (`listings.csv`):** `22,050` Toronto Airbnb listings (`17,764` with reviews; **100.0% join overlap** on `listing_id == id`), contributing `host_name`, `host_is_superhost`, `neighbourhood_cleansed`, `room_type`, `price`, and all seven 1–5 star rating dimensions (`review_scores_rating`, `review_scores_cleanliness`, `review_scores_accuracy`, `review_scores_checkin`, `review_scores_communication`, `review_scores_location`, `review_scores_value`).

---

## 4. Ten-Step Analytical Workflow (`OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb`)

Our primary deliverable notebook [`OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb`](OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb) implements all 10 steps from Team Seven's proposal while adhering strictly to the OPIM-5671 course methodology:

1. **Step 1 — Preprocessing & English-Only Executive Decision:** Strip `116,922` HTML `<br/>` tags, remove missing/short/duplicate comments, and apply a deterministic English filter across all `729,202` rows (`612,965` English reviews retained; `81,057` non-English reviews dropped rather than machine-translated to preserve lexical integrity).
2. **Step 2 — Property & Host Metadata Linkage (`listing_id`):** Inner-join `reviews.csv` with `listings.csv` on `listing_id == id` to connect text with property sub-scores and enable chronological property tracking.
3. **Step 3 — Bag-of-Words Representation:** Build sparse document–term count matrices (`CountVectorizer`) fitted strictly on training data (`64% Train / 16% Validation / 20% Held-Out Test` split, `SEED = 25`).
4. **Step 4 — Tokenization, `host_name` Normalization, & `reviewer_name` Anonymization:** Drop `reviewer_name` immediately for privacy; replace each property's `host_name` in review text with a unified `host_name_token` and track `mentions_host_name`.
5. **Step 5 — Flagging Sentiment-Flipping Negation Tokens:** Keep raw unstripped text for `TextBlob` polarity scoring so negations and punctuation work naturally, while binding negation words (`not`, `no`, `never`, `n't`) to content lemmas (`not_work`, `not_recommend`, `not_clean` vs. double-negative litotes `not_disappoint`, `not_hesitate`).
6. **Step 6 — Domain-Aware Stopword Filtering:** Remove standard English and Airbnb-boilerplate stopwords (`airbnb`, `toronto`, `stay`, `place`) and compare top 20 token frequencies before vs. after filtering.
7. **Step 7 — POS-Aware WordNet Lemmatization & Nickname Resolution:** Lemmatize verbs and nouns (`WordNetLemmatizer`) and map 35+ formal/nickname pairs (*Alexandra ↔ Alex*, *Michael ↔ Mike*, *Matthew ↔ Matt*, *Daniel ↔ Dan*) to `host_name_token`.
8. **Step 8 — Explicit TF, IDF, TF–IDF, and Truncated SVD (`TruncatedSVD`):** Compute explicit $\text{TF}$, $\text{DF}$, $\text{IDF}$, and $\text{TF–IDF}$ tables comparing core praise terms against high-IDF complaint terms (`dirty`, `smell`, `stain`, `not_work`, `not_clean`), and compress the TF–IDF matrix into 10 SVD components.
9. **Step 9 — LDA Topic Modeling with UMass Coherence:** Fit `LatentDirichletAllocation` on training word counts across $K \in \{2, 3, 4\}$, evaluate **UMass Coherence**, select $K = 3$, and validate topic interpretations against high-weight guest reviews:
   - **Topic 1:** *Space, Comfort & Host Hospitality* (`space`, `comfortable`, `helpful`, `feel`, `responsive`, `quiet`, `kitchen`)
   - **Topic 2:** *Check-In, Bath/Bed & Unit Operations* (`bed`, `check`, `bathroom`, `night`, `book`, `door`, `issue`, `water`, `floor`)
   - **Topic 3:** *Walkability, Transit & Downtown Location* (`walk`, `close`, `easy`, `park`, `restaurant`, `area`, `downtown`, `station`)
10. **Step 10 — Cross-Analysis of TF–IDF, LDA Topics, Sentiment/Emotions, Longitudinal Trajectories, & Rating Prediction:**
    - **Finding #1 (Themes & Dissatisfaction):** `Topic 2 (Check-In, Bath/Bed & Unit Operations)` exhibits a **31.6% dissatisfaction rate** — roughly **3x higher** than `Topic 1` (**11.5%**) and `Topic 3` (**10.0%**), accompanied by longer reviews (`62.4` words) and elevated `disgust_per100` and `anger_per100`.
    - **Finding #2 (Property Reviews Changing Over Time):** Macro complaint rates in Toronto rose from **4.99% (2017)** to **6.59% (2026)**. Micro property-level tracking (`Early Half` vs. `Recent Half` of each listing's review career) exposes deteriorating properties whose recent dissatisfaction surged by **+30 to +40 percentage points** while still displaying respectable cumulative ratings ($3.94–4.63$), alongside turnaround properties that eliminated early complaints.
    - **Finding #3 (The "Michelin Star" Host Interaction Question):** Reviews that mention the host by name (`host_name_token`, **32.5%** of reviews) have **less than half the dissatisfaction rate** (**9.8% vs. 21.0%**), higher `trust_per100` (**5.68 vs. 4.70**), and **+0.102 higher property ratings**. Across all `17,763` rated properties ($R^2 = 0.819$), physical **Accuracy ($\beta = 0.312$)** and **Cleanliness ($\beta = 0.258$)** establish the baseline floor, while **Host Communication ($\beta = 0.207$)** acts as the hospitality multiplier.
    - **Finding #4 (Held-Out Test Rating Prediction):** Across the 5 benchmarked models (`Mean baseline`, `Sentiment`, `Sentiment + emotions + length`, `Sentiment + emotions + length + host/neg + topics`, and `Full Hybrid with TF-IDF SVD`), every feature group lowers Validation `MAE`. The **Full Hybrid model** wins Validation (`MAE = 0.2389`, $R^2 = 0.1328$) and beats the baseline on the untouched **Held-Out Test Set (`Test MAE = 0.2443` vs. `0.2588`, `Test R² = 0.1190`)**.

---

## 5. Repository Structure

```text
Text-Mining-Analysis_Toronto-AirBnB-Reviews/
├── OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb   # Primary Team Project #2 Notebook (fully executed)
├── OPIM-5671 Team Project #2 Guideline.pdf                  # Official course project guidelines & rubric
├── OPIM 5671 - Team Project 2.docx.pdf                      # Team Seven project proposal & 10-step plan
├── Eastwood_James_OPIM_5671_Assignment_02_Fall_2026.ipynb   # Reference Assignment #2 workflow
├── 04_Quantifying_Texts_2026_10_27.ipynb                    # Course reference: Parsing, TF-IDF, and SVD
├── 05_Text_Features_and_Rating_Prediction_2026_10_27.ipynb  # Course reference: Sentiment, NRCLex, LDA, & Prediction
├── 06_RNN_and_Text_Mining_2026_11_03.ipynb                  # Course reference: Neural text classification
└── README.md                                                # Project documentation & findings summary
```

---

## 6. How to Run

1. **Google Colab (Recommended):** Upload [`OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb`](OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb) to Google Colab using the default CPU runtime and select **Runtime $\rightarrow$ Run all**. All dependencies and the Kaggle dataset (`vladimirkondratev/inside-airbnb-toronto-sep-2026`) download automatically in ~25–35 seconds.
2. **Local Python Environment:**
   ```bash
   pip install kagglehub nltk textblob==0.19.0 nrclex==4.1.0 scikit-learn pandas numpy matplotlib jupyter
   jupyter notebook OPIM_5671_Team_Project_02_Toronto_Airbnb_Reviews.ipynb
   ```
