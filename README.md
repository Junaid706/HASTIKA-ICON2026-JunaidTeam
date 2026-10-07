HASTIKA @ ICON-2026 — JunaidTeam

Hate Speech and Target Category Identification in Kannada-English Code-Mixed Text

Team submission for the HASTIKA shared task at ICON-2026.

👥 Team
Junaid Abedin Rafi
Rifah Tasnim Jui
Tasfia Tabassum Toma
🎯 Tasks
Task A — Binary Hate Speech Detection: classify each comment as Hate or Non-Hate.
Task B — Fine-Grained Classification: classify each hate comment into one of six categories — Gender, Political, Religion, Geo-political, Violence, Others.
🛠️ Approach
Preprocessing: HTML entity/URL cleanup, whitespace normalization. No aggressive spelling normalization, since spelling variation in transliterated Kanglish text is informative signal.
Features: TF-IDF combining word-level (1–2 gram) and character-level (2–5 gram, word-boundary aware) n-grams — character n-grams help generalize across inconsistent transliteration spellings.
Model: Logistic Regression (class_weight='balanced') to handle class imbalance, especially for Task B's minority categories (Geo-political, Violence).
Validation: 85/15 stratified split used to estimate Macro-F1, then the final model is retrained on 100% of the labeled data before predicting on the official test set.
📊 Results
Task	Macro-F1	Accuracy
Task A — Binary	0.80	0.80
Task B — Fine-Grained	0.60	0.71
📂 Files
File	Description
train_baseline.py	Training + evaluation pipeline for the Development phase
predict_test.py	Final prediction script run on the official test set
JunaidTeam_taskA.csv	Final Task A predictions (id,label)
JunaidTeam_taskB.csv	Final Task B predictions (id,label)
HASTIKA_JunaidTeam_Report.pdf	Detailed project report (dataset analysis, methodology, results, future work)
🚀 Future Work
Fine-tune multilingual transformers (MuRIL, IndicBERT, XLM-RoBERTa) for stronger contextual representations.
Address Task B class imbalance with Focal Loss or targeted oversampling.
Explore ensembling TF-IDF-based and transformer-based models.
📖 Citation

Kavatagi, S., Rachh, R. (2025). HASTIKA: hate speech and target identification in Kannada-English code-mixed text. Language Resources and Evaluation, 59(3), 2811–2856.

Shared task organized by Manipal Academy of Higher Education (MAHE). Official starting kit: shankarb14/Hastika-ICON2026
