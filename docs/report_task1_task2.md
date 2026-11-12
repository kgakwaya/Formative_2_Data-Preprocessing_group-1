# Report section: Task 1 (Data Merge) and Task 2 (Image Processing)

## Task 1: Data merge

**Data.** `customer_social_profiles.csv` has 155 rows: one per customer per social platform, with engagement score, purchase-interest score and review sentiment. `customer_transactions.csv` has 150 rows: one per purchase, with amount, date, product category (the prediction target) and rating.

**Exploration.** All four numeric variables are spread fairly evenly across their ranges and are nearly symmetric (|skew| ≤ 0.23). The 1.5 × IQR rule found no outliers, so no transform or capping was needed. The five product categories are balanced (16.7% to 23.3% each). The only missing values are 10 customer ratings (6.7%).

**Cleaning.**
- Dropped 5 exact duplicate social rows.
- Filled the 10 missing ratings with the median (3.0) and added a `rating_missing` flag. Filling by product category was rejected because it would leak the target into a feature.
- Parsed dates.
- Unified the two customer keys: `customer_id_new` ("A178") and `customer_id_legacy` (178) both became the integer `customer_id`.
- Kept 17 customer + platform pairs that appear more than once with different scores, and averaged them.

**Join logic.** Customers have several social rows *and* several transactions. Joining the raw tables would have turned 117 matching transactions into 213 rows, duplicating purchases. So the social table was first aggregated to one profile per customer: mean and max engagement and interest, mean sentiment, positive and negative share, platform count, and one-hot platform presence. The profiles were then inner-joined many-to-one onto the transactions. The inner join keeps only transactions whose customer has a social profile, because the model serves customers who visit our social media.

**Post-merge checks.** The result has 117 rows from 61 customers. Every check passed:
- one row per transaction
- no missing social features
- the row count equals the transactions with a profile
- every class share within 1.6 percentage points of the original

The 33 dropped transactions belonged to 14 customers with no social profile.

**Features.** There are 24: spend, rating and its missing flag, month, weekday and weekend, the customer's transaction count and average spend, this purchase relative to that average, engagement, interest, an engagement × interest interaction, sentiment, and platform presence. None is derived from the target.

Correlation analysis shows the redundant pairs: sentiment mean vs positive share (r = 0.91), engagement mean vs max (0.89), interest mean vs max (0.87). It also shows that transaction and social features are nearly independent (|r| ≤ 0.29), so each source adds information. Per-category feature means differ by at most about 0.5 standard deviations (Electronics buyers: higher interest, fewer and smaller purchases). The product model's accuracy should therefore be judged against the 24% majority-class baseline.

## Task 2: Image collection and processing

**Collection.** Each of the three members (Ketia, Twariq and Audric) supplied a neutral, a smiling and a surprised photo, named `<member>_<expression>.jpg` (for example `ketia_neutral.jpg`). The photos range from 607×1080 to 960×1280 pixels, from different phones, indoors and outdoors.

**Pipeline.** Each photo was resized to a longest side of 800 px. The YuNet CNN detector located the face (one face per photo, confidence 0.91–0.95) and five landmarks. The landmarks were used to align the face to a 112×112 crop with the eyes level. Cropping removes the background, which would otherwise let a model learn *where* a photo was taken rather than *who* is in it.

**Augmentation.** Six augmentations per photo, applied to the full photo before detection so the whole pipeline faces each condition:
- 15° rotation
- horizontal flip
- grayscale
- low light (gamma 1.8)
- Gaussian blur
- Gaussian noise (σ = 15)

Result: 63 images (21 per member), and a face was found in all 63.

**Features (`image_features.csv`, 63 × 170).**
- a 128-d SFace face embedding, L2-normalised
- an HSV colour histogram (3 × 8 bins)
- a uniform LBP texture histogram (10 bins)
- brightness and contrast

The file also carries metadata: image ID, member, expression, augmentation, path and detection score.

**Validation.**
- Same-member pairs have a mean cosine similarity of 0.703, and 97.5% of them exceed OpenCV's 0.363 match threshold.
- Different-member pairs have a mean of 0.149. Only 16 of 1,323 (1.2%) exceed the threshold, and 15 of those pair Ketia's and Audric's surprised photos. Both show the same exaggerated expression, which pulls two faces together. So the face check should pick the best-matching member first and apply the threshold after, rather than use a threshold alone.
- PCA (60% of the variance in two components) shows three separate member clusters.
- Every augmented image's nearest original photo is its own source photo (54/54), so always the correct member.
- Gaussian noise moved embeddings the most (mean similarity to the source 0.857); blur and low light moved them the least (0.958, 0.941).

**Hand-off.** The pipeline is saved as `src/image_features.py` so the command-line app processes login photos exactly like the training data.
