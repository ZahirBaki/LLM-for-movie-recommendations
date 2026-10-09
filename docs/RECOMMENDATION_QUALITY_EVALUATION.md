# Recommendation quality evaluation

Offline measurements of the current production engine. All experiments are isolated and are not deployed.

## Current ranking architecture

The API uses RecommendationEngine, ContentRecommender and persisted SVDRecommender. `scripts/legacy/content_demo.py` and `scripts/legacy/collaborative_demo.py` are historical standalone prototypes, not the active API path.

Candidates come from the 62,398-row catalog. Only overview text is embedded by normalized all-MiniLM-L6-v2 (384 dimensions). Story similarity is 70% mean cosine plus 30% strongest cosine, normalized by the maximum non-seed candidate. Separate story/genre/director/actor base weights are 0.40/0.25/0.20/0.15, renormalized across active categories, with a multi-match bonus up to 0.10. Inferred actor/director matches have strength 0.20. Inferred genres are a union; additional candidate genres receive no penalty.

At least two SVD-covered local favorites enable collaborative-primary ranking. SVD averages favorite latent vectors, scores by dot product and request min-max normalization, and retrieves up to 500 candidates. Explicit soft preferences add 0.10 times content score. Strong SVD-unavailable candidates can compete near the tenth-result boundary by content rank; remaining slots use content fallback. External/mixed requests use the existing content-only path with all seed vectors and metadata, and no invented collaborative scores.

Required genres/actors/directors demand ALL requested values; excluded fields reject ANY; year bounds are inclusive. Favorites are excluded by ID. This tool applies explicit excluded/seen/shown ID unions with `--exclude`; chat separately retains its bounded session filtering. Popularity, TMDB ratings and TMDB votes do not contribute to content ranking. MovieLens interaction counts influence SVD indirectly. Content-only `engine_final_score` is null: its actual sorting key is `content_score`, displayed separately as ranking score.

## Protocol, provenance and limits

A-F seed identities are verified against local catalog ID/title/year records. G-H use the verified public TMDB snapshot in `evaluation/fixtures/external_movies.json` (source URL and retrieval time included). Full numeric diagnostics, per-seed similarities, score contributions, filters, experimental constants and artifact SHA-256 hashes are in `evaluation/results/recommendation_quality.json`. Only the cached existing model is loaded; no large model is trained/downloaded, and no OpenAI/provider calls are made.

**Objective diagnostics:** genre Jaccard, any shared seed genre at top 5/10, scenario anchor-tag containment, popularity/vote distributions, genre/embedding diversity, duplication and exclusions. Anchor tags describe the scenario's category, not ground-truth relevance. Suite catalog coverage measures these eight scenarios, not user-wide exposure. Cross-scenario repeats can be appropriate.

**Offline relevance metrics:** trustworthy Recall@10/NDCG@10 are unavailable for these saved factors. `scripts/data/build_svd_model.py` calls `fit()` on the full ratings input; no excluded hold-out split provenance accompanies the factors. Historical evaluation scripts train separate models for their splits; their CSV metrics cannot be attributed to the current engine. No retraining or invented labels was performed. Tested metric helpers are provided for future leakage-free data.

**Subjective quality:** no human judgments were collected. Genre overlap, popularity, votes and embedding similarity cannot prove recommendation relevance. Strongest contributing seed below means highest overview cosine only, not causal attribution of a whole hybrid score.

## Baseline scenarios

### A. Serious science fiction

Seeds: Interstellar (2014), TMDB 157336, Arrival (2016), TMDB 329865

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | The Martian (2015), 286217 | 1.0000 | 1.0000 | 0.3512 | 1.0000 | 0.1805/0.2133 | 0.7500 | 29.7725 | 7.7080 | 21868.0000 | Interstellar | svd |
| 2 | Inception (2010), 27205 | 0.9832 | 0.9832 | 0.3454 | 0.9832 | 0.2146/0.2235 | 0.4000 | 51.9287 | 8.3730 | 40292.0000 | Arrival | svd |
| 3 | Ex Machina (2015), 264660 | 0.8367 | 0.8367 | 0.2885 | 0.8367 | 0.2340/0.2590 | 0.5000 | 17.5895 | 7.5700 | 14623.0000 | Interstellar | svd |
| 4 | The Imitation Game (2014), 205596 | 0.8065 | 0.8065 | 0.2959 | 0.8065 | 0.2938/0.2995 | 0.1429 | 48.2568 | 7.9790 | 18458.0000 | Arrival | svd |
| 5 | The Matrix (1999), 603 | 0.7183 | 0.7183 | 0.2798 | 0.7183 | 0.2545/0.3122 | 0.2000 | 51.0605 | 8.2600 | 28833.0000 | Interstellar | svd |
| 6 | The Wolf of Wall Street (2013), 106646 | 0.7006 | 0.7006 | 0.1116 | 0.7006 | 0.0193/0.0254 | 0.1667 | 33.7935 | 8.0270 | 26613.0000 | Interstellar | svd |
| 7 | Edge of Tomorrow (2014), 137113 | 0.6696 | 0.6696 | 0.1618 | 0.6696 | 0.1427/0.1561 | 0.2000 | 22.0547 | 7.6430 | 15548.0000 | Arrival | svd |
| 8 | Django Unchained (2012), 68718 | 0.6537 | 0.6537 | 0.1669 | 0.6537 | 0.1476/0.1696 | 0.2000 | 39.1739 | 8.1930 | 28670.0000 | Interstellar | svd |
| 9 | Whiplash (2014), 244786 | 0.6531 | 0.6531 | 0.1456 | 0.6531 | 0.1126/0.1464 | 0.1667 | 36.8739 | 8.3760 | 17148.0000 | Arrival | svd |
| 10 | Mad Max: Fury Road (2015), 76341 | 0.6427 | 0.6427 | 0.2873 | 0.6427 | 0.2179/0.2905 | 0.4000 | 45.1686 | 7.6390 | 24680.0000 | Interstellar | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 0.4000/0.2000; mean Jaccard 0.3126; genre diversity 0.7485; embedding diversity 0.8034.
Popularity distribution: {"min": 17.5895, "p25": 30.77775, "median": 38.0239, "p75": 47.48475, "max": 51.9287}; votes: {"min": 14623.0, "p25": 17475.5, "median": 23274.0, "p75": 28155.75, "max": 40292.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

Diagnostic probes (not preferred recommendations or relevance labels):

| Movie | Baseline pool rank | Raw story | Weighted story | Weighted genre | Weighted actor | Weighted director | Bonus | Content | Votes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Space Chimps | null | 0.4820 | 0.3263 | 0.1250 | 0.0000 | 0.0000 | 0.0333 | 0.4846 | 500.0000 |

### B. Animated family

Seeds: Coco (2017), TMDB 354912, Inside Out (2015), TMDB 150540, Zootopia (2016), TMDB 269149

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Up (2009), 14160 | 1.0000 | 1.0000 | 0.4327 | 1.0000 | 0.2165/0.2790 | 0.6667 | 30.0044 | 7.9600 | 22187.0000 | Coco | svd |
| 2 | WALL·E (2008), 10681 | 0.9776 | 0.9776 | 0.2352 | 0.9776 | 0.1752/0.2092 | 0.2857 | 37.6042 | 8.1250 | 20931.0000 | Inside Out | svd |
| 3 | Ratatouille (2007), 2062 | 0.9329 | 0.9329 | 0.3777 | 0.9329 | 0.2507/0.3076 | 0.4286 | 30.4734 | 7.8480 | 18992.0000 | Coco | svd |
| 4 | Big Hero 6 (2014), 177572 | 0.8950 | 0.8950 | 0.2914 | 0.8950 | 0.1243/0.2175 | 0.5714 | 34.1716 | 7.7250 | 16944.0000 | Coco | svd |
| 5 | Monsters, Inc. (2001), 585 | 0.8264 | 0.8264 | 0.3428 | 0.8264 | 0.1555/0.2248 | 0.4286 | 36.8966 | 7.8570 | 20308.0000 | Zootopia | svd |
| 6 | The Incredibles (2004), 9806 | 0.7862 | 0.7862 | 0.2976 | 0.7862 | 0.1781/0.2865 | 0.4286 | 33.4487 | 7.7290 | 19346.0000 | Zootopia | svd |
| 7 | How to Train Your Dragon (2010), 10191 | 0.7791 | 0.7791 | 0.2419 | 0.7791 | 0.1173/0.2016 | 0.4286 | 26.7812 | 7.8680 | 14761.0000 | Zootopia | svd |
| 8 | Finding Nemo (2003), 12 | 0.7575 | 0.7575 | 0.2509 | 0.7575 | 0.1440/0.1761 | 0.5000 | 37.4962 | 7.8220 | 21030.0000 | Coco | svd |
| 9 | The Martian (2015), 286217 | 0.7417 | 0.7417 | 0.2241 | 0.7417 | 0.1645/0.1889 | 0.2857 | 29.7725 | 7.7080 | 21868.0000 | Coco | svd |
| 10 | Moana (2016), 277834 | 0.7342 | 0.7342 | 0.2713 | 0.7342 | 0.1204/0.1448 | 0.6667 | 7.5028 | 7.5660 | 14054.0000 | Inside Out | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 1.0000/0.9000; mean Jaccard 0.4690; genre diversity 0.5109; embedding diversity 0.8319.
Popularity distribution: {"min": 7.5028, "p25": 29.830475, "median": 31.96105, "p75": 36.21535, "max": 37.6042}; votes: {"min": 14054.0, "p25": 17456.0, "median": 19827.0, "p75": 21005.25, "max": 22187.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

### C. Crime drama

Seeds: The Godfather (1972), TMDB 238, GoodFellas (1990), TMDB 769

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | The Godfather Part II (1974), 240 | 1.0000 | 1.0000 | 0.7975 | 1.0000 | 0.6472/0.8038 | 1.0000 | 35.5734 | 8.5730 | 14387.0000 | The Godfather | svd |
| 2 | One Flew Over the Cuckoo's Nest (1975), 510 | 0.2679 | 0.2679 | 0.3381 | 0.2679 | 0.3024/0.3340 | 0.5000 | 20.4401 | 8.4040 | 11800.0000 | GoodFellas | svd |
| 3 | The Godfather Part III (1990), 242 | 0.2217 | 0.2217 | 0.7095 | 0.2217 | 0.5014/0.6521 | 0.6667 | 42.0410 | 7.4000 | 7083.0000 | The Godfather | svd |
| 4 | Pulp Fiction (1994), 680 | 0.2169 | 0.2169 | 0.3550 | 0.2169 | 0.3394/0.3455 | 0.2500 | 45.3711 | 8.4800 | 30967.0000 | The Godfather | svd |
| 5 | The Usual Suspects (1995), 629 | 0.2033 | 0.2033 | 0.3755 | 0.2033 | 0.2049/0.2479 | 0.6667 | 20.7677 | 8.1630 | 11787.0000 | The Godfather | svd |
| 6 | Scarface (1983), 111 | 0.1986 | 0.1986 | 0.4988 | 0.1986 | 0.3107/0.3197 | 0.6667 | 26.3989 | 8.1530 | 13481.0000 | The Godfather | svd |
| 7 | City of God (2002), 598 | 0.1881 | 0.1881 | 0.3937 | 0.1881 | 0.2336/0.2862 | 1.0000 | 20.4369 | 8.4000 | 8477.0000 | The Godfather | svd |
| 8 | Taxi Driver (1976), 103 | 0.1758 | 0.1758 | 0.4704 | 0.1758 | 0.1825/0.2235 | 1.0000 | 24.4765 | 8.1210 | 13829.0000 | GoodFellas | svd |
| 9 | The Departed (2006), 1422 | 0.1696 | 0.1696 | 0.5688 | 0.1696 | 0.3629/0.3805 | 0.6667 | 28.6602 | 8.1580 | 16706.0000 | GoodFellas | svd |
| 10 | Fight Club (1999), 550 | 0.1693 | 0.1693 | 0.2539 | 0.1693 | 0.2128/0.2491 | 0.3333 | 47.1043 | 8.4370 | 32940.0000 | GoodFellas | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 0.6000/0.7000; mean Jaccard 0.6750; genre diversity 0.4493; embedding diversity 0.7037.
Popularity distribution: {"min": 20.4369, "p25": 21.6949, "median": 27.52955, "p75": 40.424099999999996, "max": 47.1043}; votes: {"min": 7083.0, "p25": 11790.25, "median": 13655.0, "p75": 16126.25, "max": 32940.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

### D. Psychological thriller

Seeds: Se7en (1995), TMDB 807, Shutter Island (2010), TMDB 11324

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | The Usual Suspects (1995), 629 | 1.0000 | 1.0000 | 0.5110 | 1.0000 | 0.3534/0.3787 | 0.7500 | 20.7677 | 8.1630 | 11787.0000 | Shutter Island | svd |
| 2 | Memento (2000), 77 | 0.8612 | 0.8612 | 0.4242 | 0.8612 | 0.3223/0.3503 | 0.5000 | 17.7240 | 8.1730 | 16815.0000 | Shutter Island | svd |
| 3 | The Prestige (2006), 1124 | 0.8377 | 0.8377 | 0.4008 | 0.8377 | 0.2978/0.3105 | 0.4000 | 27.3135 | 8.2110 | 18188.0000 | Se7en | svd |
| 4 | The Silence of the Lambs (1991), 274 | 0.7739 | 0.7739 | 0.4006 | 0.7739 | 0.2197/0.2267 | 0.7500 | 7.6566 | 8.3460 | 18570.0000 | Shutter Island | svd |
| 5 | Twelve Monkeys (1995), 63 | 0.6488 | 0.6488 | 0.4408 | 0.6488 | 0.3057/0.3135 | 0.4000 | 19.3624 | 7.6140 | 9422.0000 | Shutter Island | svd |
| 6 | The Departed (2006), 1422 | 0.6283 | 0.6283 | 0.4764 | 0.6283 | 0.2019/0.2782 | 0.7500 | 28.6602 | 8.1580 | 16706.0000 | Se7en | svd |
| 7 | Reservoir Dogs (1992), 500 | 0.6222 | 0.6222 | 0.4233 | 0.6222 | 0.3264/0.3371 | 0.5000 | 24.4557 | 8.1000 | 15861.0000 | Shutter Island | svd |
| 8 | Inception (2010), 27205 | 0.6109 | 0.6109 | 0.1978 | 0.6109 | 0.2435/0.2457 | 0.0000 | 51.9287 | 8.3730 | 40292.0000 | Se7en | svd |
| 9 | Léon: The Professional (1994), 101 | 0.6028 | 0.6028 | 0.2646 | 0.6028 | 0.1719/0.1776 | 0.4000 | 28.7453 | 8.2890 | 16574.0000 | Shutter Island | svd |
| 10 | The Sixth Sense (1999), 745 | 0.5877 | 0.5877 | 0.4510 | 0.5877 | 0.2758/0.3107 | 0.7500 | 24.6879 | 7.9550 | 13366.0000 | Se7en | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/0.9000; anchor tags top5/top10 0.4000/0.3000; mean Jaccard 0.5200; genre diversity 0.6641; embedding diversity 0.7473.
Popularity distribution: {"min": 7.6566, "p25": 19.713725, "median": 24.5718, "p75": 28.323525, "max": 51.9287}; votes: {"min": 9422.0, "p25": 13989.75, "median": 16640.0, "p75": 17844.75, "max": 40292.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

### E. Action/superhero

Seeds: X-Men (2000), TMDB 36657, The Avengers (2012), TMDB 24428

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Spider-Man (2002), 557 | 1.0000 | 1.0000 | 0.3912 | 1.0000 | 0.2618/0.2818 | 0.6667 | 42.6778 | 7.3560 | 21493.0000 | X-Men | svd |
| 2 | Batman Begins (2005), 272 | 0.9642 | 0.9642 | 0.2203 | 0.9642 | 0.1843/0.2095 | 0.2000 | 31.1375 | 7.7300 | 23253.0000 | The Avengers | svd |
| 3 | X2 (2003), 36658 | 0.9035 | 0.9035 | 0.8005 | 0.9035 | 0.5145/0.6675 | 1.0000 | 0.9573 | 7.0460 | 11217.0000 | X-Men | svd |
| 4 | Iron Man (2008), 1726 | 0.8940 | 0.8940 | 0.5470 | 0.8940 | 0.2949/0.3734 | 1.0000 | 37.8849 | 7.7000 | 28880.0000 | The Avengers | svd |
| 5 | The Dark Knight (2008), 155 | 0.8547 | 0.8547 | 0.2169 | 0.8547 | 0.1850/0.1920 | 0.2000 | 60.2589 | 8.5360 | 36839.0000 | The Avengers | svd |
| 6 | X-Men: First Class (2011), 49538 | 0.8541 | 0.8541 | 0.5882 | 0.8541 | 0.3780/0.5414 | 1.0000 | 1.2958 | 7.3070 | 13899.0000 | X-Men | svd |
| 7 | The Dark Knight Rises (2012), 49026 | 0.8358 | 0.8358 | 0.1969 | 0.8358 | 0.1492/0.1823 | 0.1667 | 33.7107 | 7.7970 | 25027.0000 | The Avengers | svd |
| 8 | Spider-Man 2 (2004), 558 | 0.8010 | 0.8010 | 0.4204 | 0.8010 | 0.2228/0.2758 | 1.0000 | 13.8098 | 7.3330 | 16969.0000 | The Avengers | svd |
| 9 | Gladiator (2000), 98 | 0.7782 | 0.7782 | 0.2786 | 0.7782 | 0.1567/0.1569 | 0.5000 | 37.8146 | 8.2310 | 21570.0000 | The Avengers | svd |
| 10 | Iron Man 2 (2010), 10138 | 0.7662 | 0.7662 | 0.5417 | 0.7662 | 0.2971/0.3365 | 1.0000 | 50.6048 | 6.8550 | 22967.0000 | The Avengers | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 0.6000/0.6000; mean Jaccard 0.6733; genre diversity 0.4952; embedding diversity 0.7148.
Popularity distribution: {"min": 0.9573, "p25": 18.141725, "median": 35.76265, "p75": 41.479575, "max": 60.2589}; votes: {"min": 11217.0, "p25": 18100.0, "median": 22268.5, "p75": 24583.5, "max": 36839.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

### F. Romantic drama

Seeds: The Notebook (2004), TMDB 11036, La La Land (2016), TMDB 313369

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Love Actually (2003), 508 | 1.0000 | 1.0000 | 0.5746 | 1.0000 | 0.3467/0.4951 | 1.0000 | 16.1679 | 7.0980 | 7582.0000 | The Notebook | svd |
| 2 | The Devil Wears Prada (2006), 350 | 0.9645 | 0.9645 | 0.3232 | 0.9645 | 0.2053/0.2216 | 0.6667 | 32.0635 | 7.4340 | 14272.0000 | La La Land | svd |
| 3 | (500) Days of Summer (2009), 19913 | 0.9596 | 0.9596 | 0.4903 | 0.9596 | 0.2691/0.2987 | 1.0000 | 17.3242 | 7.3070 | 11330.0000 | The Notebook | svd |
| 4 | 10 Things I Hate About You (1999), 4951 | 0.9406 | 0.9406 | 0.4705 | 0.9406 | 0.2251/0.3129 | 1.0000 | 17.9354 | 7.5990 | 9170.0000 | La La Land | svd |
| 5 | Mean Girls (2004), 10625 | 0.9390 | 0.9390 | 0.4234 | 0.9390 | 0.2258/0.3171 | 0.6667 | 27.5754 | 7.2300 | 9697.0000 | La La Land | svd |
| 6 | The Perks of Being a Wallflower (2012), 84892 | 0.9060 | 0.9060 | 0.2538 | 0.9060 | 0.2230/0.2427 | 0.3333 | 15.2330 | 7.8050 | 11471.0000 | La La Land | svd |
| 7 | About Time (2013), 122906 | 0.8823 | 0.8823 | 0.4565 | 0.8823 | 0.2739/0.3531 | 0.5000 | 20.3601 | 7.9010 | 9533.0000 | The Notebook | svd |
| 8 | Braveheart (1995), 197 | 0.8817 | 0.8817 | 0.2511 | 0.8817 | 0.2031/0.2771 | 0.2000 | 22.3199 | 7.9440 | 11467.0000 | The Notebook | svd |
| 9 | Silver Linings Playbook (2012), 82693 | 0.8814 | 0.8814 | 0.5465 | 0.8814 | 0.3279/0.4132 | 1.0000 | 13.3952 | 7.1540 | 12867.0000 | The Notebook | svd |
| 10 | Eternal Sunshine of the Spotless Mind (2004), 38 | 0.8791 | 0.8791 | 0.4272 | 0.8791 | 0.3033/0.3095 | 0.5000 | 23.1351 | 8.1000 | 16780.0000 | La La Land | svd |

Mode: SVD — collaborative primary
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 0.6000/0.6000; mean Jaccard 0.6867; genre diversity 0.4919; embedding diversity 0.7872.
Popularity distribution: {"min": 13.3952, "p25": 16.456975, "median": 19.147750000000002, "p75": 22.9313, "max": 32.0635}; votes: {"min": 7582.0, "p25": 9574.0, "median": 11398.5, "p75": 12518.0, "max": 16780.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

### G. External seed

Seeds: Digger (2026), TMDB 1248832

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Return with Honor (2006), 36827 | 0.5841 | null | 0.5841 | null | 0.4731/0.4731 | 0.6667 | 0.8013 | 6.1000 | 72.0000 | Digger | content |
| 2 | The End of Man (1971), 51733 | 0.5826 | null | 0.5826 | null | 0.4706/0.4706 | 0.5000 | 1.7658 | 6.0000 | 21.0000 | Digger | content |
| 3 | God Tussi Great Ho (2008), 26153 | 0.5811 | null | 0.5811 | null | 0.4682/0.4682 | 0.6667 | 1.6018 | 4.1380 | 58.0000 | Digger | content |
| 4 | Intrepido: A Lonely Hero (2013), 209413 | 0.5769 | null | 0.5769 | null | 0.4617/0.4617 | 1.0000 | 0.8859 | 6.3000 | 83.0000 | Digger | content |
| 5 | The Hostage (2006), 418433 | 0.5751 | null | 0.5751 | null | 0.4588/0.4588 | 0.6667 | 1.2618 | 7.0000 | 28.0000 | Digger | content |
| 6 | Biosphere (2023), 865797 | 0.5673 | null | 0.5673 | null | 0.4466/0.4466 | 0.5000 | 2.3928 | 5.2000 | 62.0000 | Digger | content |
| 7 | Son of Morning (2011), 74877 | 0.5669 | null | 0.5669 | null | 0.4459/0.4459 | 1.0000 | 2.7629 | 3.0000 | 21.0000 | Digger | content |
| 8 | The Employment (2008), 45336 | 0.5650 | null | 0.5650 | null | 0.4430/0.4430 | 0.6667 | 1.5809 | 7.5120 | 126.0000 | Digger | content |
| 9 | Salvation Boulevard (2011), 51995 | 0.5624 | null | 0.5624 | null | 0.4389/0.4389 | 0.5000 | 2.5777 | 5.3000 | 86.0000 | Digger | content |
| 10 | Yavade Subramanyam (2015), 332831 | 0.5550 | null | 0.5550 | null | 0.4272/0.4272 | 0.6667 | 1.2810 | 7.3000 | 22.0000 | Digger | content |

Mode: CONTENT - external movie query
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 1.0000/1.0000; mean Jaccard 0.6833; genre diversity 0.4670; embedding diversity 0.6513.
Popularity distribution: {"min": 0.8013, "p25": 1.2666, "median": 1.5913499999999998, "p75": 2.2360499999999996, "max": 2.7629}; votes: {"min": 21.0, "p25": 23.5, "median": 60.0, "p75": 80.25, "max": 126.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

Diagnostic probes (not preferred recommendations or relevance labels):

| Movie | Baseline pool rank | Raw story | Weighted story | Weighted genre | Weighted actor | Weighted director | Bonus | Content | Votes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Return with Honor | 1.0000 | 0.4731 | 0.3008 | 0.2500 | 0.0000 | 0.0000 | 0.0333 | 0.5841 | 72.0000 |
| The End of Man | 2.0000 | 0.4706 | 0.2992 | 0.2500 | 0.0000 | 0.0000 | 0.0333 | 0.5826 | 21.0000 |
| God Tussi Great Ho | 3.0000 | 0.4682 | 0.2977 | 0.2500 | 0.0000 | 0.0000 | 0.0333 | 0.5811 | 58.0000 |

### H. Mixed local/external

Seeds: Interstellar (2014), TMDB 157336, Digger (2026), TMDB 1248832

| Rank | Title (year), ID | Ranking score | Engine final | Content | SVD | Mean/max cosine | Genre Jaccard | Popularity | Rating | Votes | Strongest overview seed | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Space Chimps (2008), 11802 | 0.5443 | null | 0.5443 | null | 0.4266/0.6104 | 0.5000 | 6.6643 | 4.9000 | 500.0000 | Interstellar | content |
| 2 | Journey to the Center of the Earth (2008), 88751 | 0.5090 | null | 0.5090 | null | 0.4097/0.4750 | 0.5000 | 11.8109 | 6.1010 | 5310.0000 | Interstellar | content |
| 3 | The Great Flood (2025), 982843 | 0.5032 | null | 0.5032 | null | 0.4174/0.4281 | 0.7500 | 8.8092 | 6.0000 | 782.0000 | Digger | content |
| 4 | The Hitch Hikers Guide to the Galaxy (1981), 64353 | 0.5004 | null | 0.5004 | null | 0.3982/0.4590 | 0.7500 | 2.3042 | 7.3000 | 160.0000 | Interstellar | content |
| 5 | Voyage to the Bottom of the Sea (1961), 2160 | 0.4982 | null | 0.4982 | null | 0.3964/0.4520 | 0.7500 | 3.7902 | 5.7000 | 147.0000 | Digger | content |
| 6 | The Wild Blue Yonder (2005), 8410 | 0.4964 | null | 0.4964 | null | 0.3935/0.4502 | 0.6000 | 2.1480 | 6.0000 | 93.0000 | Interstellar | content |
| 7 | Prometheus (2012), 70981 | 0.4958 | null | 0.4958 | null | 0.5583/0.6832 | 0.1667 | 16.3588 | 6.6110 | 13671.0000 | Interstellar | content |
| 8 | James vs. His Future Self (2019), 629502 | 0.4944 | null | 0.4944 | null | 0.4033/0.4171 | 0.7500 | 2.4752 | 5.8000 | 38.0000 | Interstellar | content |
| 9 | Left Behind: World at War (2005), 38828 | 0.4916 | null | 0.4916 | null | 0.4526/0.5985 | 0.2857 | 2.8379 | 4.2000 | 79.0000 | Digger | content |
| 10 | Biosphere (2023), 865797 | 0.4906 | null | 0.4906 | null | 0.3825/0.4466 | 0.6000 | 2.3928 | 5.2000 | 62.0000 | Digger | content |

Mode: CONTENT - external movie query
Shared genre top5/top10 1.0000/1.0000; anchor tags top5/top10 0.4000/0.5000; mean Jaccard 0.5652; genre diversity 0.6266; embedding diversity 0.6195.
Popularity distribution: {"min": 2.148, "p25": 2.4133999999999998, "median": 3.31405, "p75": 8.272975, "max": 16.3588}; votes: {"min": 38.0, "p25": 82.5, "median": 153.5, "p75": 711.5, "max": 13671.0}
Duplicate rate 0.0; seed exclusion True; explicit exclusions True.

Diagnostic probes (not preferred recommendations or relevance labels):

| Movie | Baseline pool rank | Raw story | Weighted story | Weighted genre | Weighted actor | Weighted director | Bonus | Content | Votes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Space Chimps | 1.0000 | 0.4817 | 0.3234 | 0.1875 | 0.0000 | 0.0000 | 0.0333 | 0.5443 | 500.0000 |

## Measured failure modes

**Interstellar-only control:** Space Chimps is content rank 1, with overview cosine 0.6104 and content score 0.5574. Its weighted story/genre contributions are 0.3574/0.1667, plus bonus 0.0333. This is an actual engine result, not an assumed causal explanation.

**Space Chimps and mixed favorites:** it is absent from A's top-200 engine pool but ranks 1 in H. H overview similarities are TMDB 157336: 0.6104, TMDB 1248832: 0.2428. Mean/max aggregation therefore preserves a strong partial match to Interstellar even with a weaker Digger match. The inferred union is Adventure/Drama/Science Fiction/Comedy: Space Chimps matches three of four, earning weighted genre 0.1875, with weighted story 0.3234 and bonus 0.0333. Additional Family/Animation tags are unpenalized. A uses collaborative-primary ranking; H forces content-only, so changing the seed also changes the ranking mode. These measurements do not isolate a pure seed effect.

**Obscure Digger matches:** Return with Honor, The End of Man and God Tussi Great Ho are the first three results in G. Each matches both Comedy and Drama and receives weighted genre 0.25 and bonus 0.0333. Their overview cosines are 0.4731, 0.4706, 0.4682; their weighted story contributions are 0.3008, 0.2992, 0.2977. Cast/director contributions are zero. Their votes are 72/21/58; top-ten median votes are 60.0000. G anchor containment is 1.0000, so a simple Comedy/Drama mismatch does not explain the failure. Vote confidence, popularity, tone and audience are absent from this content ordering.

**Metadata quality:** {"overview": 0, "genres": 0, "actors": 733, "director": 117} catalog rows have blank fields (counts per field). 27375 rows have fewer than 50 votes. Missing text and credits can weaken features, but these global counts do not prove the reported failures are caused by missing metadata. The catalog has no explicit tone or age-certification field. External Digger uses its fixed TMDB synopsis and verified metadata; model compatibility is checked against three stored overview vectors.

**Embedding scope:** only overview text is embedded; genres and credits are separate features. High overview cosine demonstrates representation similarity, not which keywords caused it or that the audience/tone matches. Establishing keyword causality needs controlled text ablations; this milestone does not claim it. The scoring breakdown supports a combined explanation involving overview similarity, broad union-genre rewards and missing audience/reliability signals.

Across 80 baseline slots, any shared seed genre is present in every result; scenario anchor containment averages 0.6000, while mean genre Jaccard is 0.5732. There are 75 unique titles, no within-list duplicates and no returned seeds. Shared-genre checks alone are too permissive to detect the reported poor examples.

**Most promising next hypothesis:** evaluate the small combined genre/vote reranker specifically for content-only requests with human judgments. In this suite its mean Jaccard changes 0.5732 to 0.7465, anchor containment 0.6000 to 0.7375, and genre diversity 0.5567 to 0.3025. It addresses both broad genre rewards and low-confidence results, but the diversity loss is material. Genre-only reranking improves anchor containment more in the aggregate; vote-only/popularity-only signals do not reliably improve tag alignment. No variant is a proven relevance winner.

## Isolated experimental comparisons

Genre/popularity/vote rerankers use the baseline engine top-200 pool and declared constants in JSON. Audience filtering uses Family/Animation tags, a coarse proxy; no tone or age-certification field is present. Mean/strongest-story ablations use the full filtered catalog, change normalized story aggregation and hold non-story contributions and the multi-match bonus fixed. They omit SVD; their comparison against collaborative scenarios does not isolate aggregation alone.

| Method | Unique titles | Coverage | Cross-scenario repeats | Mean genre Jaccard | Anchor containment@10 | Genre diversity | Mean list median votes |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 75 | 0.0012 | 0.0625 | 0.5732 | 0.6000 | 0.5567 | 13409.5625 |
| genre_aware | 76 | 0.0012 | 0.0500 | 0.7425 | 0.7625 | 0.3307 | 12480.5625 |
| popularity_confidence | 75 | 0.0012 | 0.0625 | 0.5585 | 0.5875 | 0.5649 | 13360.7500 |
| vote_reliability | 75 | 0.0012 | 0.0625 | 0.5625 | 0.5375 | 0.5465 | 14127.4375 |
| genre_votes | 75 | 0.0012 | 0.0625 | 0.7465 | 0.7375 | 0.3025 | 12790.0000 |
| audience_proxy_filter | 74 | 0.0012 | 0.0750 | 0.5836 | 0.6125 | 0.5471 | 13399.5000 |
| mean_story | 79 | 0.0013 | 0.0125 | 0.7654 | 0.8625 | 0.3435 | 2075.5000 |
| strongest_story | 78 | 0.0013 | 0.0250 | 0.7230 | 0.7875 | 0.3705 | 1899.1875 |

Per-scenario metric deltas (relative to baseline):

| Scenario | Variant | Genre Jaccard delta | Anchor@10 delta | Genre diversity delta |
|---|---|---:|---:|---:|
| A | genre_aware | 0.1474 | 0.3000 | -0.1115 |
| A | popularity_confidence | 0.0000 | 0.0000 | 0.0000 |
| A | vote_reliability | 0.0000 | 0.0000 | 0.0000 |
| A | genre_votes | 0.1574 | 0.3000 | -0.1363 |
| A | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| A | mean_story | 0.2814 | 0.5000 | -0.1587 |
| A | strongest_story | 0.2045 | 0.1000 | -0.1172 |
| B | genre_aware | 0.0595 | 0.1000 | -0.1590 |
| B | popularity_confidence | -0.0167 | 0.0000 | 0.0193 |
| B | vote_reliability | 0.0000 | 0.0000 | 0.0000 |
| B | genre_votes | 0.0595 | 0.1000 | -0.1590 |
| B | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| B | mean_story | 0.1786 | -0.2000 | -0.0592 |
| B | strongest_story | 0.0875 | -0.1000 | 0.0567 |
| C | genre_aware | 0.1417 | 0.2000 | -0.1530 |
| C | popularity_confidence | 0.0000 | 0.0000 | 0.0000 |
| C | vote_reliability | 0.0000 | 0.0000 | -0.0000 |
| C | genre_votes | 0.1417 | 0.2000 | -0.1530 |
| C | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| C | mean_story | 0.0983 | 0.3000 | -0.1248 |
| C | strongest_story | -0.0017 | 0.3000 | -0.0426 |
| D | genre_aware | 0.1300 | 0.1000 | -0.1752 |
| D | popularity_confidence | -0.0350 | 0.0000 | 0.0289 |
| D | vote_reliability | -0.0350 | -0.1000 | 0.0067 |
| D | genre_votes | 0.1300 | 0.1000 | -0.1752 |
| D | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| D | mean_story | 0.3683 | 0.7000 | -0.4597 |
| D | strongest_story | 0.3283 | 0.7000 | -0.4100 |
| E | genre_aware | 0.3267 | 0.4000 | -0.4952 |
| E | popularity_confidence | -0.0000 | 0.0000 | 0.0000 |
| E | vote_reliability | 0.0000 | 0.0000 | 0.0000 |
| E | genre_votes | 0.3267 | 0.4000 | -0.4952 |
| E | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| E | mean_story | 0.2767 | 0.4000 | -0.3974 |
| E | strongest_story | 0.3017 | 0.4000 | -0.4452 |
| F | genre_aware | 0.2133 | 0.2000 | -0.3067 |
| F | popularity_confidence | -0.0333 | 0.0000 | 0.0248 |
| F | vote_reliability | 0.0000 | 0.0000 | 0.0000 |
| F | genre_votes | 0.2133 | 0.2000 | -0.3067 |
| F | audience_proxy_filter | 0.0000 | 0.0000 | 0.0000 |
| F | mean_story | 0.2883 | 0.4000 | -0.4419 |
| F | strongest_story | 0.3133 | 0.4000 | -0.4919 |
| G | genre_aware | 0.2167 | 0.0000 | -0.2781 |
| G | popularity_confidence | 0.0000 | 0.0000 | -0.0189 |
| G | vote_reliability | 0.0333 | 0.0000 | -0.0363 |
| G | genre_votes | 0.2833 | 0.0000 | -0.4004 |
| G | audience_proxy_filter | 0.0333 | 0.0000 | -0.0363 |
| G | mean_story | 0.0000 | 0.0000 | 0.0000 |
| G | strongest_story | 0.0000 | 0.0000 | 0.0000 |
| H | genre_aware | 0.1198 | 0.0000 | -0.1293 |
| H | popularity_confidence | -0.0319 | -0.1000 | 0.0116 |
| H | vote_reliability | -0.0836 | -0.4000 | -0.0521 |
| H | genre_votes | 0.0748 | -0.2000 | -0.2081 |
| H | audience_proxy_filter | 0.0500 | 0.1000 | -0.0401 |
| H | mean_story | 0.0464 | 0.0000 | -0.0639 |
| H | strongest_story | -0.0350 | -0.3000 | -0.0397 |

### Experimental top five: External seed

| Method | Top five | Genre Jaccard | Anchor@10 | Median votes |
|---|---|---:|---:|---:|
| baseline | Return with Honor (2006); The End of Man (1971); God Tussi Great Ho (2008); Intrepido: A Lonely Hero (2013); The Hostage (2006) | 0.6833 | 1.0000 | 60.0000 |
| genre_aware | Intrepido: A Lonely Hero (2013); Son of Morning (2011); Dasvidaniya (2008); All Hail (2022); World of Glory (1991) | 0.9000 | 1.0000 | 65.0000 |
| popularity_confidence | The End of Man (1971); God Tussi Great Ho (2008); Son of Morning (2011); Biosphere (2023); The Book of Clarence (2024) | 0.6833 | 1.0000 | 67.0000 |
| vote_reliability | Super (2010); The Big Short (2015); We Have a Pope (2011); Return with Honor (2006); Intrepido: A Lonely Hero (2013) | 0.7167 | 1.0000 | 84.5000 |
| genre_votes | The Big Short (2015); We Have a Pope (2011); Intrepido: A Lonely Hero (2013); Son of Morning (2011); All Hail (2022) | 0.9667 | 1.0000 | 111.0000 |
| audience_proxy_filter | Return with Honor (2006); The End of Man (1971); God Tussi Great Ho (2008); Intrepido: A Lonely Hero (2013); The Hostage (2006) | 0.7167 | 1.0000 | 47.0000 |
| mean_story | Return with Honor (2006); The End of Man (1971); God Tussi Great Ho (2008); Intrepido: A Lonely Hero (2013); The Hostage (2006) | 0.6833 | 1.0000 | 60.0000 |
| strongest_story | Return with Honor (2006); The End of Man (1971); God Tussi Great Ho (2008); Intrepido: A Lonely Hero (2013); The Hostage (2006) | 0.6833 | 1.0000 | 60.0000 |

### Experimental top five: Mixed local/external

| Method | Top five | Genre Jaccard | Anchor@10 | Median votes |
|---|---|---:|---:|---:|
| baseline | Space Chimps (2008); Journey to the Center of the Earth (2008); The Great Flood (2025); The Hitch Hikers Guide to the Galaxy (1981); Voyage to the Bottom of the Sea (1961) | 0.5652 | 0.5000 | 153.5000 |
| genre_aware | The Great Flood (2025); The Hitch Hikers Guide to the Galaxy (1981); Voyage to the Bottom of the Sea (1961); James vs. His Future Self (2019); Yavade Subramanyam (2015) | 0.6850 | 0.5000 | 153.5000 |
| popularity_confidence | Journey to the Center of the Earth (2008); Space Chimps (2008); Prometheus (2012); The Great Flood (2025); Voyage to the Bottom of the Sea (1961) | 0.5333 | 0.4000 | 641.0000 |
| vote_reliability | Journey to the Center of the Earth (2008); Prometheus (2012); Pacific Rim (2013); Tenet (2020); The Great Flood (2025) | 0.4817 | 0.1000 | 5751.5000 |
| genre_votes | The Great Flood (2025); Journey to the Center of the Earth (2008); The Adventures of Buckaroo Banzai Across the 8th Dimension (1984); Voyage to the Bottom of the Sea (1961); The Hitch Hikers Guide to the Galaxy (1981) | 0.6400 | 0.3000 | 2583.0000 |
| audience_proxy_filter | The Great Flood (2025); The Hitch Hikers Guide to the Galaxy (1981); Voyage to the Bottom of the Sea (1961); The Wild Blue Yonder (2005); Prometheus (2012) | 0.6152 | 0.6000 | 86.0000 |
| mean_story | Space Chimps (2008); The Great Flood (2025); Journey to the Center of the Earth (2008); James vs. His Future Self (2019); The Hitch Hikers Guide to the Galaxy (1981) | 0.6117 | 0.5000 | 181.0000 |
| strongest_story | Space Chimps (2008); Left Behind: World at War (2005); Journey to the Center of the Earth (2008); Prometheus (2012); The Hitch Hikers Guide to the Galaxy (1981) | 0.5302 | 0.2000 | 330.0000 |

## Trade-offs and next milestone

Genre-aware reranking strengthens tag alignment but can reduce diversity and cannot distinguish serious from light comedy. Popularity/vote confidence favors mainstream and older titles and may disadvantage new releases. Family/Animation filtering can wrongly discard relevant animated films. Mixed-seed union genres allow a partial match to multiple favorites without matching their tone. These are hypotheses/proxy trade-offs, not proof of enjoyment. No winner is declared from eight examples.

Next milestone: collect blinded pairwise judgments across these and additional diverse seed sets; test a small opt-in content-only genre/reliability reranker against baseline. Establish a leakage-free interaction evaluation artifact before production changes. Preserve discovery of well-matched low-vote films. Tone/audience features require a verified source and labels before evaluation.

## Reproduction

```powershell
.venv\Scripts\python.exe -m evaluation.scripts.recommendation_quality evaluate --output logs/quality-evaluation.json --report logs/quality-evaluation.md
.venv\Scripts\python.exe -m evaluation.scripts.recommendation_quality diagnose --favorites 157336 1248832 --top-n 10 --exclude 11802
```

Use `--filters-json` with the existing genre/actor/director/year filter fields. External metadata comes only from the snapshot; the local model must already be cached. No production code, weights, catalog, embeddings or SVD artifacts are modified. Hashes are checked before/after the run. Provider availability is not queried or used as a quality metric.

## Validation

- Full offline backend suite: 287 tests passed, including 20 new diagnostic tests; zero failures/errors. TMDB/OpenAI/encoder fixtures are mocked in tests.
- TypeScript: npx.cmd tsc --noEmit passed.
- Existing tests/browser/check_movie_chat_browser.py passed on desktop/mobile with real engine rankings and mocked external services.
- Existing tests/browser/check_movie_chat_presentation_browser.py passed: short/long answers, five/nine cards, consecutive messages, manual scrolling, delayed posters and responsive layouts.
- All eight real-catalog scenarios and the Interstellar-only control completed; seven isolated variants per scenario. Real mixed-seed CLI smoke returned five valid results and honored explicit exclusions.
- Production source/catalog/embedding/SVD SHA-256 hashes match before/after evaluation and after regressions. No production scoring or frontend changes were made.
- No paid OpenAI requests, model downloads, embedding rebuilds or SVD training were performed.

## Files created

`evaluation/scripts/recommendation_quality.py` (offline CLI, metrics, diagnostics, isolated reranking); `evaluation/scripts/recommendation_quality_report.py` (report renderer); `tests/backend/test_recommendation_quality.py` (offline unit coverage); `evaluation/fixtures/external_movies.json` (verified public metadata snapshot); `evaluation/results/recommendation_quality.json` (measured output); `RECOMMENDATION_QUALITY_EVALUATION.md` (this report). Regression logs are local verification artifacts. Existing production/frontend files were not edited in this milestone.
