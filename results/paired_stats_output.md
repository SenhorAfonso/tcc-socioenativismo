# Terminal Output — `python analysis/paired_stats.py`

```text
=== WESAD: per-model descriptives (recomputed from per-subject values) ===
  metric    model  n  mean  sd_population  sd_sample  ci95_low  ci95_high   min   max
accuracy      GRU 15 0.618          0.116      0.121     0.551      0.684 0.418 0.845
accuracy LightGBM 15 0.613          0.165      0.170     0.518      0.707 0.327 0.841
accuracy  XGBoost 15 0.659          0.131      0.136     0.584      0.734 0.414 0.861
      f1      GRU 15 0.495          0.149      0.155     0.410      0.581 0.201 0.820
      f1 LightGBM 15 0.496          0.160      0.166     0.404      0.588 0.206 0.713
      f1  XGBoost 15 0.521          0.151      0.156     0.434      0.607 0.184 0.745

=== WESAD: paired comparisons ===
  metric         comparison  n_pairs  mean_diff  t_ci95_low  t_ci95_high  boot_ci95_low  boot_ci95_high  wilcoxon_p  cohen_dz  wilcoxon_p_holm
accuracy     GRU - LightGBM       15     0.0050     -0.0604       0.0703        -0.0487          0.0643      0.8469    0.0421           0.8469
accuracy      GRU - XGBoost       15    -0.0413     -0.1037       0.0210        -0.0966          0.0137      0.1876   -0.3673           0.5629
accuracy LightGBM - XGBoost       15    -0.0463     -0.1024       0.0097        -0.0985         -0.0006      0.2078   -0.4578           0.5629
      f1     GRU - LightGBM       15    -0.0009     -0.0701       0.0683        -0.0601          0.0622      0.5614   -0.0071           1.0000
      f1      GRU - XGBoost       15    -0.0253     -0.0952       0.0445        -0.0866          0.0360      0.5245   -0.2010           1.0000
      f1 LightGBM - XGBoost       15    -0.0244     -0.0658       0.0169        -0.0616          0.0125      0.2524   -0.3272           0.7571

=== DREAMER: per-model descriptives (recomputed from per-subject values) ===
       metric      model  n   mean  sd_population  sd_sample  ci95_low  ci95_high    min    max
  MAE_Valence CNN-BiLSTM 23  1.172          0.189      0.193     1.088      1.255  0.751  1.572
  MAE_Valence     Fusion 23  1.284          0.195      0.199     1.198      1.370  0.824  1.582
  MAE_Arousal CNN-BiLSTM 23  0.948          0.151      0.154     0.881      1.015  0.675  1.336
  MAE_Arousal     Fusion 23  1.016          0.190      0.194     0.932      1.100  0.679  1.453
MAE_Dominance CNN-BiLSTM 23  0.931          0.233      0.238     0.828      1.034  0.525  1.580
MAE_Dominance     Fusion 23  0.995          0.210      0.214     0.902      1.088  0.522  1.361
   R2_Valence CNN-BiLSTM 23 -0.159          0.189      0.193    -0.242     -0.075 -0.843  0.029
   R2_Valence     Fusion 23 -0.481          0.310      0.317    -0.618     -0.344 -1.130  0.116
   R2_Arousal CNN-BiLSTM 23 -0.370          0.564      0.577    -0.620     -0.121 -2.150  0.134
   R2_Arousal     Fusion 23 -0.554          0.420      0.430    -0.739     -0.368 -2.074 -0.084
 R2_Dominance CNN-BiLSTM 23 -0.319          0.515      0.527    -0.547     -0.091 -1.755  0.248
 R2_Dominance     Fusion 23 -0.580          0.555      0.567    -0.825     -0.334 -2.493  0.059

=== DREAMER: paired comparisons ===
       metric          comparison  n_pairs  mean_diff  t_ci95_low  t_ci95_high  boot_ci95_low  boot_ci95_high  wilcoxon_p  cohen_dz  wilcoxon_p_holm
  MAE_Valence Fusion - CNN-BiLSTM       23     0.1122      0.0349       0.1895         0.0398          0.1809      0.0112    0.6275           0.0335
  MAE_Arousal Fusion - CNN-BiLSTM       23     0.0675     -0.0003       0.1353         0.0018          0.1255      0.0384    0.4307           0.0553
MAE_Dominance Fusion - CNN-BiLSTM       23     0.0644     -0.0127       0.1415        -0.0090          0.1310      0.0277    0.3610           0.0553
   R2_Valence Fusion - CNN-BiLSTM       23    -0.3218     -0.4945      -0.1490        -0.4764         -0.1567      0.0013   -0.8054           0.0038
   R2_Arousal Fusion - CNN-BiLSTM       23    -0.1832     -0.3926       0.0261        -0.3554          0.0262      0.0091   -0.3784           0.0091
 R2_Dominance Fusion - CNN-BiLSTM       23    -0.2607     -0.4935      -0.0279        -0.4557         -0.0299      0.0043   -0.4843           0.0087
```
