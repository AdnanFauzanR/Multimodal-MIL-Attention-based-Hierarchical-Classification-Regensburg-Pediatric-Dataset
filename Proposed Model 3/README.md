# Proposed Model 3 — FiLM + Scaled Dot-Product Cross-Attention MIL

Folder ini bersifat **terisolasi penuh** (tidak mengimpor atau mengubah file di `Proposed Model 2/`, `proposed_model/`, maupun root). Hanya membaca data mentah (`../app_data_filtered.csv`, `../image_manifest_filtered.csv`, `../US_Pictures_clean/`) dan menyimpan cache ke `Proposed Model 3/cache/` serta hasil eksperimen ke `Proposed Model 3/runs/`.

## Ringkasan Arsitektur & Perbaikan dari v4
1. **Backbone Trimmed EfficientNet-B0 (`backbone_mode`)**:
   - `stage7_8`: freeze `features[0..6]`, train `features[7, 8]` (~1,13M param).
   - `stage7_only`: freeze `features[0..6]`, train `features[7]`, buang `features[8]` (~717K param, 320-ch).
   - `stage8_only`: freeze `features[0..7]`, train `features[8]` (~412K param).
2. **Scaled Dot-Product Cross-Attention MIL (`cross_attn`)**:
   - Mengikuti *Attention Is All You Need* (4 head, $\text{softmax}(QK^\top / (\sqrt{d_k}\,\tau))$), tanpa *negative entropy penalty* yang dulu memaksa atensi seragam 0,997.
   - Mendukung perbandingan langsung dengan `mv_mean` dan `mv_lstm` pada tugas **Binary Diagnosis**, **Binary Severity**, dan **Hierarchical 3-Class** (baik Image-Only maupun Multimodal).
3. **Residual FiLM + Direct Clinical Anchor (`film`)**:
   - Modulasi *Feature-wise Linear Modulation* $\tilde{\mathbf{z}} = (1 + \Delta\boldsymbol{\gamma}(\mathbf{e}_{\text{clin}})) \odot \mathbf{z} + \boldsymbol{\beta}(\mathbf{e}_{\text{clin}})$ yang diinisialisasi identitas, dipadukan dengan jangkar klinis dan *linear clinical skip* terregularisasi.
4. **8 Konsep Sonografis Terkurasi + Auxiliary Visual Loss**:
   - Mengembalikan `Appendix_Diameter` ($\ge 6$ mm), `Appendix_Wall_Layers`, `Appendix_on_US`, `Target_Sign`, `Pathological_Lymph_Nodes`, `Surrounding_Tissue_Reaction`, `Free_Fluids`, dan `Severe_Complication_Signs`.
5. **Balanced Loss + Checkpoint AUROC + Nested Threshold Calibration**:
   - Menyeimbangkan gradien `diag` vs `comp` (1:1), memakai *Validation Macro-AUROC* untuk checkpoint, dan kalibrasi ambang $(\tau_{\text{diag}}, \tau_{\text{comp}})$ secara *nested out-of-fold* bebas *leakage*.

## Urutan & Eksekusi Paralel Notebook
1. `01_Preprocessing_Audit_dan_Baseline_B1.ipynb` — Audit kohort, perbandingan **Clean 28 vs Leaky 53**, bangun cache USG, dan benchmark B1 Tabular-Only.
2. `02_Ablasi_Backbone_dan_Aggregator_B2.ipynb` — Jalankan **Bagian 1** terlebih dahulu untuk ablasi 3 backbone (`stage7_8` vs `stage7_only` vs `stage8_only`) dan tentukan `BEST_BACKBONE`. Catat `BEST_BACKBONE` terbaik ke `03` dan `04`. (Bagian 2–3 di `02` opsional jika ingin sequential di satu notebook, namun dapat digantikan oleh eksekusi paralel `03` + `04` di bawah).
3. **Eksekusi Paralel Tanpa Duplikasi (2 Worker GPU di RTX 3060 6GB, masing-masing ~0,98 GiB VRAM):**
   - **Worker 1 — `04_Paralel_Worker_ImageOnly.ipynb`:** Menjalankan **15 eksperimen unik Image-Only** (9 B2 Aggregator -> `02_b2_image_only_{task}_{agg}` dan 6 Bag/View `appendix_single` & `appendix_plus_1_2` -> `03_bag_image_only_{task}_{bmode}` sambil me-reuse varian `all` dari `02_b2_image_only_{task}_cross_attn`).
   - **Worker 2 — `03_Proposed_Model_3_Multimodal_FiLM_dan_XAI.ipynb`:** Menjalankan **18 eksperimen unik Multimodal** dengan *zero-duplication* — otomatis me-reuse `02_backbone_{BEST_BACKBONE}` (untuk `hierarchical + cross_attn + film + concepts + all`) serta `02_b2_multimodal_{task}_cross_attn` (untuk `binary_diag`/`binary_sev` + `all`), menggabungkan tabel hasil `04` bila sudah tersedia, dan menjalankan dashboard XAI (Grad-CAM, Attention, 8 Konsep, Integrated Gradients, FiLM).
4. `05_Audit_Diagnostik_Overfit_MIL_dan_BagSize_Shortcut.ipynb` — Audit diagnostik *reproducible* (pure CPU) untuk kurva *overfitting* 5-fold, dekomposisi *loss*, kerja MIL-Attention, FiLM & 8 konsep, serta pembuktian empiris *bag-size shortcut* (`mv_lstm` vs `cross_attn`), menyimpan seluruh tabel CSV, grafik PNG, dan laporan ke `runs/05_diagnostic_audits/`.
5. `06_Analisis_Statistik_Q1_Kalibrasi_Subgrup_dan_Master_Rekap.ipynb` — Analisis statistik berpasangan (*paired stratified bootstrap* B=5000 & *exact McNemar*), Brier/ECE calibration, *Decision Curve Analysis* (DCA), audit tantangan dataset, subgrup klinis & *error audit*, serta rekap master berintegritas untuk draft jurnal Q1/Q2 di `runs/06_q1_statistical_analysis/`.
6. `07_Eksperimen_Lanjutan_TwoStage_Concept_dan_Ensemble.ipynb` — Tiga uji lanjutan 5-fold CV di `runs/07_advanced_two_stage_and_ensemble/`: **(1)** *Stage-1 Pure Image Concept Pretraining -> Stage-2 Frozen Visual Multimodal*, **(2)** *Nested-OOF Decision-Level Ensemble & Uncertainty-Based Adaptive Selector (`Clinical + Image` & `PM3 + Clinical`)*, dan **(3)** *Stage-1 Clinical-Guided Concept Teaching -> Stage-2 End-to-End Hierarchical Fine-Tuning*, beserta uji statistik berpasangan & kalibrasi Brier/ECE.
7. `08_Decoupled_DualHead_AntiOverfit_dan_GatedMIL.ipynb` — Eksperimen cabang independen terpisah (`ClinFiLM-Decoupled`) untuk mengatasi *coupled overfitting* dan *Clinical-Conditioned Gated MIL* pada cabang komplikasi.
8. `09_PM3_DualStage7_AsyncPartialFreeze.ipynb` — Eksperimen *Dual-Stage7 Forked Asynchronous Partial-Freeze* (`ClinFiLM-APF`) dengan regularisasi asimetris, *Model EMA*, dan *loss-guarded early stopping* serta ansambel hibrida dengan `Hier-RF`.
9. `10_Replikasi_Paper_Referensi_MVCBM_dan_Barthwal2026.ipynb` — Skrip/notebook replikasi benchmark eksternal (`MVCBM` & `Barthwal 2026`).
10. `Laporan_dan_Presentasi/` — Manuskrip jurnal terformat IJAIN (`6025261004-Laporan FP Versi 1.docx`, `IJAIN_ClinFiLM_HMIL_Pediatric_Appendicitis.md`, beserta figur 300 DPI di `ijain_figures/`).

## Quick Start & Dependensi

Folder `Proposed Model 3/cache/` dan `Proposed Model 3/runs/` tidak disertakan di repositori (di-ignore di `.gitignore`) dan akan **dibangun ulang secara otomatis** dari `../app_data_filtered.csv`, `../image_manifest_filtered.csv`, dan `../US_Pictures_clean/` saat notebook `01` s.d. `09` dijalankan.

```bash
pip install -r "Proposed Model 3/requirements.txt"
```
