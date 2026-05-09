# Section 9 Findings — Cosine Similarity Validation (wondr)

## Methodology recap

- **D1 (Embedding-level)**: cosine sim antar bulan dari mean centroid IndoBERT per (topic, bulan). Sensitif terhadap pergeseran semantic.
- **D2 (c-TF-IDF level)**: cosine sim antar bulan dari binary word-presence vector (top-words BERTopic). Sensitif terhadap pergeseran lexical.
- **Threshold kategorisasi**: D1 ≥ 0.95 (stable semantic), D2 ≥ 0.6 (stable lexical).

## Pattern distribution

- **Topic drift**: 7 topic
- **Stable**: 4 topic
- **Lexical-anchor drift (rare)**: 1 topic
- **Paraphrase (semantic stable, lexical drift)**: 1 topic

## Per-topic findings

| Topic | Label | D1 mean | D2 mean | Pattern |
|-------|-------|---------|---------|---------|
| T0 | Mobile Banking comparison | 0.9758 | 0.6061 | Stable |
| T1 | Verifikasi wajah gagal | 0.9811 | 0.8333 | Stable |
| T2 | Saldo terpotong / transaksi gagal | 0.9817 | 0.6364 | Stable |
| T3 | Gangguan & maintenance | 0.9251 | 0.3182 | Topic drift |
| T4 | Tarik tunai tanpa kartu | 0.9612 | 0.8485 | Stable |
| T5 | Aplikasi gak bisa dibuka | 0.9195 | 0.1758 | Topic drift |
| T6 | OTP / kode verifikasi email | 0.9263 | 0.8606 | Lexical-anchor drift (rare) |
| T7 | Error 'kendala tim perbaiki' | 0.9482 | 0.197 | Topic drift |
| T8 | Password / PIN salah | 0.9589 | 0.4424 | Paraphrase (semantic stable, lexical drift) |
| T9 | Sesi berakhir saat login | 0.8575 | 0.1515 | Topic drift |
| T10 | Premature launch complaint | 0.8351 | 0.2655 | Topic drift |
| T11 | Permintaan perbaikan bug | 0.8488 | 0.0939 | Topic drift |
| T12 | Limit top-up wallet | 0.8323 | 0.3745 | Topic drift |

## Outlier months (candidate untuk narrative BAB 4)

Bulan yang sim mean-nya (ke bulan lain) jauh di bawah baseline topic — mengindikasikan disruption / topic shift di bulan tersebut.

### T0: Mobile Banking comparison
- **D2 outlier**: M10 (sim=0.2909)

### T1: Verifikasi wajah gagal
- **D2 outlier**: M10 (sim=0.6)

### T2: Saldo terpotong / transaksi gagal
- **D2 outlier**: M1 (sim=0.4), M4 (sim=0.4727)

### T3: Gangguan & maintenance
- **D1 outlier**: M1 (sim=0.8523)
- **D2 outlier**: M1 (sim=0.0), M2 (sim=0.0), M4 (sim=0.1636)

### T4: Tarik tunai tanpa kartu
- **D2 outlier**: M9 (sim=0.6545)

### T5: Aplikasi gak bisa dibuka
- **D1 outlier**: M2 (sim=0.8069)
- **D2 outlier**: M1 (sim=0.0), M3 (sim=0.0), M8 (sim=0.0)

### T6: OTP / kode verifikasi email
- **D1 outlier**: M11 (sim=0.8716)
- **D2 outlier**: M11 (sim=0.6)

### T7: Error 'kendala tim perbaiki'
- **D2 outlier**: M3 (sim=0.0)

### T8: Password / PIN salah
- **D2 outlier**: M5 (sim=0.2909), M9 (sim=0.2545), M10 (sim=0.2909)

### T9: Sesi berakhir saat login
- **D1 outlier**: M2 (sim=0.7996), M3 (sim=0.7947), M10 (sim=0.7374)
- **D2 outlier**: M1 (sim=0.0), M2 (sim=0.0), M3 (sim=0.0), M10 (sim=0.0), M11 (sim=0.0)

### T10: Premature launch complaint
- **D1 outlier**: M9 (sim=0.7705), M11 (sim=0.6993)
- **D2 outlier**: M9 (sim=0.0), M12 (sim=0.0)

### T11: Permintaan perbaikan bug
- **D1 outlier**: M2 (sim=0.7559), M3 (sim=0.7142)

### T12: Limit top-up wallet
- **D1 outlier**: M1 (sim=0.6958), M2 (sim=0.7698), M3 (sim=0.7749)
- **D2 outlier**: M1 (sim=0.0), M2 (sim=0.0), M3 (sim=0.0)

## Cross-validation insights

Topic dengan outlier signal di **kedua** D1 dan D2 = strongest evidence untuk pergeseran substantif (bukan cuma noise).

Topic dengan signal cuma di D2 (vocab shift tapi semantic stable) = paraphrase pattern, kemungkinan user pakai kata berbeda untuk keluhan yang sama (misal 'lemot' → 'lambat' → 'lag').

## Ke BAB 4

_TODO: integrate findings ini dengan tabel ringkasan dari Section 8 (frequency / proportion peak). Topic dengan outage signature M5 (Section 8) — apakah cosine sim-nya juga drop di transisi M4→M5 atau M5→M6? Cross-confirmation akan memperkuat narrasi._