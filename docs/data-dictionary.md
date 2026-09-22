# 数据字典（自动生成）

本文件由 `scripts/verify_data.py` 从当前下载的数据生成。

## `test.parquet`

- 行组：1
- 行数：1213

| 字段 | Arrow 类型 |
| --- | --- |
| `molecule_id` | `large_string` |
| `spectrum_id` | `large_string` |
| `ms2_mzs` | `list<element: double>` |
| `ms2_normalized_intensities` | `list<element: double>` |
| `base_peak_intensity` | `double` |
| `adduct` | `large_string` |
| `ionization_mode` | `large_string` |
| `instrument_type` | `large_string` |
| `precursor_mz` | `double` |
| `collision_energy_orig` | `large_string` |
| `collision_energy_ev` | `list<element: double>` |
| `collision_energy_orig_units` | `large_string` |

## `train.parquet`

- 行组：21
- 行数：2539608

| 字段 | Arrow 类型 |
| --- | --- |
| `ingest_lib` | `string` |
| `normalized_smiles` | `string` |
| `inchikey` | `string` |
| `inchikey14` | `string` |
| `molecular_formula` | `string` |
| `ionization_mode` | `string` |
| `instrument_type` | `string` |
| `adduct` | `string` |
| `adduct_orig` | `string` |
| `precursor_mz` | `double` |
| `precursor_error_ppm` | `double` |
| `ms2_mzs` | `list<element: double>` |
| `ms2_normalized_intensities` | `list<element: double>` |
| `num_peaks` | `int64` |
| `base_peak_intensity` | `double` |
| `collision_energy_ev` | `list<element: double>` |
| `collision_energy_orig` | `string` |
| `collision_energy_orig_units` | `string` |

## `sample_submission.csv`

- 行数：400

| 字段 | pandas 类型 |
| --- | --- |
| `molecule_id` | `str` |
| `smiles` | `str` |
