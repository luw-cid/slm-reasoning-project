# Kết quả RPDI-EE trên vLLM theo ngân sách token

- Số câu chung: **500** (id 100–599)
- Giải mã greedy. Mỗi cặp (câu, phương pháp) sinh một lần ở ngân sách lớn nhất; các ngân sách nhỏ hơn suy ra bằng cách cắt chuỗi token.

## Accuracy % (token trung bình)

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 51.2 (444) | 71.0 (536) | 76.2 (568) | 77.0 (593) | 77.6 (608) | 77.8 (628) | 77.8 (643) | 78.0 (679) | 78.0 (728) |
| `rpdi_legacy_W16_l1.5` | 51.6 (445) | 73.2 (534) | 78.2 (568) | 80.0 (592) | 80.4 (605) | 80.4 (627) | 81.0 (644) | 81.4 (669) | 81.4 (669) |
| `standard` | 39.2 (463) | 52.2 (600) | 61.4 (699) | 71.6 (826) | 77.0 (901) | 80.2 (985) | 81.2 (1033) | 82.4 (1115) | 82.4 (1167) |

## Tỷ lệ cạn ngân sách %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 54.4 | 19.0 | 7.8 | 3.6 | 2.2 | 1.8 | 1.4 | 0.6 | 0.6 |
| `rpdi_legacy_W16_l1.5` | 50.2 | 19.8 | 7.4 | 2.6 | 2.6 | 1.8 | 1.0 | 0.0 | 0.0 |
| `standard` | 63.8 | 45.6 | 32.4 | 19.4 | 11.8 | 6.2 | 3.8 | 1.0 | 0.6 |

## Tỷ lệ phanh trong pha suy nghĩ %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 |
| `rpdi_legacy_W16_l1.5` | 64.8 | 65.4 | 65.4 | 65.4 | 65.4 | 65.4 | 65.4 | 65.4 | 65.4 |
| `standard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## So sánh chính (biên δ = 3 điểm, bootstrap 10000 lần)

| Target | Reference | n | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Phục hồi | `standard` cần ngân sách | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| `rpdi_legacy_W16_l1.5@768` | `standard@768` | 500 | 73.2 | 52.2 | +21.0 | [+16.6; +25.4] | [+17.2; +24.6] | 3.8e-19 | 0.89 | – | 2048 | Hơn |
| `rpdi_legacy_W16_l1.5@1024` | `standard@1024` | 500 | 78.2 | 61.4 | +16.8 | [+13.0; +20.6] | [+13.6; +20.0] | 4.7e-17 | 0.81 | – | 3072 | Hơn |
| `hardtrunc_N100@768` | `rpdi_legacy_W16_l1.5@768` | 500 | 71.0 | 73.2 | -2.2 | [-5.8; +1.6] | [-5.2; +1.0] | 0.3 | 1.00 | 90% | 1536 | Chưa kết luận |
| `hardtrunc_N100@1024` | `rpdi_legacy_W16_l1.5@1024` | 500 | 76.2 | 78.2 | -2.0 | [-5.2; +1.2] | [-4.8; +0.8] | 0.275 | 1.00 | 88% | 2048 | Chưa kết luận |
| `rpdi_legacy_W16_l1.5@1024` | `standard@16384` | 500 | 78.2 | 82.4 | -4.2 | [-7.0; -1.6] | [-6.6; -2.0] | 0.0038 | 0.49 | 80% | 3072 | Kém hơn (chưa chứng minh trong δ) |
| `hardtrunc_N100@1024` | `standard@16384` | 500 | 76.2 | 82.4 | -6.2 | [-9.2; -3.0] | [-8.8; -3.6] | 0.000152 | 0.49 | 70% | 2048 | Kém hơn (chưa chứng minh trong δ) |
| `rpdi_legacy_W16_l1.5@16384` | `standard@16384` | 500 | 81.4 | 82.4 | -1.0 | [-3.4; +1.4] | [-3.0; +1.0] | 0.511 | 0.57 | – | 8192 | Chưa kết luận |

- `rpdi_legacy_W16_l1.5@768` so với `standard@768`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 127 câu; ngược lại: 22 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `standard@1024`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 97 câu; ngược lại: 13 câu.
- `hardtrunc_N100@768` so với `rpdi_legacy_W16_l1.5@768`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 41 câu; ngược lại: 52 câu.
- `hardtrunc_N100@1024` so với `rpdi_legacy_W16_l1.5@1024`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 29 câu; ngược lại: 39 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `standard@16384`: **Kém hơn, chưa chứng minh được nằm trong biên δ (cận dưới KTC 95% ≤ −δ)**. Target đúng mà reference sai: 14 câu; ngược lại: 35 câu.
- `hardtrunc_N100@1024` so với `standard@16384`: **Kém hơn, chưa chứng minh được nằm trong biên δ (cận dưới KTC 95% ≤ −δ)**. Target đúng mà reference sai: 17 câu; ngược lại: 48 câu.
- `rpdi_legacy_W16_l1.5@16384` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 16 câu; ngược lại: 21 câu.

- *Phục hồi* = (Acc T − Acc baseline ở ngân sách của T) / (Acc R − Acc baseline ở ngân sách của T).
- *Cần ngân sách* = ngân sách nhỏ nhất để `standard` đạt accuracy bằng target trên cùng tập câu.

