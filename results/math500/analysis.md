# Kết quả RPDI-EE trên vLLM theo ngân sách token

- Số câu chung: **500** (id 0–499)
- Giải mã greedy. Mỗi cặp (câu, phương pháp) sinh một lần ở ngân sách lớn nhất; các ngân sách nhỏ hơn suy ra bằng cách cắt chuỗi token.

## Accuracy % (token trung bình)

| Phương pháp | 1024 | 1536 | 2048 | 3072 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 42.2 (899) | 54.2 (1168) | 58.2 (1375) | 61.6 (1696) | 63.6 (1976) |
| `rpdi_legacy_W16_l1.5` | 43.4 (910) | 53.2 (1190) | 60.4 (1414) | 65.0 (1755) | 67.0 (2036) |
| `standard` | 10.0 (1004) | 28.0 (1415) | 39.6 (1744) | 55.6 (2227) | 60.0 (2619) |

## Tỷ lệ cạn ngân sách %

| Phương pháp | 1024 | 1536 | 2048 | 3072 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 62.4 | 46.0 | 36.6 | 29.0 | 25.8 |
| `rpdi_legacy_W16_l1.5` | 64.6 | 48.4 | 39.0 | 29.4 | 26.2 |
| `standard` | 89.8 | 71.2 | 58.4 | 40.8 | 36.0 |

## Tỷ lệ phanh trong pha suy nghĩ %

| Phương pháp | 1024 | 1536 | 2048 | 3072 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 |
| `rpdi_legacy_W16_l1.5` | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 |
| `standard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## So sánh chính (biên δ = 3 điểm, bootstrap 10000 lần)

| Target | Reference | n | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Phục hồi | `standard` cần ngân sách | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| `rpdi_legacy_W16_l1.5@1024` | `standard@1024` | 500 | 43.4 | 10.0 | +33.4 | [+29.2; +37.6] | [+29.8; +37.0] | 1.44e-46 | 0.91 | – | 3072 | Hơn |
| `rpdi_legacy_W16_l1.5@2048` | `standard@2048` | 500 | 60.4 | 39.6 | +20.8 | [+16.8; +24.8] | [+17.4; +24.2] | 1.55e-22 | 0.81 | – | > 4096 | Hơn |
| `rpdi_legacy_W16_l1.5@4096` | `standard@4096` | 500 | 67.0 | 60.0 | +7.0 | [+3.2; +10.8] | [+3.8; +10.2] | 0.000562 | 0.78 | – | > 4096 | Hơn |
| `hardtrunc_N100@1024` | `rpdi_legacy_W16_l1.5@1024` | 500 | 42.2 | 43.4 | -1.2 | [-5.0; +2.6] | [-4.4; +2.0] | 0.606 | 0.99 | 96% | 3072 | Chưa kết luận |
| `hardtrunc_N100@2048` | `rpdi_legacy_W16_l1.5@2048` | 500 | 58.2 | 60.4 | -2.2 | [-6.0; +1.6] | [-5.4; +1.0] | 0.294 | 0.97 | 89% | 4096 | Chưa kết luận |
| `hardtrunc_N100@4096` | `rpdi_legacy_W16_l1.5@4096` | 500 | 63.6 | 67.0 | -3.4 | [-7.0; +0.4] | [-6.4; -0.4] | 0.0893 | 0.97 | 51% | > 4096 | Chưa kết luận |
| `rpdi_legacy_W16_l1.5@2048` | `standard@4096` | 500 | 60.4 | 60.0 | +0.4 | [-3.4; +4.2] | [-2.8; +3.6] | 0.917 | 0.54 | 102% | > 4096 | Chưa kết luận |
| `hardtrunc_N100@2048` | `standard@4096` | 500 | 58.2 | 60.0 | -1.8 | [-5.8; +2.2] | [-5.2; +1.6] | 0.426 | 0.53 | 91% | 4096 | Chưa kết luận |

- `rpdi_legacy_W16_l1.5@1024` so với `standard@1024`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 170 câu; ngược lại: 3 câu.
- `rpdi_legacy_W16_l1.5@2048` so với `standard@2048`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 116 câu; ngược lại: 12 câu.
- `rpdi_legacy_W16_l1.5@4096` so với `standard@4096`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 67 câu; ngược lại: 32 câu.
- `hardtrunc_N100@1024` so với `rpdi_legacy_W16_l1.5@1024`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 44 câu; ngược lại: 50 câu.
- `hardtrunc_N100@2048` so với `rpdi_legacy_W16_l1.5@2048`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 40 câu; ngược lại: 51 câu.
- `hardtrunc_N100@4096` so với `rpdi_legacy_W16_l1.5@4096`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 36 câu; ngược lại: 53 câu.
- `rpdi_legacy_W16_l1.5@2048` so với `standard@4096`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 47 câu; ngược lại: 45 câu.
- `hardtrunc_N100@2048` so với `standard@4096`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 46 câu; ngược lại: 55 câu.

- *Phục hồi* = (Acc T − Acc baseline ở ngân sách của T) / (Acc R − Acc baseline ở ngân sách của T).
- *Cần ngân sách* = ngân sách nhỏ nhất để `standard` đạt accuracy bằng target trên cùng tập câu.

