# Kết quả RPDI-EE trên vLLM theo ngân sách token

- Số câu chung: **500** (id 100–599)
- Giải mã greedy. Mỗi cặp (câu, phương pháp) sinh một lần ở ngân sách lớn nhất; các ngân sách nhỏ hơn suy ra bằng cách cắt chuỗi token.

## Accuracy % (token trung bình)

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 54.4 (437) | 73.8 (522) | 79.4 (550) | 80.0 (568) | 80.2 (578) | 80.2 (594) | 80.0 (611) | 80.2 (676) | 80.2 (807) |
| `hardtrunc_N130` | 52.2 (433) | 71.8 (519) | 77.2 (543) | 78.4 (562) | 78.6 (577) | 78.6 (604) | 78.6 (628) | 78.8 (727) | 78.6 (923) |
| `random` | 54.2 (442) | 72.8 (529) | 78.0 (559) | 79.4 (586) | 79.8 (605) | 79.8 (638) | 79.8 (669) | 79.6 (792) | 79.6 (1037) |
| `random_matched` | 56.0 (444) | 73.4 (526) | 78.2 (555) | 79.6 (582) | 80.0 (602) | 79.8 (637) | 79.8 (672) | 79.6 (811) | 79.6 (1090) |
| `rpdi_W512_l2.0_newline` | 46.8 (455) | 53.8 (583) | 63.4 (674) | 74.4 (784) | 77.2 (850) | 78.8 (957) | 78.8 (1059) | 78.8 (1452) | 79.2 (2239) |
| `rpdi_W512_l2.0_wait` | 46.6 (455) | 54.6 (584) | 63.8 (674) | 74.2 (781) | 76.2 (848) | 77.0 (964) | 77.6 (1073) | 77.8 (1485) | 78.2 (2304) |
| `rpdi_legacy_W16_l1.5` | 54.0 (445) | 75.6 (527) | 80.0 (558) | 81.4 (583) | 81.2 (601) | 81.0 (636) | 81.2 (671) | 81.0 (810) | 81.0 (1089) |
| `standard` | 46.4 (455) | 54.6 (583) | 64.0 (673) | 74.8 (780) | 77.6 (848) | 79.0 (964) | 78.8 (1075) | 78.4 (1509) | 79.2 (2377) |

## Tỷ lệ cạn ngân sách %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 52.0 | 19.4 | 6.2 | 2.2 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 |
| `hardtrunc_N130` | 50.2 | 17.8 | 5.2 | 3.2 | 2.8 | 2.4 | 2.4 | 2.4 | 2.4 |
| `random` | 53.4 | 18.6 | 8.0 | 4.2 | 3.4 | 3.0 | 3.0 | 3.0 | 3.0 |
| `random_matched` | 50.2 | 18.2 | 7.6 | 4.2 | 3.6 | 3.4 | 3.4 | 3.4 | 3.4 |
| `rpdi_W512_l2.0_newline` | 59.6 | 43.0 | 28.8 | 15.2 | 11.2 | 10.2 | 9.6 | 9.6 | 9.6 |
| `rpdi_W512_l2.0_wait` | 59.6 | 43.2 | 28.2 | 15.0 | 11.8 | 11.0 | 10.2 | 10.0 | 10.0 |
| `rpdi_legacy_W16_l1.5` | 51.8 | 18.4 | 7.4 | 4.2 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 |
| `standard` | 59.4 | 43.2 | 28.0 | 15.2 | 11.8 | 11.0 | 10.6 | 10.6 | 10.6 |

## Tỷ lệ phanh trong pha suy nghĩ %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hardtrunc_N100` | 93.2 | 93.2 | 93.2 | 93.2 | 93.2 | 93.2 | 93.2 | 93.2 | 93.2 |
| `hardtrunc_N130` | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 |
| `random` | 81.4 | 81.8 | 81.8 | 81.8 | 81.8 | 81.8 | 81.8 | 81.8 | 81.8 |
| `random_matched` | 58.2 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 |
| `rpdi_W512_l2.0_newline` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rpdi_W512_l2.0_wait` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rpdi_legacy_W16_l1.5` | 60.4 | 60.8 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 |
| `standard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## So sánh chính (biên δ = 3 điểm, bootstrap 10000 lần)

| Target | Reference | n | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Phục hồi | `standard` cần ngân sách | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| `rpdi_legacy_W16_l1.5@768` | `hardtrunc_N100@768` | 500 | 75.6 | 73.8 | +1.8 | [-1.8; +5.4] | [-1.2; +4.8] | 0.374 | 1.01 | 109% | 2048 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@768` | `hardtrunc_N130@768` | 500 | 75.6 | 71.8 | +3.8 | [+0.4; +7.2] | [+1.0; +6.6] | 0.037 | 1.02 | 122% | 2048 | Hơn |
| `rpdi_legacy_W16_l1.5@768` | `random@768` | 500 | 75.6 | 72.8 | +2.8 | [-0.6; +6.2] | [+0.0; +5.6] | 0.135 | 1.00 | 115% | 2048 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@768` | `random_matched@768` | 500 | 75.6 | 73.4 | +2.2 | [-0.8; +5.2] | [-0.4; +4.8] | 0.2 | 1.00 | 112% | 2048 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `hardtrunc_N100@1024` | 500 | 80.0 | 79.4 | +0.6 | [-2.6; +3.8] | [-2.0; +3.2] | 0.804 | 1.01 | 104% | > 16384 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `hardtrunc_N130@1024` | 500 | 80.0 | 77.2 | +2.8 | [-0.2; +5.8] | [+0.4; +5.4] | 0.0925 | 1.03 | 121% | > 16384 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `random@1024` | 500 | 80.0 | 78.0 | +2.0 | [-1.0; +5.0] | [-0.6; +4.6] | 0.245 | 1.00 | 114% | > 16384 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `random_matched@1024` | 500 | 80.0 | 78.2 | +1.8 | [-1.0; +4.4] | [-0.4; +4.0] | 0.243 | 1.00 | 113% | > 16384 | Không kém hơn quá δ |
| `hardtrunc_N100@1024` | `standard@16384` | 500 | 79.4 | 79.2 | +0.2 | [-3.4; +3.6] | [-2.8; +3.2] | 1 | 0.23 | 101% | > 16384 | Chưa kết luận |
| `hardtrunc_N130@1024` | `standard@16384` | 500 | 77.2 | 79.2 | -2.0 | [-5.2; +1.2] | [-4.8; +0.6] | 0.282 | 0.23 | 87% | 2048 | Chưa kết luận |
| `random@1024` | `standard@16384` | 500 | 78.0 | 79.2 | -1.2 | [-4.6; +2.2] | [-4.0; +1.6] | 0.556 | 0.24 | 92% | 3072 | Chưa kết luận |
| `random_matched@1024` | `standard@16384` | 500 | 78.2 | 79.2 | -1.0 | [-4.0; +2.0] | [-3.6; +1.6] | 0.603 | 0.23 | 93% | 3072 | Chưa kết luận |
| `hardtrunc_N100@16384` | `standard@16384` | 500 | 80.2 | 79.2 | +1.0 | [-2.6; +4.4] | [-2.0; +4.0] | 0.653 | 0.34 | – | > 16384 | Không kém hơn quá δ |
| `hardtrunc_N130@16384` | `standard@16384` | 500 | 78.6 | 79.2 | -0.6 | [-3.8; +2.4] | [-3.2; +2.0] | 0.801 | 0.39 | – | 3072 | Chưa kết luận |
| `random@16384` | `standard@16384` | 500 | 79.6 | 79.2 | +0.4 | [-2.6; +3.6] | [-2.2; +3.0] | 0.901 | 0.44 | – | > 16384 | Không kém hơn quá δ |
| `random_matched@16384` | `standard@16384` | 500 | 79.6 | 79.2 | +0.4 | [-2.4; +3.2] | [-2.0; +2.8] | 0.89 | 0.46 | – | > 16384 | Tương đương (±δ) |

- `rpdi_legacy_W16_l1.5@768` so với `hardtrunc_N100@768`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 45 câu; ngược lại: 36 câu.
- `rpdi_legacy_W16_l1.5@768` so với `hardtrunc_N130@768`: **Hơn (cận dưới KTC 95% > 0)**. Target đúng mà reference sai: 47 câu; ngược lại: 28 câu.
- `rpdi_legacy_W16_l1.5@768` so với `random@768`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 45 câu; ngược lại: 31 câu.
- `rpdi_legacy_W16_l1.5@768` so với `random_matched@768`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 36 câu; ngược lại: 25 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `hardtrunc_N100@1024`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 34 câu; ngược lại: 31 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `hardtrunc_N130@1024`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 37 câu; ngược lại: 23 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `random@1024`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 35 câu; ngược lại: 25 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `random_matched@1024`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 28 câu; ngược lại: 19 câu.
- `hardtrunc_N100@1024` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 41 câu; ngược lại: 40 câu.
- `hardtrunc_N130@1024` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 30 câu; ngược lại: 40 câu.
- `random@1024` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 33 câu; ngược lại: 39 câu.
- `random_matched@1024` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 27 câu; ngược lại: 32 câu.
- `hardtrunc_N100@16384` so với `standard@16384`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 42 câu; ngược lại: 37 câu.
- `hardtrunc_N130@16384` so với `standard@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 30 câu; ngược lại: 33 câu.
- `random@16384` so với `standard@16384`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 33 câu; ngược lại: 31 câu.
- `random_matched@16384` so với `standard@16384`: **Tương đương trong biên ±δ (KTC 90% nằm trong (−δ, +δ), tương đương kiểm định TOST α = 0,05)**. Target đúng mà reference sai: 27 câu; ngược lại: 25 câu.

- *Phục hồi* = (Acc T − Acc baseline ở ngân sách của T) / (Acc R − Acc baseline ở ngân sách của T).
- *Cần ngân sách* = ngân sách nhỏ nhất để `standard` đạt accuracy bằng target trên cùng tập câu.

