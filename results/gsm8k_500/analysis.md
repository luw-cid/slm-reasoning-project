# Kết quả RPDI-EE trên vLLM theo ngân sách token

- Số câu chung: **500** (id 100–599)
- Giải mã greedy. Mỗi cặp (câu, phương pháp) sinh một lần ở ngân sách lớn nhất; các ngân sách nhỏ hơn suy ra bằng cách cắt chuỗi token.

## Accuracy % (token trung bình)

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rpdi_W512_l2.0_newline` | 46.8 (455) | 53.8 (583) | 63.4 (674) | 74.4 (784) | 77.2 (850) | 78.8 (957) | 78.8 (1059) | 78.8 (1452) | 79.2 (2239) |
| `rpdi_W512_l2.0_wait` | 46.6 (455) | 54.6 (584) | 63.8 (674) | 74.2 (781) | 76.2 (848) | 77.0 (964) | 77.6 (1073) | 77.8 (1485) | 78.2 (2304) |
| `rpdi_legacy_W16_l1.5` | 54.0 (445) | 75.6 (527) | 80.0 (558) | 81.4 (583) | 81.2 (601) | 81.0 (636) | 81.2 (671) | 81.0 (810) | 81.0 (1089) |
| `standard` | 46.4 (455) | 54.6 (583) | 64.0 (673) | 74.8 (780) | 77.6 (848) | 79.0 (964) | 78.8 (1075) | 78.4 (1509) | 79.2 (2377) |

## Tỷ lệ cạn ngân sách %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rpdi_W512_l2.0_newline` | 59.6 | 43.0 | 28.8 | 15.2 | 11.2 | 10.2 | 9.6 | 9.6 | 9.6 |
| `rpdi_W512_l2.0_wait` | 59.6 | 43.2 | 28.2 | 15.0 | 11.8 | 11.0 | 10.2 | 10.0 | 10.0 |
| `rpdi_legacy_W16_l1.5` | 51.8 | 18.4 | 7.4 | 4.2 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 |
| `standard` | 59.4 | 43.2 | 28.0 | 15.2 | 11.8 | 11.0 | 10.6 | 10.6 | 10.6 |

## Tỷ lệ phanh trong pha suy nghĩ %

| Phương pháp | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 | 16384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rpdi_W512_l2.0_newline` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rpdi_W512_l2.0_wait` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rpdi_legacy_W16_l1.5` | 60.4 | 60.8 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 |
| `standard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Kiểm tra pipeline: vLLM so với HF ở ngân sách 768

Đạt khi accuracy lệch không quá 2 điểm trên cùng tập câu.

| vLLM | HF | n | Acc vLLM | Acc HF | Δ | Đồng thuận đúng/sai | Trùng văn bản | Trùng số token suy nghĩ | Kết quả |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `standard` | `standard` | 200 | 55.0 | 52.5 | +2.5 | 66.5% | 0.0% | 24.5% | KHÔNG ĐẠT |
| `rpdi_legacy_W16_l1.5` | `rpdi_W16_l1.5` | 200 | 76.0 | 71.5 | +4.5 | 78.5% | 0.0% | 0.5% | KHÔNG ĐẠT |

## So sánh chính (biên δ = 3 điểm, bootstrap 10000 lần)

| Target | Reference | n | Acc T | Acc R | Δ (T−R) | KTC 95% | KTC 90% | McNemar p | Token T/R | Phục hồi | `standard` cần ngân sách | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| `rpdi_legacy_W16_l1.5@768` | `standard@16384` | 500 | 75.6 | 79.2 | -3.6 | [-7.0; -0.4] | [-6.4; -0.8] | 0.0474 | 0.22 | 85% | 2048 | Kém hơn (chưa chứng minh trong δ) |
| `rpdi_legacy_W16_l1.5@768` | `rpdi_W512_l2.0_wait@16384` | 500 | 75.6 | 78.2 | -2.6 | [-6.0; +0.6] | [-5.4; +0.2] | 0.16 | 0.23 | 89% | 2048 | Chưa kết luận |
| `rpdi_legacy_W16_l1.5@768` | `rpdi_W512_l2.0_newline@16384` | 500 | 75.6 | 79.2 | -3.6 | [-7.0; -0.4] | [-6.4; -0.8] | 0.0444 | 0.24 | 85% | 2048 | Kém hơn (chưa chứng minh trong δ) |
| `rpdi_legacy_W16_l1.5@1024` | `standard@16384` | 500 | 80.0 | 79.2 | +0.8 | [-2.2; +3.8] | [-1.8; +3.4] | 0.699 | 0.23 | 105% | > 16384 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `rpdi_W512_l2.0_wait@16384` | 500 | 80.0 | 78.2 | +1.8 | [-1.2; +4.8] | [-0.8; +4.4] | 0.314 | 0.24 | 113% | > 16384 | Không kém hơn quá δ |
| `rpdi_legacy_W16_l1.5@1024` | `rpdi_W512_l2.0_newline@16384` | 500 | 80.0 | 79.2 | +0.8 | [-2.2; +3.8] | [-1.8; +3.4] | 0.699 | 0.25 | 105% | > 16384 | Không kém hơn quá δ |

- `rpdi_legacy_W16_l1.5@768` so với `standard@16384`: **Kém hơn, chưa chứng minh được nằm trong biên δ (cận dưới KTC 95% ≤ −δ)**. Target đúng mà reference sai: 28 câu; ngược lại: 46 câu.
- `rpdi_legacy_W16_l1.5@768` so với `rpdi_W512_l2.0_wait@16384`: **Chưa kết luận được (khoảng tin cậy quá rộng so với δ)**. Target đúng mà reference sai: 30 câu; ngược lại: 43 câu.
- `rpdi_legacy_W16_l1.5@768` so với `rpdi_W512_l2.0_newline@16384`: **Kém hơn, chưa chứng minh được nằm trong biên δ (cận dưới KTC 95% ≤ −δ)**. Target đúng mà reference sai: 27 câu; ngược lại: 45 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `standard@16384`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 32 câu; ngược lại: 28 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `rpdi_W512_l2.0_wait@16384`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 36 câu; ngược lại: 27 câu.
- `rpdi_legacy_W16_l1.5@1024` so với `rpdi_W512_l2.0_newline@16384`: **Không kém hơn quá δ (cận dưới KTC 95% > −δ) nhưng chưa đủ để kết luận tương đương**. Target đúng mà reference sai: 32 câu; ngược lại: 28 câu.

- *Phục hồi* = (Acc T − Acc baseline ở ngân sách của T) / (Acc R − Acc baseline ở ngân sách của T).
- *Cần ngân sách* = ngân sách nhỏ nhất để `standard` đạt accuracy bằng target trên cùng tập câu.

