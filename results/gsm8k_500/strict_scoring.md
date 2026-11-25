# Độ nhạy với cách chấm (n = 500)

| Phương pháp | Ngân sách | lenient | closed | finished |
| :--- | ---: | ---: | ---: | ---: |
| `standard` | 512 | 46.4 | 35.0 | 33.2 |
| `standard` | 768 | 54.6 | 47.0 | 45.4 |
| `standard` | 1024 | 64.0 | 60.2 | 59.2 |
| `standard` | 1536 | 74.8 | 71.6 | 71.6 |
| `standard` | 2048 | 77.6 | 74.8 | 74.8 |
| `standard` | 3072 | 79.0 | 75.6 | 75.6 |
| `standard` | 4096 | 78.8 | 76.0 | 76.0 |
| `standard` | 8192 | 78.4 | 76.0 | 76.0 |
| `standard` | 16384 | 79.2 | 76.0 | 76.0 |
| `rpdi_legacy_W16_l1.5` | 512 | 54.0 | 53.8 | 40.4 |
| `rpdi_legacy_W16_l1.5` | 768 | 75.6 | 75.6 | 68.0 |
| `rpdi_legacy_W16_l1.5` | 1024 | 80.0 | 80.0 | 77.0 |
| `rpdi_legacy_W16_l1.5` | 1536 | 81.4 | 81.4 | 80.2 |
| `rpdi_legacy_W16_l1.5` | 2048 | 81.2 | 81.2 | 80.4 |
| `rpdi_legacy_W16_l1.5` | 3072 | 81.0 | 81.0 | 80.4 |
| `rpdi_legacy_W16_l1.5` | 4096 | 81.2 | 81.2 | 80.4 |
| `rpdi_legacy_W16_l1.5` | 8192 | 81.0 | 81.0 | 80.4 |
| `rpdi_legacy_W16_l1.5` | 16384 | 81.0 | 81.0 | 80.4 |

| So sánh | Cách chấm | Acc T | Acc R | Δ | KTC 95% (bootstrap) | McNemar p |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: |
| `rpdi_legacy_W16_l1.5@768` vs `standard@768` | lenient | 75.6 | 54.6 | +21.0 | [+16.8; +25.2] | 3.9e-21 |
| `rpdi_legacy_W16_l1.5@768` vs `standard@768` | closed | 75.6 | 47.0 | +28.6 | [+24.4; +32.8] | 5.03e-34 |
| `rpdi_legacy_W16_l1.5@768` vs `standard@768` | finished | 68.0 | 45.4 | +22.6 | [+18.6; +26.6] | 1.87e-26 |
| `rpdi_legacy_W16_l1.5@1024` vs `standard@16384` | lenient | 80.0 | 79.2 | +0.8 | [-2.2; +3.8] | 0.699 |
| `rpdi_legacy_W16_l1.5@1024` vs `standard@16384` | closed | 80.0 | 76.0 | +4.0 | [+0.8; +7.2] | 0.0205 |
| `rpdi_legacy_W16_l1.5@1024` vs `standard@16384` | finished | 77.0 | 76.0 | +1.0 | [-2.4; +4.2] | 0.63 |
| `rpdi_legacy_W16_l1.5@16384` vs `standard@16384` | lenient | 81.0 | 79.2 | +1.8 | [-1.2; +4.6] | 0.272 |
| `rpdi_legacy_W16_l1.5@16384` vs `standard@16384` | closed | 81.0 | 76.0 | +5.0 | [+2.0; +8.0] | 0.00187 |
| `rpdi_legacy_W16_l1.5@16384` vs `standard@16384` | finished | 80.4 | 76.0 | +4.4 | [+1.4; +7.4] | 0.00715 |
