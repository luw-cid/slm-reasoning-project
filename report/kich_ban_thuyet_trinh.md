# Kịch bản thuyết trình: Phanh suy luận cho mô hình ngôn ngữ nhỏ dưới ngân sách token hẹp

Dùng kèm slide `Phanh suy luận cho mô hình ngôn ngữ nhỏ dưới ngân sách token hẹp.pptx` (19 slide).

- **Tổng thời gian dự kiến:** khoảng 20 phút trình bày, cộng demo và hỏi đáp.
- **Ký hiệu:** `[...]` là hành động trên sân khấu (chỉ vào slide, chuyển slide, dừng lại). **In đậm** là con số hoặc ý cần nhấn giọng.
- **Trước giờ báo cáo, sửa 2 chỗ còn để trống trên slide:** slide 1 `[Khoa / Trường]`, và slide 14 `[cập nhật bước hiện tại]` (có thể xoá dòng này nếu không muốn nêu số bước).

| Phần | Slide | Thời gian |
|---|---|---|
| Mở đầu, vấn đề, câu hỏi nghiên cứu | 1–4 | 4 phút |
| Phương pháp 1: RPDI-EE | 5–7 | 3,5 phút |
| Kết quả phương pháp 1 | 8–12 | 6,5 phút |
| Phương pháp 2: RLVR | 13–15 | 3 phút |
| Đối chiếu, kết luận, hạn chế | 16–18 | 3 phút |
| Demo và hỏi đáp | 19 | tuỳ thầy |

---

## Slide 1 · Trang bìa (≈ 30 giây)

> Em chào thầy ạ. Em là Phạm Tiến Lực, cùng bạn Nguyễn Hoàng Phúc. Hôm nay tụi em xin báo cáo đồ án **"Phanh suy luận cho mô hình ngôn ngữ nhỏ dưới ngân sách token hẹp"**.
>
> Nói ngắn gọn, đề tài hỏi: khi một mô hình suy luận nhỏ, cỡ **1,5 tỷ tham số**, chỉ được phép sinh **768 token**, thì làm sao để nó trả lời đúng nhiều hơn?
>
> Tụi em khảo sát hai hướng. Hướng thứ nhất là **RPDI-EE**, can thiệp lúc sinh, phần này đã có kết quả đầy đủ. Hướng thứ hai là **RLVR với GRPO + LoRA**, tức huấn luyện mô hình, phần này đã xong pipeline và đang huấn luyện.

`[Chuyển slide]`

---

## Slide 2 · Vấn đề (≈ 1 phút 15)

> Trước hết là vấn đề. Các mô hình suy luận như DeepSeek-R1 sẽ "nghĩ" trước trong một khối nằm giữa thẻ `<think>` và `</think>`, rồi mới viết lời giải.
>
> Với mô hình lớn, nghĩ dài thường có lợi. Nhưng mô hình 1,5B hay bị lặp, tự nghi ngờ, kiểm tra đi kiểm tra lại.

`[Chỉ vào ba con số lớn]`

> Khi giới hạn 768 token, mô phỏng việc chạy trên thiết bị biên, có **47,1%** số câu, tức **332 trên 705 câu**, dùng hết ngân sách mà vẫn chưa đóng thẻ `</think>`. Những câu này không kịp viết đáp số, nên bị chấm sai, dù có thể mô hình đã tính đúng trong lúc nghĩ.
>
> Kết quả là độ chính xác chỉ **53,2%**.

`[Chỉ vào thanh ngang phía dưới]`

> Thanh này cho thấy trung bình một câu dùng ngân sách ra sao: **464 token để nghĩ**, chỉ **148 token để trả lời**. Phần "còn dư" là do lấy trung bình, các câu ngắn không dùng hết ngân sách.
>
> Nói cách khác, **mô hình thua vì nghĩ quá dài, chứ không hẳn vì nghĩ sai.**

`[Chuyển slide]`

---

## Slide 3 · Câu hỏi nghiên cứu (≈ 1 phút)

> Từ vấn đề đó, đề tài đặt ra ba câu hỏi.
>
> **RQ1:** RPDI-EE là phương pháp phanh pha suy nghĩ dựa trên entropy. Bài gốc thử ở **16 nghìn token**. Vậy khi chỉ còn **768 token**, nó có còn hiệu quả không, và siêu tham số có giữ nguyên không?
>
> **RQ2:** Nếu có hiệu quả, thì lợi ích đến từ **tín hiệu entropy**, hay chỉ đơn giản là vì **dừng suy nghĩ sớm**? Bài gốc không có đối chứng nào để trả lời câu này.
>
> **RQ3:** Huấn luyện RLVR thay đổi tín hiệu entropy như thế nào, và sau RLVR thì phanh entropy còn giúp được gì?
>
> RQ1 và RQ2 đã có câu trả lời. RQ3 đang làm.

`[Dừng một nhịp, nói chậm]`

> Em xin nói rõ ngay từ đầu: cả hai phương pháp đều **không phải do tụi em đề xuất**. Điểm mới của đề tài nằm ở **chế độ ngân sách hẹp**, ở **phân tích cơ chế**, và ở **tương tác giữa hai phương pháp**.

`[Chuyển slide]`

---

## Slide 4 · Hai phương pháp (≈ 1 phút)

> Hai phương pháp này đại diện cho hai trường phái.

`[Chỉ cột trái]`

> **RPDI-EE** can thiệp **lúc sinh**, không cần huấn luyện. Nó theo dõi entropy của từng token trong pha suy nghĩ. Khi độ bất định cục bộ tăng vọt so với mức chung, nó chèn `</think>` để ép mô hình chuyển sang trả lời.

`[Chỉ cột phải]`

> **RLVR** thì thay đổi chính mô hình, bằng học tăng cường với phần thưởng kiểm chứng tự động: đúng đáp số và đúng định dạng thì được thưởng, còn lời giải bị cắt trước `</think>` thì nhận **0 điểm**. Tụi em huấn luyện trên **một GPU T4**, dùng LoRA hạng 16.

`[Chỉ khung 4 kịch bản]`

> Cuối cùng, hai phương pháp được đặt chung trong khung **4 kịch bản** trên cùng một tập câu hỏi: mô hình gốc, gốc cộng RPDI-EE, mô hình RLVR, và RLVR cộng RPDI-EE. Mục tiêu là xem hai cách này **bổ trợ hay thay thế** nhau.

`[Chuyển slide]`

---

## Slide 5 · RPDI-EE hoạt động thế nào (≈ 1 phút 15)

> Cụ thể RPDI-EE hoạt động như sau.

`[Chỉ lần lượt từng công thức]`

> Tại mỗi bước t, tính **H_t** là entropy của phân phối xác suất token, tức mô hình đang phân vân đến mức nào.
>
> **LTF** là entropy trung bình của **W token gần nhất**, tức mức phân vân cục bộ.
>
> **GTF** là entropy trung bình **từ đầu pha suy nghĩ**, tức mức phân vân chung.
>
> **RPDI** là tỷ số LTF chia GTF. Khi tỷ số này vượt ngưỡng **λ**, nghĩa là mô hình đột nhiên phân vân hơn nhiều so với bình thường, dấu hiệu nó đang lạc hướng hoặc luẩn quẩn. Lúc đó, tại **biên câu** gần nhất, hệ thống chèn `</think>`.

`[Chỉ khung cấu hình]`

> Bài gốc dùng **W = 512, λ = 2,0** ở 16 nghìn token. Nhưng ở 768 token, cửa sổ 512 gần như không bao giờ kích hoạt. Tụi em quét lưới trên 100 câu riêng, và cấu hình tốt nhất dịch về **W = 16, λ = 1,5**.

`[Chuyển slide]`

---

## Slide 6 · Hiện thực (≈ 1 phút 15)

> Về hiện thực, có ba điểm kỹ thuật.
>
> **Thứ nhất**, phanh được viết thành một **LogitsProcessor**, tính entropy ngay trên GPU, nên chạy trong **một lệnh generate duy nhất**, giữ nguyên KV-cache, không phải sinh lại từ đầu.
>
> **Thứ hai**, vì giải mã greedy là tất định, phần token trước điểm phanh của mọi cấu hình đều trùng với baseline. Nên em chỉ cần ghi vết entropy **một lần**, rồi mô phỏng điểm phanh của 16 cấu hình trên CPU. Cách này tiết kiệm **hơn 85% thời gian GPU**. Điểm phanh mô phỏng khớp với chạy thật ở **93/100 câu**.
>
> **Thứ ba**, trong quá trình làm em phát hiện và sửa **hai lỗi**. Một là rào `min_steps = 64` làm các ca phanh dồn ứ ngay sau bước 64, em gọi là "kẹt đập nhân tạo". Hai là bộ phanh vẫn hoạt động sau khi mô hình đã tự đóng `</think>`, làm tỷ lệ phanh bị thổi phồng từ 71,5% lên 80,1%. Lỗi thứ hai không làm đổi độ chính xác, nhưng đã được sửa để số liệu cơ chế đúng.

`[Chuyển slide]`

---

## Slide 7 · Thiết lập thực nghiệm (≈ 1 phút)

> Thiết lập thực nghiệm: mô hình **DeepSeek-R1-Distill-Qwen-1.5B**, giải mã **greedy** để kết quả tất định và tái lập được, ngân sách **768 token**, dữ liệu là **705 câu đầu của GSM8K test**.

`[Chỉ thanh chia id]`

> Điểm em muốn nhấn mạnh là cách chia dữ liệu.
>
> - **100 câu đầu, id 0 đến 99**, chỉ dùng để **chọn W và λ**.
> - **Kết quả chính** báo cáo trên **605 câu còn lại**, id 100 đến 704, không tham gia chọn tham số.
> - **Thí nghiệm đối chứng** chạy trên **200 câu id 100 đến 299**, nằm trong phần held-out.
>
> Mọi so sánh cặp dùng **kiểm định McNemar chính xác**, khoảng tin cậy dùng Wilson 95%.

`[Chuyển slide]`

---

## Slide 8 · Kết quả 1: Quét lưới (≈ 1 phút 15)

> Đây là kết quả quét 16 cấu hình trên 100 câu dùng để chọn tham số.

`[Chỉ ô góc trên trái]`

> Cấu hình tốt nhất là **W = 16, λ = 1,5**, đạt **78%** trong mô phỏng, so với **59%** của baseline. Khi W lớn hoặc λ cao thì hầu như không phanh, nên kết quả trùng baseline, đó là các ô 59 ở góc dưới.

`[Chỉ khung đối chiếu bên phải, nói chậm]`

> Có hai điểm em muốn nói thẳng.
>
> **Một**, cấu hình tốt nhất nằm ở **góc của lưới**, nên cần mở rộng lưới mới chắc chắn đó là tối ưu thật.
>
> **Hai**, **cắt cứng ở 128 token đạt 75%**, chỉ kém 3 câu. Thậm chí cắt ở 32 token cũng hơn baseline 10 điểm. Điều này gợi ý rằng phần lớn lợi ích **có thể chỉ đến từ việc dừng sớm**. Em sẽ kiểm tra kỹ nghi vấn này ở slide đối chứng.
>
> Chạy thật cấu hình này trên cùng 100 câu được **73%**, chênh với mô phỏng vì pha trả lời được sinh khác nhau.

`[Chuyển slide]`

---

## Slide 9 · Kết quả 2: 605 câu held-out (≈ 1 phút 15)

> Đây là **kết quả chính** của phương pháp 1.

`[Chỉ hai con số lớn]`

> Trên **605 câu không dùng để chọn tham số**, độ chính xác tăng từ **52,23% lên 71,57%**, tức **+19,34 điểm**. Khoảng tin cậy 95% từ **15,5 đến 23,2** điểm. Kiểm định McNemar cho **p khoảng 2,4 nhân 10 mũ trừ 21**.
>
> Cụ thể, RPDI-EE **cứu được 141 câu** baseline làm sai, và **làm hỏng 24 câu** baseline làm đúng.
>
> Token suy nghĩ trung bình giảm từ **465 xuống 134**, tức giảm khoảng 71%. Tỷ lệ cạn ngân sách giảm hơn một nửa, từ **47,6% xuống 22,6%**.
>
> Trên toàn bộ 705 câu, kết quả gần như y hệt: 53,19% lên 71,77%.
>
> Vậy **RQ1 đã có câu trả lời: có, RPDI-EE hiệu quả ở 768 token**, với điều kiện chỉnh lại siêu tham số.

`[Chuyển slide]`

---

## Slide 10 · Kết quả 3: Cơ chế (≈ 1 phút 15)

> Câu hỏi tiếp theo là: **19 điểm này đến từ đâu?**
>
> Em chia 605 câu thành hai nhóm, theo việc baseline có cạn ngân sách hay không.

`[Chỉ cặp cột bên trái]`

> Ở **317 câu baseline tự đóng `</think>`**, RPDI-EE **không cải thiện**, thậm chí giảm nhẹ 1,3 điểm: 80,4 xuống 79,2.

`[Chỉ cặp cột bên phải]`

> Ở **288 câu baseline cạn ngân sách**, độ chính xác tăng từ **21% lên 63%**.
>
> Vậy ở ngân sách hẹp, **RPDI-EE hoạt động như một cơ chế phân bổ lại ngân sách**: nó chuyển token từ pha suy nghĩ sang pha trả lời, chứ không sửa mạch suy luận sai.
>
> Đây là khác biệt lớn với bài gốc ở 16 nghìn token. Ở đó, trên nhóm Math Easy, mô hình 1,5B chỉ dùng trung bình khoảng 5,6 nghìn token, còn xa trần ngân sách, và mức tăng chỉ khoảng 3,7 điểm.

`[Chuyển slide]`

---

## Slide 11 · Kết quả 4: Đối chứng, RQ2 (≈ 1 phút 30)

> Đây là thí nghiệm em thấy **quan trọng nhất về mặt khoa học**, và bài gốc không có.
>
> Nếu lợi ích chỉ đến từ việc dừng sớm, thì một cách dừng "ngây thơ" cũng phải đạt kết quả tương tự. Em chạy thêm hai đối chứng trên 200 câu held-out:
>
> - **Cắt cứng:** ép đóng `</think>` đúng sau **130 token**, bằng số token suy nghĩ trung bình của RPDI-EE.
> - **Phanh ngẫu nhiên:** phanh tại biên câu, sau một số bước lấy ngẫu nhiên từ phân phối điểm phanh thật của RPDI-EE. Tức là giữ mọi thứ, **chỉ bỏ tín hiệu entropy**.

`[Chỉ lần lượt bốn thanh]`

> Kết quả: cả ba cách phanh đều vượt baseline 52,5% rất rõ. Nhưng **cắt cứng đạt 72,0%, ngang RPDI-EE 71,5%**, p = 1,0. RPDI-EE hơn phanh ngẫu nhiên 3 điểm, nhưng p = 0,38, không có ý nghĩa thống kê.

`[Dừng, nói chậm]`

> Kết luận trung thực của em: **ở 768 token, chưa thấy tín hiệu entropy tạo thêm khác biệt có ý nghĩa** so với cắt cố định ở cùng mức token. Với 200 câu, nếu RPDI-EE có lợi thế thật thì lợi thế đó **không quá khoảng 5 đến 6 điểm**.

`[Chuyển slide]`

---

## Slide 12 · Kết quả 5: Quan sát thêm (≈ 1 phút 15)

> Dù tổng thể hòa nhau, hành vi của RPDI-EE vẫn khác cắt cứng ở bốn điểm.
>
> **Một, ít làm hỏng câu đúng hơn.** Trong 99 câu baseline tự kết thúc, RPDI-EE làm hỏng **0 câu**, cắt cứng làm hỏng 4, phanh ngẫu nhiên làm hỏng 5. Lý do là RPDI-EE không phanh phần lớn các câu này, còn cắt cứng thì cắt mọi câu dài hơn 130 token. Đổi lại, RPDI-EE cứu ít hơn 2 câu ở nhóm cạn ngân sách, nên tổng thể hòa. Số nhỏ, nên em chỉ coi đây là quan sát.
>
> **Hai, điểm phanh thích nghi theo từng câu**, trải từ khoảng 60 đến 239 token. 58 trên 59 câu không bị phanh là câu baseline tự kết thúc.
>
> **Ba, cạn ngân sách chuyển sang pha trả lời.** Sau khi bị ép đóng `</think>`, mô hình 1,5B tiếp tục suy luận ngay trong lời giải, nên khoảng **27% câu vẫn cạn ngân sách**. Đây là giới hạn chung của mọi cách phanh, và là hướng cải tiến.
>
> **Bốn, tái lập trên hai phần cứng.** GPU bf16 và T4 fp16 cho kết quả đúng sai trùng **200 trên 200 câu**.

`[Chuyển slide]`

`[Nếu có hai người trình bày, có thể đổi người tại đây]`

---

## Slide 13 · Phương pháp 2: RLVR (≈ 1 phút 15)

> Tiếp theo là phương pháp 2, huấn luyện RLVR.
>
> Tụi em dùng **GRPO**: với mỗi câu hỏi, mô hình sinh **4 lời giải**, phần thưởng được chuẩn hóa trong nhóm để biết lời giải nào tốt hơn trung bình, nên không cần mô hình critic. Chỉ huấn luyện **LoRA** để vừa GPU T4.
>
> Dữ liệu là **400 câu GSM8K train**, đã loại trùng với tập test và validation. Ngân sách huấn luyện là **768 token**, giống lúc đánh giá.
>
> Phần thưởng gồm: **đúng đáp số được 2 điểm**, nhưng chỉ khi có `</think>`; **đúng định dạng được 1 điểm**, có `\boxed` thêm 0,25.

`[Chỉ khung câu hỏi bên dưới]`

> Vì sao lời giải bị cắt lại nhận 0 điểm? Vì nếu không, bộ chấm sẽ lấy con số cuối trong phần nháp và vẫn có thể thưởng, nên mô hình không bị phạt khi nghĩ vượt ngân sách. Cách làm này giống ThinkPrune.
>
> Điểm riêng của đề tài là **probe RPDI**: cứ mỗi 50 bước, sinh greedy trên 8 câu validation cố định, ghi lại entropy pha suy nghĩ, số token suy nghĩ, và tỷ lệ RPDI-EE sẽ phanh. Nhờ vậy em thấy được **RLVR thay đổi tín hiệu entropy thế nào theo thời gian**.

`[Chuyển slide]`

---

## Slide 14 · Tiến độ RLVR (≈ 1 phút)

> Pipeline huấn luyện đã chạy được từ đầu đến cuối trên Colab T4.

`[Chỉ bảng bên trái]`

> Lần chạy thử đầu tiên ở **512 token** cho thấy **95% lời giải bị cắt**, phần thưởng gần như toàn 0, và 60% nhóm có phần thưởng bằng nhau. GRPO học từ **chênh lệch** phần thưởng trong nhóm, nên khi mọi lời giải đều 0 điểm thì **gradient bằng 0**, không học được gì. Vì vậy em chuyển ngân sách huấn luyện lên **768 token**, bằng ngân sách đánh giá, và tăng learning rate lên 1e-5.

`[Chỉ khung giữa]`

> Một phát hiện phụ: trên cùng 4 câu, bản **4-bit chỉ nghĩ 266 token**, còn bản **fp16 nghĩ 582 token**. Lượng tử hóa thay đổi hành vi đáng kể, nên em huấn luyện và đo đều bằng fp16.

`[Chỉ khung phải]`

> Ngoài ra có vài lỗi môi trường trên T4, ví dụ T4 báo hỗ trợ bf16 nhưng thực ra không có nhân bf16. Các lỗi này đã được xử lý.
>
> Bản huấn luyện đầy đủ 400 bước, khoảng 8 giờ trên T4, đang chạy.

`[Chuyển slide]`

---

## Slide 15 · RQ3: Sau RLVR, phanh còn giúp gì? (≈ 1 phút)

> Câu hỏi nghiên cứu cho phần 2 là **tương tác giữa hai phương pháp**. GRPO trên mô hình 1,5B đã có nhiều công trình như DeepScaleR, Tina, ThinkPrune, nên bản thân việc huấn luyện không mới.
>
> Tụi em đặt hai giả thuyết.
>
> **H1:** RLVR thường làm entropy giảm, hiện tượng gọi là *entropy collapse*. Nhưng vì RPDI là **tỷ số** giữa entropy cục bộ và toàn cục, khi cả tử và mẫu cùng giảm thì RPDI có thể thay đổi theo hướng khó đoán. Nên cần đo trực tiếp, và đó là lý do có probe.
>
> **H2:** Mô hình RLVR tự đóng `</think>` sớm hơn, nhóm câu cạn ngân sách thu nhỏ, nên lợi ích của mọi cách phanh sẽ giảm.

`[Chỉ khung kế hoạch]`

> Sau khi huấn luyện xong, em đánh giá **5 kịch bản** trên cùng 200 câu, có thêm kịch bản **RLVR cộng cắt cứng** để lặp lại phép đối chứng quan trọng nhất của phần 1.

`[Chuyển slide]`

---

## Slide 16 · Đối chiếu với bài gốc (≈ 1 phút)

> Em muốn nói rõ vị trí của đề tài so với bài gốc của Guan và cộng sự.
>
> Bài gốc **đã thử RPDI-EE trên chính mô hình 1,5B này**, cũng dùng greedy, nhưng ở **16 nghìn token**, và tăng khoảng **3,7 điểm** trên nhóm Math Easy.
>
> Đề tài làm ở **768 token**, nơi gần một nửa số câu cạn ngân sách. Nên mức tăng **19 điểm phản ánh hiệu ứng giải cứu**, không phải cải thiện chất lượng suy luận.

`[Chỉ dòng chú thích cuối slide]`

> Vì vậy, **không thể nói RPDI-EE ở đây "tốt hơn gấp 5 lần"**. Hai con số đo hai hiệu ứng khác nhau. Đóng góp của đề tài là chế độ ngân sách hẹp, sự dịch chuyển siêu tham số, phân tích cơ chế, và các đối chứng mà bài gốc không có.

`[Chuyển slide]`

---

## Slide 17 · Kết luận (≈ 1 phút)

> Tóm lại, thông điệp chính là: **ở ngân sách hẹp, nút thắt của mô hình nhỏ là ngân sách, không phải tín hiệu dừng.**
>
> **Một**, RPDI-EE tăng **19,3 điểm** trên 605 câu held-out, nhưng siêu tham số phải dịch về W = 16, λ = 1,5.
>
> **Hai**, mức tăng đến từ việc **giải cứu các câu cạn ngân sách**, tức là một cơ chế phân bổ lại token.
>
> **Ba**, và đây là kết quả em thấy quan trọng nhất: **cắt cứng 130 token đạt ngang RPDI-EE**, nên ở chế độ này chưa thể gán lợi ích cho tín hiệu entropy.
>
> **Bốn**, pipeline RLVR có probe entropy đã chạy được trên một GPU T4, và câu hỏi tương tác giữa hai phương pháp đang được đo.

`[Chuyển slide]`

---

## Slide 18 · Hạn chế và hướng tiếp theo (≈ 1 phút)

> Về hạn chế, em xin nêu thẳng:
>
> - Với 200 câu, chỉ phát hiện được chênh lệch từ khoảng **6 điểm** trở lên, nên nếu RPDI-EE có lợi thế nhỏ thì sẽ không thấy.
> - **N = 130 được chọn theo số token của RPDI-EE**, tức là có lợi thế "biết trước". Trong thực tế phải chọn N trước khi chạy.
> - Kết quả mới ở **một mô hình, một tập dữ liệu, một ngân sách**, và dùng greedy, khác khuyến nghị nhiệt độ 0,6 của DeepSeek.
> - Điểm tối ưu W, λ nằm ở **góc lưới** đã quét.
>
> Hướng tiếp theo:
>
> - Hoàn tất huấn luyện RLVR và đánh giá 5 kịch bản.
> - Thử ngân sách **2048 token**, nơi N cố định có thể cắt quá sớm, và lợi thế thích nghi của RPDI-EE có thể lộ ra.
> - Chèn **"Final Answer:"** sau `</think>` để giảm 27% câu vẫn cạn ngân sách trong pha trả lời.
> - Mở rộng đối chứng lên đủ 605 câu.
> - Hoàn thiện các chương luận văn.

`[Chuyển slide]`

---

## Slide 19 · Demo và hỏi đáp

> Em xin cảm ơn thầy đã lắng nghe. Tụi em có chuẩn bị ba demo.
>
> **Demo 1** là trang mô phỏng RPDI-EE: chọn một câu, chỉnh W và λ, sẽ thấy ngay đường entropy, đường RPDI, và vị trí phanh trên vết sinh thật của mô hình.
>
> **Demo 2** so sánh đầu ra của 4 phương pháp trên từng câu trong 200 câu đối chứng, có lọc các câu RPDI-EE cứu được hoặc cắt cứng làm hỏng.
>
> **Demo 3** là chạy trực tiếp trên Colab, nếu thầy muốn thử một bài toán mới.
>
> Thầy muốn xem demo nào trước, hay tụi em xin nhận câu hỏi trước ạ?

**Gợi ý khi demo:**

- Demo 1: chọn một câu baseline cạn ngân sách nhưng RPDI-EE làm đúng, chỉnh λ từ 1,5 lên 2,5 để thầy thấy điểm phanh biến mất và câu trả lời bị cắt.
- Demo 2: lọc "cắt cứng làm hỏng, RPDI-EE đúng" để minh hoạ quan sát ở slide 12.
- Demo 3: mở Colab và tải mô hình **trước giờ báo cáo**, vì tải trọng số mất vài phút. Sinh 768 token trên T4 mất khoảng 40 đến 60 giây mỗi câu.

---

## Phụ lục: Câu hỏi thầy có thể hỏi

**1. Nếu cắt cứng ngang RPDI-EE thì đóng góp của đề tài là gì?**

> Đóng góp chính chính là phát hiện đó. Bài gốc không có đối chứng cắt cứng, nên không phân biệt được lợi ích đến từ entropy hay từ dừng sớm. Đề tài cho thấy ở ngân sách hẹp, phần lớn lợi ích đến từ việc phân bổ lại ngân sách. Ngoài ra còn có sự dịch chuyển siêu tham số, phân tích cơ chế theo nhóm cạn ngân sách, và pipeline RLVR có probe entropy.

**2. Vì sao chọn 768 token?**

> Để mô phỏng triển khai trên thiết bị biên, nơi độ trễ và bộ nhớ giới hạn độ dài sinh. Ở mức này, gần một nửa số câu cạn ngân sách, nên vấn đề "nghĩ quá dài" thể hiện rõ nhất. Bài gốc đã làm ở 16 nghìn token, nên 768 là chế độ bổ sung.

**3. N = 130 chọn theo RPDI-EE, như vậy có công bằng không?**

> Thầy nói đúng, đây là hạn chế em đã nêu. N = 130 được chọn để so sánh ở cùng mức token, nên cắt cứng có lợi thế biết trước. Nhưng ở quét lưới trên 100 câu riêng, cắt cứng ở 128 và 256 token cũng đạt 75% và 73%, nên kết luận không quá nhạy với N. Ở ngân sách lớn hơn, chọn N trước sẽ khó hơn, và đó là lý do em đề xuất thử 2048 token.

**4. Vì sao dùng greedy mà không dùng nhiệt độ 0,6 như DeepSeek khuyến nghị?**

> Greedy cho kết quả tất định, nên so sánh cặp McNemar giữa các phương pháp sạch hơn, và cho phép mô phỏng offline để quét lưới. Bài gốc cũng dùng greedy. Đánh giá với sampling là hướng mở rộng.

**5. Mô phỏng offline có đáng tin không?**

> Điểm phanh mô phỏng khớp chạy thật ở 93/100 câu. Độ chính xác mô phỏng 78% so với chạy thật 73% trên cùng 100 câu, chênh do pha trả lời. Vì vậy mô phỏng chỉ dùng để chọn tham số, còn mọi kết quả báo cáo đều từ chạy thật.

**6. 200 câu có đủ không?**

> Đủ để thấy cả ba cách phanh vượt baseline rõ ràng, nhưng chỉ phát hiện được chênh lệch từ khoảng 6 điểm. Nên kết luận đúng là "chưa thấy khác biệt có ý nghĩa", không phải "chắc chắn không có khác biệt". Mở rộng lên 605 câu nằm trong kế hoạch.

**7. Vì sao RPDI-EE giảm 1,3 điểm ở nhóm baseline tự kết thúc?**

> Khả năng cao là ở một số câu, RPDI-EE phanh khi mô hình đang tính dở, nên mô hình mất bước suy luận cần thiết. Em chưa phân tích riêng từng câu này. Mức giảm nhỏ và không có ý nghĩa thống kê, nhưng cho thấy phanh không phải lúc nào cũng vô hại.

**8. Phần RLVR khác gì Tina hay ThinkPrune?**

> Bản thân GRPO + LoRA trên 1,5B không mới. Điểm em khảo sát là tương tác: RLVR làm thay đổi tín hiệu entropy mà RPDI-EE dựa vào như thế nào, đo trực tiếp bằng probe trong lúc huấn luyện, và sau RLVR thì phanh hay cắt cứng còn giúp gì.

**9. Vì sao không full fine-tune?**

> Chỉ có một GPU T4 16GB trên Colab. LoRA hạng 16 vừa bộ nhớ, và Tina đã cho thấy LoRA đủ để GRPO cải thiện mô hình 1,5B.

**10. Chèn "Final Answer:" giúp gì?**

> Sau khi bị phanh, mô hình 1,5B vẫn tiếp tục suy luận trong pha trả lời, nên khoảng 27% câu vẫn cạn ngân sách. Chèn "Final Answer:" ngay sau `</think>`, như budget forcing của s1, sẽ buộc mô hình viết đáp số ngay.
