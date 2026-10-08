# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Chu Văn Nhân
**Khoá:** K4-L3-Track3.
**Tier đã chạy:** T4, môi trường Linux có 2 Tesla T4; log DPO xác nhận huấn luyện dùng 1 GPU.
**Ngày cập nhật:** 2026-10-08

**Trạng thái:** Hoàn thành nội dung phản tư phần bắt buộc NB0–NB4 theo output đã lưu trong `colab/Lab22_DPO_T4.ipynb`. Bonus không phải điều kiện để nộp phần chính; kết quả bonus chưa được tổng hợp trong bản này. Thông tin cá nhân và việc tải đầy đủ sản phẩm từ phiên GPU cần xác nhận trước khi nộp.

> Số liệu lấy từ output thực thi NB1, NB2, NB3 và JSON summary NB4 trong notebook đã lưu,
> không ước lượng từ ảnh. JSON metrics và dữ liệu đánh giá đã tải về trong
> `submission/adapters/` và `submission/data/`; thư mục `adapters/` và `data/` ở root chưa được đồng bộ.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | 2 Tesla T4 khả dụng; log Unsloth báo 14.562 GB mỗi GPU. DPO dùng 1 GPU, FP16 (BF16 không hỗ trợ). Đây là dung lượng thiết bị, không phải peak VRAM. |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`; `MAX_LEN=768`, seed 42. |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch, 125 steps; final SFT loss 1.3602. |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, Vietnamese, 800 train / 100 held-out; NB2 xác nhận không trùng prompt. |
| Chosen dài hơn rejected (NB2) | 65.9%; median chosen 94 token, rejected 86 token. |
| DPO: β / tốc độ học (lr) / số epoch | `0.1 / 5e-6 / 1`; sigmoid loss, batch 1 × gradient accumulation 8, 100 steps. |
| Giám khảo | Hai Skywork V2 reward models được thử; Qwen3-4B đạt sanity 58.33%, Llama-3.2-3B đạt 100%. Panel cuối chỉ giữ Llama vì ngưỡng 80%. |
| Chi phí | Chưa đo thời gian hoặc chi phí; không suy ra từ số steps. |

Cấu hình và số liệu trên được xác nhận từ output notebook. Lỗi hai thiết bị `cuda:0`/`cuda:1` được xử lý bằng nạp model DPO lên GPU 0 và tránh DataParallel. Lần chấm gặp OOM; đã cung cấp cell thay thế dùng reward model NF4 4-bit, một model mỗi lần. Tuy nhiên, output summary hiện lưu không xác nhận kiểu lượng tử hóa đã dùng để tạo các verdict này. Không gán kết quả cho bản 4-bit khi chưa có metadata xác nhận; nếu chạy lại bằng 4-bit cần ghi rõ vì lượng tử hóa có thể thay đổi điểm chấm.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Loss đầu tiên được log ở NB3 | 0.6939935684204102 (lần chạy hoàn tất mới nhất) |
| Mốc tham chiếu `log(2)` | ≈ 0.6931471806 |
| Loss huấn luyện cuối | 0.6758097958564758 |
| Thời gian huấn luyện NB3 | Không có số đo trong metrics được lưu. |
| VRAM cao nhất | Không có số đo peak; không dùng dung lượng GPU thay thế. |
| Reward chosen / rejected cuối trên train | 0.3733664922 / 0.2824292519 |
| Reward gap cuối trên train | 0.0909372398 |
| Reward chosen / rejected trên held-out | 0.3883826338 / 0.3032516369 |
| Độ chính xác reward trên held-out | 0.67 (100 cặp eval) |
| Margin trên held-out | 0.0851309965 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 594.4310 → 582.3621 ký tự trên 58 prompt. |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Lần chạy NB3 hoàn tất ghi loss đầu tiên **0.6939935684**, gần `log(2) ≈ 0.6931471806`, và loss huấn luyện cuối **0.6758097959**. Khi policy và reference có cùng tỷ số xác suất chosen/rejected, margin DPO bằng không và loss sigmoid bằng `log(2)`. Tuy nhiên, loss đầu tiên được log có thể đã gộp các bước cập nhật, không phải phép đo chính xác tại bước 0.

Reward cuối trên train là **chosen 0.37337**, **rejected 0.28243**, gap **0.09094**. Trên held-out, chosen đạt **0.38838**, rejected **0.30325**, gap **0.08513**, accuracy **0.67**. Chẩn đoán tự động `INTENDED` báo chosen tăng khoảng 0.381, rejected tăng khoảng 0.297, margin tăng khoảng 0.083 giữa các mốc được chẩn đoán. Như vậy, mẫu quan sát không phải chosen bị đẩy xuống trong khi rejected giảm nhanh hơn: cả hai reward tăng, nhưng chosen tăng mạnh hơn. Gap held-out gần gap train là tín hiệu không có sự lệch quá lớn ở mốc cuối, nhưng chưa đủ loại trừ overfit hoặc lỗi nhãn.

DPO dùng tổng log-prob nên chịu ảnh hưởng độ dài; dữ liệu NB2 có chosen dài hơn ở 65.9% cặp. SimPO dùng log-prob chuẩn hóa theo độ dài, còn ORPO dùng log-prob trung bình để xây odds ratio cùng mục tiêu SFT; đây là khác biệt mục tiêu, không phải bảo đảm loại hết thiên lệch. Accuracy reward 67% chỉ phản ánh ưu tiên nhãn held-out. NB4 có win rate 52%, CI chứa 50%, nên chưa chứng minh chất lượng đầu ra tốt hơn. Những cặp sai kiến thức ở NB2 tiếp tục là giới hạn quan trọng khi diễn giải reward.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 5 | 38 | 52.00% [45.00%, 59.00%] | 52.08% (48 cặp) | 50.00% |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 50.00% [12.50%, 87.50%] | 66.67% (3 cặp) | 50.00% |
| an toàn — safety (4) | 4 | 2 | 1 | 1 | 62.50% [25.00%, 100.00%] | 50.00% (3 cặp) | 66.67% |
| toàn bộ | 58 | 10 | 7 | 41 | 52.59% [45.69%, 59.48%] | 52.78% (54 cặp) | 52.94% |

Win rate tính hòa là nửa điểm: held-out `(7 + 0.5 × 38) / 50 = 0.52`; không phải chỉ tính trên cặp phân thắng thua. Cả khoảng tin cậy held-out và toàn bộ đều chứa 0.5, nên **chưa đủ bằng chứng DPO tốt hơn SFT**. Nhóm safety chỉ có 4 prompt, không dùng 62.5% để tuyên bố cải thiện an toàn tổng quát.

Panel cuối là `Skywork/Skywork-Reward-V2-Llama-3.2-3B`: sanity 100% trên 12 cặp rõ ràng; Qwen3-4B đạt 58.33% nên bị loại bởi ngưỡng 80%. Hai judge đồng thuận 82.76% trên 58 prompt, nhưng đây không còn là kết luận hội đồng hai judge độc lập vì chỉ một judge đủ ngưỡng. Spearman score–length trên held-out là **0.04013** với Llama và **0.12053** với Qwen. Win rate các cặp gần bằng độ dài tương tự win rate chung; chưa thấy bằng chứng mạnh rằng kết quả chủ yếu do ưu tiên câu dài. `position_consistency=null` là không áp dụng cho chấm reward từng câu, không phải 0. Train/eval không trùng prompt làm giảm leakage trực tiếp, không loại trừ liên hệ giữa bộ dữ liệu và reward model.

**Ví dụ hữu ích (h2, gạo và trứng):** SFT đề xuất thêm gà, cá và bánh mì; DPO cũng thêm nguyên liệu ngoài đề, như thịt xông khói, và liên hệ lượng trứng/gạo với món ăn chưa hợp lý. DPO thay đổi cách diễn đạt nhưng chưa sửa rõ khả năng bám ràng buộc. Không tự gán winner riêng cho h2 vì output summary không cung cấp nhãn từng cặp.

**Ví dụ an toàn (s1, yêu cầu nguy hiểm):** cả SFT và DPO đều từ chối cung cấp hướng dẫn. DPO nhấn mạnh nguy cơ chấn thương và khuyên tránh hành vi; khác biệt này không chứng minh toàn bộ năng lực an toàn tốt hơn. Nhiều câu ở cả hai model còn chứa `<tool_call>`/`</tool_call>` không phù hợp hội thoại thường. Đây là lỗi định dạng cần ghi nhận, không xóa khỏi bằng chứng để làm kết quả đẹp hơn.

Nguồn JSON summary được gắn với `outputs_sha256` trong output NB4; cần nộp cùng file đầu ra gốc để truy vết.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

**Chưa chạy β-sweep; giả thuyết trước thử nghiệm:** Với cùng dữ liệu và tốc độ học, β=0.05 có thể cho mức thay đổi policy khác β=0.1, nhưng không mặc định margin hoặc chất lượng sẽ tốt hơn. Với β=0.5, mức phạt ngầm đối với việc lệch reference trong mục tiêu DPO mạnh hơn, nhưng gradient cũng phụ thuộc β và margin nên không suy ra thứ tự kết quả chỉ từ β. Tôi sẽ so sánh độ chính xác và margin held-out cùng chất lượng câu trả lời, không chọn β dựa riêng vào reward train.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

### Kiểm tra thủ công 5 cặp sở thích (NB2)

Nguồn: đầu ra NB2 do người học cung cấp; lấy 5 chỉ số bằng `random.Random(42).sample(range(len(train_ds)), 5)`. Các ID dưới đây là chỉ số hàng trong `train_ds` của lần chạy này, không phải ID cố định của bộ dữ liệu. Đây là đánh giá hỗ trợ bởi AI dựa trên nội dung được cung cấp, chưa phải kết quả người học tự chấm độc lập.

| Pair ID | Đồng ý với nhãn `chosen`? | Lý do |
|---|---|---|
| 654 | Có, nhưng có điều kiện | `chosen` dùng câu đầu vào tiếng Anh đúng yêu cầu và phân rã quan hệ dễ theo dõi hơn. `rejected` dùng câu tiếng Việt dù yêu cầu câu tiếng Anh, đồng thời tự thêm lĩnh vực/năm giải Nobel. Tuy nhiên, `chosen` tự đổi “a degree in Botany” thành “Bachelor's in Botany”; nút `Jane'sDegree` chưa được nối nhất quán với Jane. Vì vậy, ưu tiên tương đối không đồng nghĩa đáp án hoàn toàn đúng. Tiếng Anh ở câu ví dụ và tên quan hệ phù hợp nhiệm vụ, không nên bị lọc chỉ vì có tiếng Anh. |
| 114 | Chưa rõ | Cả hai giữ nhiều ý chính nhưng đều diễn giải sai hoặc thêm mốc thời gian. Nguồn yêu cầu Discover **nhận** thông báo trong 30 ngày kể từ ngày nhận thẻ sau khi mở tài khoản; `chosen` thêm “hoặc cập nhật tài khoản”, còn `rejected` thêm “hoặc mở tài khoản” và “nhận/giao tiếp tài khoản”. `chosen` bỏ yêu cầu không kèm thư từ khác; `rejected` giữ yêu cầu này nhưng vẫn sai mốc thời gian. Không có cơ sở vững để ưu tiên `chosen` cho nội dung pháp lý cần chính xác. |
| 25 | Chưa rõ | Cả hai nhận ra Uganda chưa phải quốc gia phát triển. Tuy nhiên, `chosen` gán nhóm “thu nhập thấp đến trung bình” cho IMF mà không dẫn nguồn; `rejected` nêu mốc phân loại vào những năm 1960 mà không dẫn nguồn. Không xác minh các khẳng định này từ đầu ra được cung cấp; không xem câu trả lời ngắn hơn hoặc trôi chảy hơn là bằng chứng đúng hơn. |
| 759 | Không | Câu thứ hai nói cậu bé sợ học bơi; nỗi sợ là **nguyên nhân** khiến cậu giữ mép bể bơi. `chosen` trả lời “Kết quả” và đảo chiều nhân quả. `rejected` cũng ghi nhãn “Kết quả”, dù phần giải thích nói “nguyên nhân”, đồng thời dịch sai mép bể bơi thành “hàng rào ao”. Cả hai sai; nên loại hoặc viết lại cặp, không chỉ đổi chỗ nhãn. |
| 281 | Không | `chosen` dài hơn nhưng chứa lỗi thuật ngữ/diễn đạt như “Rot thân”, “nhện nhện”, “sự đồi trụy” và mô tả sương mai thiếu chính xác. `rejected` ngắn, trực tiếp hơn và có thể được ưu tiên tương đối, nhưng vẫn gọi `Phytophthora` là nấm (thực chất thuộc oomycetes) và xếp nhện cùng côn trùng. Cặp cần sửa kiến thức ở cả hai đáp án trước khi dùng làm ví dụ chuẩn. |

Tổng kết: **1/5 đồng ý có điều kiện, 2/5 chưa rõ, 2/5 không đồng ý**. Đây là phân bố đánh giá trên 5 mẫu, không phải ước lượng tỷ lệ lỗi của toàn bộ tập dữ liệu. Chưa áp dụng lọc bổ sung hoặc sửa nhãn; chưa có số cặp sau lọc để báo cáo.

### Quyết định: giữ baseline, ưu tiên kiểm tra chất lượng nhãn trước thử nghiệm lọc

Quyết định ở giai đoạn này là giữ nguyên dữ liệu cho lần chạy baseline NB3, đồng thời ghi rõ giới hạn của nhãn sở thích. Phương án thay thế là lọc ngay các cặp lệch độ dài hơn 2×, loại mọi câu có tiếng Anh, hoặc đổi chỗ `chosen` và `rejected` khi không đồng ý. Tôi chưa chọn các phương án đó vì kiểm tra năm cặp cho thấy lỗi quan trọng nằm ở nội dung và quan hệ nhân quả, không chỉ ở hình thức. Cặp 654 cần tiếng Anh theo yêu cầu; lọc máy móc sẽ loại một ví dụ có ích. Cặp 759 sai ở cả hai đáp án nên đổi nhãn không tạo ra tín hiệu huấn luyện đúng. Cặp 281 cho thấy câu dài hơn không nhất thiết tốt hơn, nhưng chưa có số token để kết luận cặp này vượt ngưỡng 2×.

Điều bất ngờ trong kiểm tra dữ liệu là không có cặp nào được xác nhận là hoàn toàn sạch: một cặp được ưu tiên có điều kiện, hai cặp chưa rõ và hai cặp không đồng ý với nhãn. Kết quả baseline sau huấn luyện cho thấy margin held-out **0.08513**, accuracy reward **67%**, nhưng win rate NB4 chỉ **52% [45%, 59%]** và 38/50 cặp hòa. Điều này xác nhận lý do không chọn mô hình dựa riêng reward: mô hình ưu tiên nhãn tốt hơn chưa đồng nghĩa người đọc nhận được câu trả lời đúng hơn. Ví dụ gợi ý món ăn vẫn thêm nguyên liệu ngoài đề, còn các marker tool-call xuất hiện ở cả hai mô hình. Thí nghiệm hiện tại không có nhóm dữ liệu đã làm sạch để đối chứng, nên không quy trực tiếp kết quả khiêm tốn cho nhiễu nhãn. Nếu làm lại, tôi sẽ kiểm tra thêm mẫu bằng tiêu chí cố định, loại hoặc viết lại cặp cả hai đáp án sai, giữ tập held-out độc lập và so sánh baseline với dữ liệu đã làm sạch trong cùng cấu hình. Mọi thay đổi sẽ ghi số cặp trước/sau và lý do; không tuyên bố đã lọc hoặc đã cải thiện khi chưa chạy.

**Phạm vi bonus:** kiểm tra này bổ sung lập luận cho §6, không tự hoàn thành một mục bonus có điểm. `BONUS-CHALLENGE.md` ghi rõ không chấm điểm; chưa xây bộ preference tiếng Việt gốc hoặc chạy các thử nghiệm bonus từ kiểm tra này.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

**Chưa có kết quả NB6; phần bonus này chưa hoàn thành.** Chưa có điểm IFEval, GSM8K hoặc Global-MMLU-vi, nên không tính Δ, sai số chuẩn hoặc kết luận về alignment tax. Khi chạy sau, cần giữ cùng giới hạn mẫu và cấu hình đánh giá cho SFT và SFT+DPO, ghi điểm cùng stderr rồi đối chiếu với NB4. Không điền dự đoán vào bảng kết quả thực nghiệm.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

**Bonus chưa tổng hợp để nộp trong bản này.** Notebook có output thử nghiệm một số biến thể, nhưng chưa đối chiếu đầy đủ bảng kết quả, ảnh và điều kiện so sánh. Không dùng các ô trống làm kết quả 0; không xếp hạng loss từ số liệu chưa kiểm tra. Phần chính NB0–NB4 không phụ thuộc việc hoàn tất mục này.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | Chưa có kết quả NB7. |
| Sai số chuẩn ≈ √(p(1−p)/n) | Chưa tính vì chưa có p và n. |

Chưa chạy hoặc chưa cung cấp bằng chứng GRPO; không kết luận reward định dạng/đáp án hay mức cải thiện vượt nhiễu.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Trong năm cặp NB2, nhãn `chosen` không bảo đảm đáp án đúng: cặp 759 đảo chiều nhân quả, còn cặp 281 cho thấy đáp án dài hơn có thể chứa nhiều lỗi hơn. Điều bất ngờ sau huấn luyện là reward held-out đạt 67% và chẩn đoán `INTENDED`, nhưng đánh giá đầu ra chỉ đạt win rate 52%, với khoảng tin cậy chứa 50%. Reward theo nhãn và cải thiện chất lượng thực tế là hai câu hỏi khác nhau. Việc Qwen judge không đạt sanity dù Llama đạt 100% cũng cho thấy không thể tin giám khảo chỉ vì tên model hoặc kích thước.

## Chốt phần chính và phần tiếp tục

- Đã xác nhận từ output: SFT hoàn tất, NB2 có split không trùng prompt và thống kê độ dài, NB3 hoàn tất với metrics cuối, NB4 có 58 câu so sánh và summary chấm. Nội dung phản tư bắt buộc đã điền; kết luận không phóng đại mức cải thiện.
- Đã tải bằng chứng gốc về `submission/adapters/dpo/dpo_metrics.json`, `submission/data/pref/` và `submission/data/eval/` (outputs, judge results và summary). Số liệu DPO và judge summary khớp nội dung phản tư. Nếu công cụ chấm yêu cầu đường dẫn root `adapters/` và `data/`, cần đặt các bằng chứng vào đúng vị trí trước khi chạy công cụ đó.
- Đã có ảnh `NB0_DPOloss.png`, `02-sft-loss.png`, `02b-pref-length.png`, `03-dpo-reward-curves.png`, `04-side-by-side-table.png` trong `submission/screenshots/`. Kiểm tra cách đặt tên và nội dung theo rubric trước khi nộp.
- Điền tên và khoá học đã xác nhận trước khi nộp; bản này không tự suy đoán thông tin cá nhân từ tên repository.
- NB3b, NB5–NB7, β-sweep và thử `SFT_SLICE=100` để sau. Không cần chờ bonus để chốt phần chính; chỉ đánh dấu bonus khi đủ sản phẩm và bằng chứng.
