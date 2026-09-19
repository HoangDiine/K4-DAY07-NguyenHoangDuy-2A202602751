# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hoàng Duy
**Nhóm:** YBY1
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Độ tương tự cosine đo góc giữa hai vector trong không gian đa chiều. Giá trị gần 1 nghĩa là hai vector chỉ về cùng một hướng, thể hiện hai văn bản có sự tương đồng ngữ nghĩa rất lớn, bất kể độ dài ngắn khác nhau.

**Ví dụ có độ tương tự CAO:**

- Câu A: Cửa hàng ngừng đón khách lúc nửa đêm.
- Câu B: Tiệm kết thúc giờ bán hàng khi sang ngày mới.
- Tại sao tương đồng: Dùng các từ vựng khác nhau ("cửa hàng" vs "tiệm", "ngừng đón khách" vs "kết thúc giờ bán hàng", "nửa đêm" vs "sang ngày mới") nhưng biểu đạt cùng một ý nghĩa và cùng thời điểm thực tế.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Mặt trời mọc ở hướng đông vào mỗi buổi sáng sớm.
- Câu B: Mạng nơ-ron học sâu tối ưu hàm mất mát bằng thuật toán lan truyền ngược.
- Tại sao khác: Thuộc hai lĩnh vực hoàn toàn tách biệt (hiện tượng tự nhiên đời sống thường ngày vs khoa học máy tính/học sâu), không có giao thoa về ngữ nghĩa.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Khoảng cách Euclid đo độ dài đường thẳng tuyệt đối giữa 2 điểm nên bị ảnh hưởng nặng nề bởi độ dài văn bản (văn bản dài chứa nhiều từ làm độ lớn vector lớn, dẫn đến khoảng cách Euclid xa dù cùng chủ đề). Cosine similarity chỉ đo góc định hướng và triệt tiêu yếu tố độ dài vector, giúp so sánh thuần túy ngữ nghĩa của hai văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> *Trình bày phép tính:*
> Áp dụng công thức: $\lceil (\text{độ dài} - \text{overlap}) / (\text{chunk\_size} - \text{overlap}) \rceil$
> $= \lceil (10000 - 50) / (500 - 50) \rceil = \lceil 9950 / 450 \rceil = \lceil 22.11 \rceil = 23$
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> Khi overlap tăng lên 100: $\lceil (10000 - 100) / (500 - 100) \rceil = \lceil 9900 / 400 \rceil = 25$ chunks (tăng 2 chunks). Cần tăng overlap khi muốn giảm thiểu rủi ro đứt gãy ngữ cảnh tại các ranh giới cắt, bảo đảm thông tin quan trọng nằm giữa hai chunk không bị chia tách làm mất khả năng truy xuất của mô hình.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

> Dùng biểu thức chính quy Positive Lookbehind `(?<=[.!?])\s+` để phân tách tại vị trí khoảng trắng ngay sau dấu câu, giữ nguyên vẹn dấu chấm câu `.!?` cho câu trước thay vì làm mất dấu. Sau khi làm sạch khoảng trắng (`strip()`), gom từng nhóm `max_sentences_per_chunk` câu thành một chunk hoàn chỉnh. Đã ghi nhận edge case chưa xử lý triệt để: các chữ viết tắt (`TS.`, `v.v.`) hoặc số thập phân (`3.14`) sẽ bị nhận diện nhầm thành điểm ngắt câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

> Áp dụng thuật toán chia đệ quy ưu tiên giữ cấu trúc ngữ nghĩa lớn: duyệt danh sách separator từ lớn đến nhỏ (`\n\n` -> `\n` -> `. ` -> ` ` -> `""`). Thuật toán gồm 3 base case (chuỗi rỗng, độ dài $\le$ chunk_size, hoặc hết separator thì cắt cứng) kết hợp bước "gom lên" (merge) nối các mảnh liền kề bằng separator tương ứng chừng nào chưa vượt ngưỡng `chunk_size`, ngăn chặn tình trạng tạo ra các mảnh chunk vụn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

> Bỏ nhánh ChromaDB để loại bỏ rủi ro phụ thuộc môi trường ngoài, sử dụng cấu trúc in-memory list lưu trữ các dictionary record đã chuẩn hóa (`id`, `content`, `metadata`, `embedding`). Khi tìm kiếm (`search`), tính tích vô hướng (dot product) giữa query vector và vector từng chunk (do embedding đã được chuẩn hóa chuẩn $L_2$ nên dot product tương đương cosine similarity), sau đó sắp xếp điểm giảm dần và trả về top-k (đã loại bỏ trường `embedding` để output sạch).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

> Thực hiện lọc trước (pre-filtering): duyệt lọc toàn bộ kho để chọn ra các chunk có metadata thỏa mãn đầy đủ các cặp key-value của `metadata_filter` trước, sau đó mới chạy similarity search trên tập ứng viên này để đảm bảo top-k không bị tài liệu sai lấn chiếm. Hàm `delete_document` lọc bỏ tất cả chunk có `metadata['doc_id'] == doc_id` hoặc `id == doc_id`, trả về `True` nếu kích thước bộ sưu tập giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

> Kiểm tra an toàn: nếu store rỗng thì trả về thông báo ngay lập tức, không tốn tài nguyên gọi LLM. Ngữ cảnh được đưa vào prompt dưới dạng đánh số thứ tự trích dẫn `[1]`, `[2]`,... kèm tên nguồn (`source`/`doc_id`) để đáp ứng tiêu chuẩn Source Traceability; prompt bổ sung ràng buộc chống ảo giác (hallucination), bắt buộc mô hình chỉ dùng thông tin từ ngữ cảnh và nêu rõ nếu không tìm thấy.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- D:\Ai in Action\K4-L3A-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai in Action\K4-L3A-Data-Foundations
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Đo lường thực tế với mô hình `gemini-embedding-001` (qua `CachedEmbedder` và `compute_similarity`):


| Cặp | Câu A                                                                                       | Câu B                                                                             | Dự đoán | Điểm thực tế | Đúng?  |
| ---- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------- | ---------------- | -------- |
| 1    | Quy trình đăng ký môn học trực tuyến của trường đại học.                       | Các bước đăng ký học phần qua cổng thông tin sinh viên.                 | cao        | 0.8337           | Đúng   |
| 2    | Sinh viên hoàn thành đúng hạn học phí học kỳ.                                      | Sinh viên nợ học phí và bị khóa thời khóa biểu.                          | cao        | 0.7946           | Đúng   |
| 3    | Ngân hàng câu hỏi thi kết thúc học phần.                                             | Sinh viên nộp tiền học phí qua tài khoản ngân hàng.                       | thấp      | 0.6349           | Khá cao |
| 4    | Quy định tạm hoãn nghĩa vụ quân sự cho sinh viên chính quy.                        | Thời tiết hôm nay tại Thành phố Hồ Chí Minh nhiều mây và có mưa rào. | thấp      | 0.5197           | Đúng   |
| 5    | Học bổng khuyến khích học tập dành cho sinh viên có điểm rèn luyện và GPA cao. | Chính sách khen thưởng và trợ cấp tài chính cho sinh viên xuất sắc.    | cao        | 0.7893           | Đúng   |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Bất ngờ nhất là Cặp 3: cụm từ "ngân hàng câu hỏi" và "tài khoản ngân hàng" có ngữ nghĩa thực tế khác nhau hoàn toàn (ngân hàng đề thi vs tổ chức tài chính), nhưng embedding vẫn cho điểm 0.6349 (khá cao). Điều này chứng minh rằng mô hình embedding bị ảnh hưởng một phần bởi sự trùng khớp từ vựng ("ngân hàng") và ngữ cảnh giáo dục đại học chung, chứ chưa tách bạch triệt để các nét nghĩa đa nghĩa nếu không có thêm ngữ cảnh xung quanh đủ rộng.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên chiến lược được phân công: **FixedSizeChunker** (`chunk_size=700`, `overlap=80`), backend embedding: OpenAI `text-embedding-3-small`.


| # | Câu hỏi (Query)                                                                                                                | Top-1 Chunk truy xuất được (tóm tắt)                                                                                                              | Điểm Score | Có liên quan không? (Relevant)                                           | Câu trả lời của Agent (tóm tắt)                                                                                                                                   |
| - | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Sinh viên đăng ký môn học Đợt 1 ở đâu và cần lưu ý giới hạn tín chỉ nào?                                     | `ầu học...". SV xem Loại hình lớp và đối tượng sinh viên để chọn loại đăng ký đúng: Quy định về học vụ...`                     | 0.6324       | Không (Trúng tài liệu nhưng chunk thiếu cụm "tối đa 25 tín chỉ") | Sinh viên đăng ký tại MyBK > Đăng ký môn học; phần quy định chi tiết 25 tín chỉ bị chia sang chunk lân cận.                                          |
| 2 | Nếu sinh viên không đăng ký môn học và không có thời khóa biểu trong học kỳ thì có thể bị xử lý thế nào? | `đăng ký vài học phần để giữ tình trạng của SV...` (Chunk chứa đáp án nằm ở Rank 2: score 0.6290)                                     | 0.6381       | Có (Trúng trọn vẹn tại Rank 2)                                         | Sinh viên không có thời khóa biểu trong học kỳ sẽ bị xử lý ra quyết định xóa tên vì không có thời khóa biểu.                                     |
| 3 | Học phí HK1 và HK2 phải thanh toán vào thời điểm nào?                                                                  | `# Học phí Học phí Trung tâm hỗ trợ...` (Chunk chứa đáp án nằm ở Rank 2: score 0.6554)                                                     | 0.6726       | Có (Trúng trọn vẹn tại Rank 2)                                         | Học phí HK1 và HK2 thanh toán 100% học phí, kết thúc ở tuần 4 của học kỳ, thời gian thanh toán trong 1 tuần.                                            |
| 4 | Những môn nào không được phúc tra bài thi cuối kỳ?                                                                    | `# Phúc tra bài thi cuối kỳ... - Những môn không được phúc tra: môn thi trắc nghiệm, môn thí nghiệm...`                                | 0.6598       | Có (Trúng trực tiếp tại Rank 1)                                        | Các môn không được phúc tra gồm: môn thi trắc nghiệm, môn thí nghiệm, thực hành, thực tập, đồ án, đề cương luận văn, luận văn.             |
| 5 | Đăng ký giấy chứng nhận sinh viên thực hiện trên hệ thống nào và nhận ở đâu?*(có filter)*                     | `kiện : SV hệ chính quy... Đăng ký tại MyBK >> Ứng dụng cho Sinh viên >> Đăng ký in Giấy xác nhận sinh viên... nhận tại Phòng CTSV` | 0.6576       | Có (Trúng trực tiếp tại Rank 1)                                        | Đăng ký tại MyBK > Ứng dụng cho Sinh viên > Đăng ký in Giấy xác nhận sinh viên; khi chuyển "Đã in" thì đến nhận tại Phòng Công tác sinh viên. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** (Đạt 8/10 điểm trên bộ test)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> Khi chạy FixedSizeChunker với kích thước 700 ký tự và 80 ký tự overlap trên OpenAI `text-embedding-3-small`, điểm truy xuất đạt mức rất cao (8/10 điểm). Tuy nhiên, hạn chế cố hữu của FixedSize là ranh giới cắt cố định có thể tách rời các ý quan trọng trong cùng một mục quy định (như ở câu 1). Khi đối chiếu với HeadingChunker (10/10) và SentenceChunker (10/10), chiến lược bám sát ngữ pháp câu và cấu trúc điều khoản giúp bảo toàn nguyên vẹn ngữ cảnh. Điểm tương đồng của OpenAI phân bố rất ổn định (0.50 – 0.68) và phân biệt rõ nét giữa chunk chứa thông tin cốt lõi so với chunk thông tin râu ria.

---

## Tự Đánh Giá (Phần Cá Nhân)


| Tiêu chí                                           | Điểm tự đánh giá |
| ---------------------------------------------------- | ---------------------- |
| Khởi động (Warm-up)                               | 5 / 5                  |
| Hướng tiếp cận của tôi (My Approach)           | 10 / 10                |
| Hoàn thiện code (Core Implementation — tests)     | 30 / 30                |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5                  |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10                |
| **Tổng phần cá nhân**                            | **60 / 60**            |
