# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.826      0.905        0.919   
2. BoW + MLP (deep learning)         0.988   0.924      0.955        0.974   
3a. Encoder (MiniLM) + kNN           0.817   0.967      0.886        1.000   
3b. Encoder (MiniLM) + LogReg        0.967   0.967      0.967        0.976   
3c. Encoder (MiniLM) + MLP head      0.978   0.957      0.967        1.000   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.141      58674.832       0.030   
2. BoW + MLP (deep learning)         3.492      55306.344       0.134   
3a. Encoder (MiniLM) + kNN          57.291         34.254      30.760   
3b. Encoder (MiniLM) + LogReg       57.467         44.576      26.377   
3c. Encoder (MiniLM) + MLP head     58.088         36.903      22.876   
4a. LLM nhỏ zero-shot                0.000          0.923     743.966   
4b. LLM lớn zero-shot                0.000          0.948     995.112   
4c. LLM nhỏ few-shot                 0.000          1.196     739.651   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.051         0.000  
2. BoW + MLP (deep learning)          0.217         0.000  
3a. Encoder (MiniLM) + kNN           46.592         0.000  
3b. Encoder (MiniLM) + LogReg        35.557         0.000  
3c. Encoder (MiniLM) + MLP head      32.726         0.000  
4a. LLM nhỏ zero-shot              1400.456         0.008  
4b. LLM lớn zero-shot              1374.678         0.083  
4c. LLM nhỏ few-shot               1308.664         0.017  

## 3. Dự đoán và thực tế

1. predict("hard_eval", "(1) accuracy  BoW + LogReg: 21/50, BoW + MLP: 25/50, Encoder + LogReg: 32/50, LLM nhỏ zero-shot: 38/50, LLM lớn zero-shot: 43/50, LLM nhỏ few-shot: 42/50, Encoder + kNN: 28/50, Encoder + MLP head: 33/50. (2) Nhóm obfuscation khó nhất với BoW, Nhóm prompt_injection khó nhất với LLM")
    Dự đoán: LLM nhỏ zero-shot đúng 38/50 tin, tương đương 76%.
    Kết quả: Chỉ đúng 21/50 tin, tương đương 42%, thấp hơn dự đoán 34 điểm phần trăm; F1 bằng 0.
    Vì sao lệch: Kỳ vọng LLM nhận diện được ngữ cảnh và các kiểu spam khó, nhưng hệ thống thực tế trả toàn ham. Phần kiểm tra SUB cũng có invalid=40/40, và code chuyển phản hồi không hợp lệ thành ham, chênh lệch này phản ánh vấn đề đầu ra của hệ thống.
2. predict("dl_relu", "Bỏ ReLU thì 2 lớp Linear sẽ thành 1 phép biến đổi tuyến tính là nhân ma trận với nhau tương đương với mô hình logistic regression, Dự đoán F1 giống với logistic regression.")
    Dự đoán: Bỏ ReLU thì 2 lớp Linear sẽ thành 1 phép biến đổi tuyến tính là nhân ma trận với nhau tương đương với mô hình logistic regression.
    Kết quả: Mạng không ReLU đạt F1 = 0,961, trong khi Logistic Regression tự code đạt 0,905, chênh 0,056.
    Vì sao lệch: Nhầm giữa cùng dạng hàm biểu diễn và cùng kết quả sau huấn luyện. Hai lớp Linear có thể gộp thành một phép biến đổi affine, nhưng cách tham số hóa, khởi tạo và tối ưu khác nhau. Mạng dùng Adam với mini-batch, còn logistic regression tự code dùng gradient descent toàn bộ dữ liệu, nên trọng số và F1 cuối cùng không nhất thiết giống nhau.
3. predict("dl_curve", "(1) vai loss không giảm mãi (2) MLP sẽ nhỏ hơn logistic regression một chút về F1")
    Dự đoán: “(2) MLP sẽ nhỏ hơn logistic regression một chút về F1.”
    Kết quả: MLP đạt F1 = 0,955, cao hơn logistic regression 0,905 là 0,050, ngược với chiều dự đoán.
    Vì sao lệch: Đã đánh giá thấp lợi ích của lớp ẩn và ReLU. MLP có thể học các quan hệ phi tuyến giữa những từ xuất hiện trong tin, trong khi logistic regression cộng các đóng góp tuyến tính. Khác biệt về cách tối ưu cũng có thể góp phần tạo ra kết quả này; không thể quy toàn bộ mức tăng cho kiến trúc mạng.

## 4. Phân tích 50 case khó

BoW + LogReg tụt F1 nhiều nhất ở nhóm 50 case khó: từ 0,919 trên SUB xuống 0,333 trên 50 case khó
    - #13 [Spam giọng tự nhiên] thật=spam: Hope you're well! Quick reminder that your photos from the event are ready, view & order prints
       ↳ bẫy: Giả làm nhắc nhở thân thiện, link bán hàng không yêu cầu
       Nhãn thật: spam; dự đoán: ham. BoW chỉ ghi nhận các từ xuất hiện, không biểu diễn được mối quan hệ giữa lời mở đầu thân thiện và mục đích quảng cáo.
    - #16 [Spam giọng tự nhiên] thật=spam: Long time no see! I've started a little side business with skincare, you'd love it, can I send 
       ↳ bẫy: Giả làm bạn cũ, bán hàng đa cấp. Nhãn thật: spam; dự đoán: ham. Tin dùng giọng người quen để tiếp cận bán hàng. Những từ trò chuyện thông thường có thể khiến tổng trọng số nghiêng về ham. BoW không trực tiếp suy luận được ý định tiếp thị ẩn sau cách nói thân mật.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
- Đề bài: 🎯 Đề #5: Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
- Cách làm: Tạo danh sách my_messages, gán nhãn thật cho từng tin và đưa cùng nội dung vào hai hàm trong MODELS. Sau đó, Lập bảng đối chiếu dự đoán với nhãn thật.
- Kết quả: BoW + LogReg đúng 3/4 tin, Embedding + LogReg đúng 4/4 tin.
- Giải thích: 
Tin nhắn đầu tiên: dạng spam né từ khoá, thật là spam. Các từ c4sh, l0an, n0 bị biến dạng nên BoW nhầm thành ham, Embed tận dụng ngữ cảnh vay tiền và lời mời truy cập link nên đánh dấu là spam.
Tin nhắn thứ 2: dạng spam né từ khoá, thật là spam. Cũng có các từ biến dạng nhưng vẫn còn thông tin về giá tiền và signup nên cả 2 model đều phân biệt là spam
Tin nhắn thứ 3: Ham chứa free, thật là ham. free ở trong ngữ cảnh này là phòng đang trống, ngữ cảnh đặt phòng và thảo luận dự án cho thấy đây là tin công việc bình thường. Cả 2 model đều đánh dấu là ham
Tin nhắn thứ 4: Ham chứa win, thật là ham. win trong ngữ cảnh này là chiến thắng trò chơi chứ không liên quan đến lời mời trúng thưởng. BoW tiếp tục dựa vào trọng số của các từ liên quan để nghiêng về ham. cả 2 model đều đánh dấu là ham.


## 6. Phương án cho sản phẩm thật

- Phương án: Dùng BoW + MLP xử lý bước đầu chuyển những tin có xác suất spam sang nằm trong khoảng [0.3, 0.7] sau đó chuyển sang Embedding + LogReg để lọc những trường hợp dùng từ ngữ biến dạng sau đó sử dụng LLM để xử lý những tin khó hơn là bước cuối cùng. Cách này tốt hơn là dùng 1 cách duy nhất vì n tận dụng tốc độ của BoW + MLP, khả năng hiểu ngữ nghĩa của Embedding + LogReg và cuối cùng là khả năng phân tích ngữ cảnh và ý định quan hệ của LLM.
- Số liệu thống kê ở bảng kết quả chính: BoW + MLP có p95 khoảng 0,17 ms/tin, thông lượng khoảng 65.521 tin/giây, vượt nhu cầu trung bình 116 tin/giây của 10 triệu SMS/ngày. Chất lượng đạt F1 = 0,974 trên SUB, nhưng giảm còn 0,450 trên 50 case khó. Embedding + LogReg có p95 khoảng 25,97 ms/tin, F1 SUB = 0,976 và F1 case khó = 0,692. Vì thông lượng đo được chỉ khoảng 48 tin/giây, tôi chỉ chuyển một phần tin sang model này và bổ sung năng lực xử lý khi cần.
- Chấp nhận đánh đổi: BoW + MLP nhanh và tiết kiệm tài nguyên nhưng dễ bỏ sót spam có cách diễn đạt mới. Bổ sung embedding giúp xử lý một số trường hợp khó, đổi lại tăng tài nguyên và độ trễ.


## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

| STT | Nội dung hỏi AI | Nội dung AI hỗ trợ | Cách kiểm chứng |
|:---:|:---------------|:-------------------|:----------------|
| 1 | Công thức cosine similarity | Giải thích tích vô hướng, độ dài vector và từng bước tính cosine | Tính lại ví dụ trong đề và chạy kiểm tra `HAND_CALC` |
| 2 | Công thức sigmoid và cách kNN dự đoán | Giải thích cách sigmoid chuyển điểm số thành xác suất; cách kNN chọn k láng giềng gần nhất và bỏ phiếu | Đối chiếu công thức với code và kết quả trong notebook |
| 3 | Cơ chế BoW + Logistic Regression | Giải thích cách biểu diễn từ và tổng hợp trọng số | Đối chiếu kết quả; phân biệt dự đoán đúng với hiểu ngữ nghĩa |

- Điều hiểu ra hôm nay là: BoW có thể phân loại đúng nhờ các dấu hiệu thống kê mà không trực tiếp hiểu nghĩa cả câu. Tôi cũng nhận ra rằng cần kiểm tra tính hợp lệ của đầu ra và đánh giá trên nhiều nhóm dữ liệu. Khi chọn model cho sản phẩm thật, phải xét đồng thời chất lượng, độ trễ, thông lượng và chi phí.