# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi checkpoint, package, threshold hoặc mã inference.

### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model dự đoán; chưa phải nhãn đúng đã được xác nhận |
| Ground truth | Nhãn do con người tạo theo guideline |
| Taxonomy | Danh sách lớp cố định đi kèm dữ liệu/checkpoint |
| Checkpoint | File model `.pt` đã được huấn luyện |
| Guideline | Quy tắc dùng để gán nhãn nhất quán |
| Annotator | Người tạo nhãn |
| Reviewer / QC | Người kiểm tra chất lượng nhãn |
| Escalation | Đưa trường hợp chưa có quy tắc rõ ràng lên Lab Coach/GV |
| Model score | Mức tin cậy của model đối với prediction, không phải điểm chất lượng ground truth |

## 1. Phân loại ảnh

Evidence lấy từ `classification_predictions.json`, `sample_id = traffic`, `coco_image_id = 210273`.

1. Record hạng 1: `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K`.
2. Record này nói về cả bức ảnh. Tác vụ là `image_classification`, record không chứa box, polygon hay instance riêng, và model xếp hạng một danh sách lớp cho toàn bộ ảnh đầu vào.
3. Danh sách tên lớp không do model tự nghĩ ra mà thuộc taxonomy ImageNet-1K đi kèm checkpoint `yolo11n-cls.pt`.
4. `class_id` giúp máy xác định lớp một cách không mơ hồ, `class_name` giúp con người đọc được ý nghĩa, còn `taxonomy_name` cho biết ID và tên đó thuộc bộ lớp nào. Cùng một ID hoặc tên có thể được định nghĩa khác trong taxonomy khác.
5. Nếu ảnh có nhiều vật hoặc chủ đề, guideline cần định nghĩa tiêu chí chọn lớp chính, chẳng hạn dựa trên chủ thể trung tâm, diện tích hoặc mục đích của ảnh, đồng thời có quy tắc xử lý khi nhiều lớp cùng phù hợp hoặc không có lớp phù hợp rõ ràng.
6. Không được dùng `score` làm ground truth vì đây chỉ là mức tin cậy của model đối với prediction. Model có thể dự đoán sai dù score cao; ground truth phải được con người xác nhận theo guideline.

## 2. Phát hiện vật thể

Evidence lấy từ `detection_predictions.json`, `sample_id = kitchen`, `coco_image_id = 397133`.

1. Record được chọn: `class_name=person`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68`.
2. Box có góc trên-trái tại pixel `(385.33, 69.24)` và góc dưới-phải tại pixel `(498.92, 348.92)`, tính từ gốc `(0,0)` ở góc trên-trái ảnh. Box rộng `113.58` pixel và cao `279.68` pixel.
3. Với threshold `0.20` có 17 prediction; với threshold `0.60` có 6 prediction. Như vậy hạ threshold giữ lại nhiều prediction hơn.
4. Khi hạ threshold, reviewer phải kiểm tra nhiều box hơn, đặc biệt là false positive, box trùng lặp, sai class và box không ôm đúng vật thể. Reviewer không được coi mọi box được giữ lại là ground truth.
5. Quy tắc box: vẽ một hình chữ nhật nhỏ nhất có thể bao trọn phần vật thể được yêu cầu gán nhãn, các cạnh bám sát biên ngoài của phần nhìn thấy và không chứa thêm nền không cần thiết.
6. Ở mép trái ảnh `kitchen` có một người chỉ xuất hiện một phần (`score=0.610931`, box bắt đầu gần `x=0`). Vì guideline hiện chưa quy định tỷ lệ nhìn thấy tối thiểu hoặc cách xử lý vật bị cắt mép, tôi sẽ escalation lên Lab Coach trước khi quyết định giữ hay bỏ nhãn. Sau khi có quyết định, quy tắc về vật cắt mép cần được bổ sung vào guideline để áp dụng nhất quán cho các ảnh khác.

## 3. Phân đoạn theo từng đối tượng

Evidence lấy từ `segmentation_predictions.json`, `sample_id = kitchen`, `coco_image_id = 397133`.

1. Record được chọn: `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; năm điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
2. Box chỉ cho biết hình chữ nhật bao quanh vật thể, còn polygon mô tả đường biên và hình dạng cụ thể của các phần nhìn thấy. Nhờ đó có thể tách vật thể khỏi nền chính xác hơn.
3. `instance_id` dùng để phân biệt từng đối tượng riêng trong bộ output. Hai người hoặc hai cái bát cùng class vẫn có instance ID khác nhau. Nó không biểu diễn loại lớp như `class_id` và cũng không dùng để theo dõi cùng một vật thể qua nhiều khung hình như tracking ID.
4. Quy tắc mask: đặt các điểm polygon bám theo biên ngoài của phần vật thể nhìn thấy, không lấy nền, không nối sang vật thể khác và không tự suy đoán phần bị che nếu guideline không yêu cầu.
5. Ở góc dưới-trái ảnh `kitchen`, nhiều bát nằm sát hoặc che lên nhau nên một phần đường biên giữa các instance không rõ. Phần biên nhìn thấy rõ có thể xử lý theo guideline; phần bị che khiến phải suy đoán thì cần escalation nếu guideline chưa quy định vẽ theo biên nhìn thấy hay nội suy toàn bộ vật thể. Quyết định của Lab Coach sau đó cần được ghi thành quy tắc chung.

## 4. Vòng đời và kiểm tra chất lượng

Chuỗi khái niệm: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp được xác nhận cho toàn ảnh trong ví dụ single-label này | Ảnh `traffic` có nhiều xe và model chọn `cab`, trong khi nhiều lớp khác cũng có score; lớp chính có thể mơ hồ | Đọc toàn ảnh và chọn đúng một lớp theo tiêu chí lớp chính trong guideline | Kiểm tra lớp có đúng taxonomy, đúng phạm vi toàn ảnh và annotator có áp dụng tiêu chí nhất quán không |
| Phát hiện vật thể | Một `class` và một `bbox_xyxy` cho mỗi vật thể cần gán nhãn | Người ở mép trái bị cắt, ngoài ra nhiều bát nằm gần nhau; threshold thấp làm tăng prediction cần kiểm tra | Vẽ box sát vật thể, gán đúng class và escalation ca cắt mép chưa có quy tắc | Kiểm tra thiếu/thừa vật thể, sai class, box lỏng/chặt, box trùng và cách xử lý vật cắt mép |
| Instance segmentation | Một class và một mask/polygon riêng cho mỗi instance | Các bát ở góc dưới-trái chạm hoặc che nhau, khiến biên từng instance khó xác định | Vẽ polygon theo biên nhìn thấy, giữ từng instance riêng và không tự đoán phần bị che | Kiểm tra mask có lấy nền, thủng, tràn sang vật khác, nhập hai instance hoặc bỏ sót vùng không |

Prediction chỉ là kết quả sau huấn luyện; nếu QC phát hiện sai lớp, thiếu vật thể hoặc box/mask lệch thì nhãn phải được sửa theo guideline và đưa qua vòng rework, không sửa guideline chỉ để làm prediction của model trở nên đúng.

## 5. An toàn dữ liệu

1. Chỉ sử dụng ba ảnh COCO công khai do notebook cung cấp; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng/VinFast hoặc dữ liệu nội bộ lên Colab và GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi sẽ dừng xử lý và báo cho Lab Coach/GV thay vì tự tải lên, chia sẻ hoặc tiếp tục gán nhãn.

