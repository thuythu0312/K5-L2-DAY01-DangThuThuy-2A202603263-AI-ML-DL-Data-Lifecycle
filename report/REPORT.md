# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh, không phải một vật thể riêng lẻ. Vì mô hình đang thực hiện tác vụ Image Classification, tức là phân loại ảnh tổng thể và đưa ra lớp phù hợp nhất. Trong ảnh có nhiều phương tiện giao thông như xe buýt, ô tô và xe tải nhỏ, nhưng mô hình vẫn đưa ra một kết quả phân loại đứng đầu là cab. Kết quả này không chỉ rõ vị trí hay ranh giới của một chiếc xe cụ thể nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp không phải do mô hình tự nghĩ ra mà được quy định sẵn trong bộ nhãn ImageNet-1K đi kèm với checkpoint yolo11n-cls.pt. Khi dự đoán, mô hình lựa chọn và tính điểm cho các lớp đã có trong taxonomy này, chứ không tự tạo ra tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả ba thông tin để xác định chính xác nhãn được sử dụng. class_id là mã định danh của lớp, class_name là tên giúp người đọc hiểu kết quả, còn taxonomy_name cho biết nhãn đó thuộc bộ phân loại nào. Nếu chỉ ghi tên lớp thì có thể gây nhầm lẫn khi các bộ taxonomy khác nhau sử dụng cách đặt tên hoặc đánh số khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí lựa chọn một nhãn duy nhất, chẳng hạn ưu tiên đối tượng chính, đối tượng nổi bật nhất hoặc nội dung thể hiện chủ đề chính của bức ảnh. Đồng thời, cần có hướng dẫn xử lý khi nhiều đối tượng cùng xuất hiện hoặc không xác định được đối tượng chính, để những người gán nhãn khác nhau có thể đưa ra kết quả thống nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ thể hiện mức độ tin cậy mà mô hình gán cho một lớp dự đoán, không chứng minh dự đoán đó là chính xác. Ví dụ, trong ảnh giao thông, mô hình dự đoán cab với score 0.510915, nhưng ảnh thực tế có nhiều loại phương tiện khác nhau. Mô hình vẫn có thể dự đoán sai dù score cao. Vì vậy, ground truth cần được xác định dựa trên dữ liệu thực tế, quy tắc gán nhãn rõ ràng và quá trình kiểm tra của con người, thay vì lấy trực tiếp kết quả dự đoán của mô hình.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`= 0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box này bao quanh người đang đứng ở phía bên phải bức ảnh, quay lưng về phía máy ảnh. Góc trên bên trái của box có tọa độ (385.33, 69.24), còn góc dưới bên phải là (498.92, 348.92). Box có chiều rộng khoảng 113.58 pixel và chiều cao 279.68 pixel, được xác định theo hệ tọa độ của ảnh gốc có kích thước 640 × 427 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Em chọn hai ngưỡng confidence là 0.35 và 0.60.
Ngưỡng 0.35: có 11 prediction.
Ngưỡng 0.60: có 6 prediction.
Như vậy, khi tăng ngưỡng từ 0.35 lên 0.60, số lượng prediction giảm từ 11 xuống 6. Điều này cho thấy ngưỡng cao hơn sẽ loại bớt những dự đoán có độ tin cậy thấp, nhưng cũng có thể bỏ sót một số vật thể thực sự xuất hiện trong ảnh.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, mô hình giữ lại nhiều bounding box hơn nên reviewer phải kiểm tra nhiều dự đoán hơn. Cụ thể, cần xác định box nào phát hiện đúng vật thể, box nào bị nhầm, bị trùng lặp hoặc có vị trí chưa chính xác. Reviewer cũng cần kiểm tra lại tên lớp và điều chỉnh hoặc loại bỏ những box không phù hợp trước khi chấp nhận kết quả gán nhãn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể cần được gán một bounding box hình chữ nhật ôm sát phần vật thể nhìn thấy được trong ảnh, không lấy thừa quá nhiều vùng nền xung quanh. Tọa độ box phải nằm trong giới hạn của ảnh và được xác định thống nhất theo định dạng [x_min, y_min, x_max, y_max]. Quy tắc này giúp việc gán nhãn chính xác và đồng nhất giữa các ảnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, có một người ở phía mép trái bị cắt khỏi khung hình, chỉ nhìn thấy một phần cơ thể. Em sẽ đưa trường hợp này vào guideline vì đây là tình huống khá phổ biến trong bài toán phát hiện vật thể. Quy tắc đề xuất là vẫn gán nhãn cho vật thể bị cắt mép nếu phần còn lại đủ rõ để xác định đúng lớp, đồng thời chỉ vẽ bounding box trong phạm vi ảnh. Với những trường hợp bị che khuất quá nhiều, không thể nhận diện chắc chắn, người gán nhãn cần đánh dấu để hỏi Lab Coach thay vì tự suy đoán.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 
3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ xác định vị trí và phạm vi của vật thể bằng một hình chữ nhật, còn polygon thể hiện đường viền chi tiết theo hình dạng thực tế của vật thể. Ví dụ, với người đang đứng trong bếp, polygon có thể mô tả rõ phần đầu, vai, tay và chân thay vì bao gồm cả vùng nền xung quanh như bounding box. Nhờ vậy, việc phân đoạn giúp xác định chính xác hơn những pixel thuộc vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng cụ thể được phát hiện trong một ảnh, kể cả khi chúng thuộc cùng một lớp. Ví dụ, hai người trong ảnh kitchen có thể cùng thuộc lớp person nhưng được gán hai instance_id khác nhau. class_id chỉ xác định loại vật thể, còn tracking ID dùng để theo dõi cùng một vật thể qua nhiều khung hình trong video. Vì vậy, instance_id ở đây chỉ giúp nhận diện riêng từng đối tượng trong kết quả phân đoạn của ảnh, không có ý nghĩa theo dõi chuyển động.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Khi vẽ mask, người gán nhãn cần đi theo đường viền thực tế của phần vật thể nhìn thấy được, bao gồm cả những phần lồi lõm đặc trưng và hạn chế lấy thừa vùng nền xung quanh. Nếu vật thể bị che khuất, chỉ đánh dấu phần có thể quan sát được, không tự suy đoán phần bị che. Quy tắc này giúp các mask chính xác và thống nhất giữa những người gán nhãn.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, trên bàn bếp có nhiều chiếc bát và dụng cụ đặt sát nhau, một số vị trí bị che khuất một phần bởi các đồ vật khác. Em sẽ ưu tiên xử lý bằng guideline vì đây là tình huống thường gặp khi phân đoạn đối tượng. Quy tắc là mỗi vật thể cần có mask riêng, đường biên chỉ bao quanh phần nhìn thấy được và không gộp hai vật thể thành một. Nếu ranh giới quá mờ hoặc không thể xác định vật thể thuộc lớp nào thì mới chuyển lên hỏi Lab Coach để thống nhất cách xử lý.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho toàn bộ ảnh | Ảnh traffic có nhiều loại xe nhưng mô hình dự đoán cab với score 0.510915, khó xác định một lớp đại diện | Quan sát toàn ảnh và chọn nhãn phù hợp nhất theo guideline | Kiểm tra nhãn có phản ánh đúng nội dung chính của ảnh và tuân thủ guideline khôn |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và một bounding box | Ảnh kitchen có người ở mép trái bị cắt một phần, mô hình vẫn phát hiện person với score 0.610931 | Xác định từng vật thể, vẽ box sát phần nhìn thấy và gán đúng lớp | Kiểm tra vị trí, kích thước box, nhãn sai, box trùng và vật thể bị bỏ sót |
| Instance segmentation | Mỗi đối tượng có một nhãn lớp và một mask/polygon thể hiện vùng pixel của vật thể | Ảnh kitchen có nhiều bát và dụng cụ trên bàn đặt sát hoặc che khuất nhau, khiến việc xác định đường biên khó khăn | Vẽ mask theo đường viền phần vật thể nhìn thấy, tách riêng từng đối tượng | Kiểm tra mask có ôm sát vật thể, lấn sang nền, gộp nhầm hoặc bỏ sót vùng nào không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Em chỉ sử dụng dữ liệu được cung cấp trong phạm vi bài Lab, không tự ý chia sẻ, tải lên nền tảng công khai hoặc sử dụng dữ liệu cho mục đích khác khi chưa được cho phép
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi bài Lab, em sẽ dừng xử lý ngay và báo cho Lab Coach để được hướng dẫn, không tự ý sử dụng, chỉnh sửa hoặc tiếp tục gán nhãn

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
