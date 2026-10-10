# S2IT cho Aspect Sentiment Quad Prediction

Triển khai mã nguồn theo paper **S²IT: Stepwise Syntax Integration Tuning for Large Language Models in Aspect Sentiment Quad Prediction**. Pipeline làm việc với JSONL trong `asqp_data/` và dự đoán quadruple gồm đúng bốn trường:

- `aspect_term`
- `aspect_category`
- `opinion_term`
- `sentiment`

`aspect_term` hoặc `opinion_term` có thể là `null` với thành phần implicit.

## Các bước theo paper

1. **Global Syntax-guided Extraction:** Stanza tạo dependency tree; các cạnh được biểu diễn bằng ngôn ngữ tự nhiên (`head modify dependent`, `root depend token`) và nối vào câu. Mô hình được fine-tune cả hai hướng aspect→opinion và opinion→aspect.
2. **Local Syntax-guided Classification:** Dùng các cặp aspect–opinion (gold khi huấn luyện, dự đoán khi inference), tạo subgraph lân cận một bước cho từng aspect và opinion rồi phân loại quadruple thành category và sentiment.
3. **Fine-grained Structure Instruction Tuning:** fine-tune tiếp trên tác vụ liên kết aspect→opinion/opinion→aspect dựa trên global dependency relations, cùng các tác vụ node classification (category/sentiment cho aspect/opinion) và phân loại quadruple dựa trên local one-hop subgraphs.

Khi inference, model giải mã cặp aspect–opinion hai chiều bằng beam search (`num_beams=4`), gộp các cặp trùng nhau, rồi phân loại cặp với local syntax để tạo quadruple. Cách gộp hợp nhất hai hướng là quy ước triển khai cho trường hợp kết quả hai hướng lệch nhau.

Mặc định dùng Qwen2.5-7B-Instruct và LoRA rank 32, alpha 32, dropout 0.1, learning rate `5e-5`, 5 epochs, batch size 4, gradient accumulation 2, cosine scheduler — cấu hình được nêu trong paper. Cần GPU đủ bộ nhớ để fine-tune mô hình này; có thể chọn model khác qua `--model`.

## Cài đặt

Từ thư mục project, trong virtual environment Python:

```powershell
python -m pip install -e ".[dev]"
```

Paper minh họa tiếng Anh, trong khi các JSONL mẫu có thể gồm nhiều ngôn ngữ. Gắn mã ngôn ngữ cho từng câu bằng lệnh `prepare-language` dưới đây; Stanza cần có model tương ứng cho mọi language code thực tế trong data. Lệnh tải resources mặc định cho các ngôn ngữ:

```powershell
asqp-s2it prepare-parser --languages en,es,vi
```

Có thể truyền ngôn ngữ cố định như `--language en` khi toàn bộ input chỉ có một ngôn ngữ. Notebook Kaggle chỉ cho phép tiếng Anh (`en`), Việt (`vi`), Trung giản thể (`zh-hans`), Nhật (`ja`), Đức (`de`), Thái (`th`) và Pháp (`fr`). `langdetect` vẫn nhận diện mọi record, nhưng mã ngoài danh sách bị loại khỏi từng split và được thống kê trước khi tải model; các biến thể tiếng Trung được ánh xạ vào `zh-hans`. Notebook chỉ tải các model `tokenize,pos,depparse` vào `/kaggle/working/stanza_resources` (không tải `default.zip` chứa toàn bộ processor). Pipeline cấu hình lemma identity để đáp ứng prerequisite bắt buộc của depparse nhưng không cần tải model lemma; parser chỉ dùng token và dependency heads. Sau mỗi language, notebook xóa cache Hugging Face riêng của Stanza language đó; cache của model LLM không bị xóa. Nếu một split không còn record nào sau bước lọc, notebook dừng với lỗi rõ ràng thay vì tiếp tục train/evaluate trên split rỗng.

Nếu đã chạy bản notebook cũ và Kaggle báo hết dung lượng, notebook mới tự xóa cache Stanza cũ cho language đang tải, gồm `default.zip`/các file được bung từ gói cũ và bản sao riêng trong Hugging Face cache, trước khi tải các processor cần thiết.

### Gắn sẵn ngôn ngữ cho từng bản ghi

Có thể chuẩn bị language trước khi tải dữ liệu lên Kaggle bằng lệnh dưới đây. Bước này cần `langdetect`; notebook Kaggle cài và dùng package này để nhận diện **mọi câu**, luôn ghi đè trường `language` cũ nếu có.

```powershell
python -m pip install langdetect
python -m asqp_s2it.language_prep --input asqp_data/train.jsonl --output asqp_data_tagged/train.jsonl
python -m asqp_s2it.language_prep --input asqp_data/dev.jsonl --output asqp_data_tagged/dev.jsonl
python -m asqp_s2it.language_prep --input asqp_data/test.jsonl --output asqp_data_tagged/test.jsonl
```

Lệnh giữ nguyên mọi annotation và ghi trường `"language"` mới (ví dụ `"en"`, `"es"`, `"vi"`) vào mỗi JSONL record. Kiểm tra phân bố ngôn ngữ do lệnh in ra, sau đó upload `asqp_data_tagged/` lên Kaggle và đặt `DATA_DIR` trỏ tới thư mục này. Language đã có trong input không được sử dụng.

Nếu text rỗng hoặc không có đặc trưng để `langdetect` nhận diện, tool/notebook gán record đó language phổ biến nhất trong toàn bộ train/dev/test và in cảnh báo kèm số lượng/vị trí mẫu. Nếu không phát hiện được language nào, quá trình dừng với lỗi hướng dẫn thay vì tự gán tùy tiện.

Trên Kaggle, bật **Settings → Internet** để cài dependencies và tải registry/parser models. `langdetect` không cần model tải riêng: các profile ngôn ngữ đi kèm package Python. Kaggle chỉ giữ en, vi, zh-hans, ja, de, th, fr; mọi language khác bị loại khỏi train/dev/test và được thống kê, không tải model Stanza cho chúng. Khi dùng model LLM local, vẫn cần thêm model đó qua Kaggle Input; có thể dùng Hugging Face model ID nếu muốn tải LLM trực tuyến.

**Lưu ý source code trên Kaggle:** chọn **Add Input** cho notebook hoặc model không tự thêm source repository. Nếu log chỉ có `/kaggle/input/notebooks` và `/kaggle/input/models`, hãy tạo Kaggle Dataset từ project có `pyproject.toml` và `src/asqp_s2it`, sau đó thêm dataset đó bằng **Add Input**. Để `PROJECT_DIR=None` cho notebook tự dò repository; nếu có nhiều repo, đặt `PROJECT_DIR` đúng thư mục chứa hai thành phần trên. Chỉ dùng đường dẫn `/kaggle/input/...` thực sự tồn tại trong runtime.

Sau khi cập nhật code, cần đóng gói/tải lại Kaggle Dataset chứa source repository (hoặc sửa source được mount làm Input) rồi khởi động lại session. Nếu không, Kaggle tiếp tục chạy bản cũ từ `/kaggle/input` dù notebook đã được cập nhật; traceback còn chọn model lemma như `imst_nocharlm.pt` hoặc `gsdsimp_nocharlm.pt` là dấu hiệu source cũ chưa có cấu hình `lemma="identity"`. Notebook in đường dẫn source parser/training đang import và dừng sớm nếu phát hiện hai file source cũ.

### Theo dõi thời gian và tiến độ train

Trong lúc chuẩn bị dependency examples, log in tiến độ train/dev khoảng mỗi 5% record, kèm thời gian đã chạy và ETA ước tính. Trong từng stage fine-tuning, callback in step hiện tại/tổng số step, phần trăm, epoch, loss, elapsed và ETA mỗi 10 optimizer steps (kèm log step đầu tiên). Tải model và tokenization của từng stage cũng có thông báo bắt đầu/kết thúc và thời lượng; tổng thời gian từng stage và toàn bộ vẫn được ghi vào `training_metrics.json`.

Nếu muốn chạy hoàn toàn offline, notebook vẫn hỗ trợ chỉ định Stanza resources hoặc model files tĩnh. Khi chạy online mặc định, resources được tải vào `/kaggle/working/stanza_resources` và Stanza chọn theo `language` (`en`, `vi`, …) của từng record.

Với bộ model rời có cấu trúc:

```text
stanza-models/
├── tokenize/combined.pt
├── pos/combined_charlm.pt
├── lemma/combined_nocharlm.pt
├── depparse/combined_charlm.pt
├── pretrain/conll17.pt
├── forward_charlm/1billion.pt
└── backward_charlm/1billion.pt
```

đặt `STANZA_MODEL_DIR` thành thư mục `stanza-models` trên Kaggle và `STANZA_MODEL_LANGUAGE` thành mã ngôn ngữ của các model (ví dụ `en`). Notebook kiểm tra đủ file và yêu cầu tất cả bản ghi JSONL có cùng language đó; model cho một ngôn ngữ không thể dùng an toàn cho records thuộc ngôn ngữ khác. Notebook tạo metadata resources tối thiểu trong `/kaggle/working` để Stanza nạp các model bằng tham số đường dẫn trực tiếp; không cần `resources.json` trong thư mục model rời. Nếu dữ liệu nhiều ngôn ngữ, dùng bộ Stanza resources đầy đủ có các model tương ứng cho tất cả languages.

## Huấn luyện

```powershell
asqp-s2it train `
  --train asqp_data/train.jsonl `
  --dev asqp_data/dev.jsonl `
  --model Qwen/Qwen2.5-7B-Instruct `
  --output-dir runs/s2it
```

Ba giai đoạn được huấn luyện lần lượt trên cùng LoRA adapter; adapter cuối cùng và danh sách category lưu ở `runs/s2it/final/`. Thời gian từng giai đoạn, tổng thời gian fine-tune và tổng thời gian chạy được in ra terminal và lưu trong `runs/s2it/final/training_metrics.json` (đồng thời có bản sao ở `runs/s2it/training_metrics.json`). `training_time` là tổng thời gian chạy fine-tune của ba giai đoạn; `total_elapsed_time` bao gồm tạo dữ liệu, tải model và lưu adapter. JSONL có dạng một câu mỗi dòng với `text` và `quads`; loader lấy `aspect_term`, `aspect_category`, `opinion_term`, `sentiment` và không dùng các metadata khác để tính điểm.

## Dự đoán và F1

```powershell
asqp-s2it predict `
  --input asqp_data/test.jsonl `
  --model-dir runs/s2it/final `
  --output runs/s2it/predictions.jsonl
```

File dự đoán lưu `instance_id`, `text`, `quadruples`. Khi đầu vào có gold `quads`, lệnh cũng in F1 cùng thời gian training đọc từ `training_metrics.json`. Để tính lại:

```powershell
asqp-s2it evaluate `
  --gold asqp_data/test.jsonl `
  --predictions runs/s2it/predictions.jsonl
```

Lệnh `predict` tự đọc `training_metrics.json` cạnh model và in cả thời gian training lẫn F1. `evaluate` cũng hiển thị thời gian nếu file metrics mặc định tồn tại; có thể chỉ định file khác bằng `--training-metrics`.

Precision/recall/F1 là micro trên các quadruple duy nhất. Một dự đoán chỉ là true positive khi cả bốn nhãn cùng khớp trong cùng instance; thiếu/thừa nhãn hoặc gán nhãn cho sai instance đều không khớp. Chuẩn hóa Unicode NFC, khoảng trắng và chữ hoa/thường trước khi so sánh; không tính partial-match từng trường. F1 được in theo thang 0–1.

Chạy kiểm thử:

```powershell
python -m pytest
```
