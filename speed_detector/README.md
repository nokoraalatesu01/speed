# Vehicle Speed Detector - Step 6

Buoc 6 nay da co nhan dien xe, tracking ID, vung do hinh thang, tinh toc do km/h, canh bao qua toc do, xuat CSV, va file cau hinh JSON de chay lai de hon.

## Cai dat

```powershell
cd C:\Users\Lenovo\PycharmProjects\PythonProject1\speed-detector
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chay bang webcam

```powershell
python detect_vehicles.py --source 0
```

## Chay bang file video

```powershell
python detect_vehicles.py --source videos\road.mp4
```

## Chay voi du lieu mau

Thu muc `Camera_giao_thong` dang nam o project goc, ngang hang voi `speed-detector`.

```powershell
cd C:\Users\Lenovo\PycharmProjects\PythonProject1\speed-detector
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --config camera_config.json
```

Neu muon xu ly ca 3 video va luu ket qua:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --output-dir outputs
```

Video ket qua se co bounding box, ten loai xe, `ID`, vet di chuyen ngan cua xe, 2 vach A/B, va 2 canh trang gioi han vung do.
Nhung xe nam ngoai hinh thang se khong duoc ve box/ID.
Mac dinh YOLO van nhin toan khung hinh de tracking xe may on dinh hon, nhung chi ve/tinh xe khi tam xe nam trong hinh thang.
Mac dinh toa do vach dang phu hop voi anh mau:

```text
Line A: 0.20,0.67,0.58,0.67
Line B: 0.00,0.81,0.62,0.81
```

Dinh dang la `x1,y1,x2,y2`, moi gia tri la ty le theo kich thuoc khung hinh tu `0` den `1`.

Neu can chinh vi tri 2 vach:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --line-a 0.20,0.67,0.58,0.67 --line-b 0.00,0.81,0.62,0.81
```

Khi xe di qua vach, terminal se in dang:

```text
Track ID 3 crossed line A at 1.25s
Track ID 3 crossed line B at 2.10s
Track ID 3 speed: 42.4 km/h
```

Mac dinh script tam coi khoang cach that giua vach A va B la `10m`. Can do/chon lai dung khoang cach tren mat duong de toc do gan dung:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --distance-meters 10
```

Mac dinh script bo qua toc do tinh duoc tren `150 km/h` de giam nhieu do tracker sai ID hoac xe cat vach qua gan nhau:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --max-speed-kmh 120
```

Mac dinh xe tren `60 km/h` se duoc danh dau qua toc do bang box mau do va chu `OVER`:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --speed-limit-kmh 50
```

Ket qua toc do duoc luu mac dinh vao `outputs\speed_results.csv`. Co the doi file CSV:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --csv outputs\my_results.csv
```

## Cau hinh

File cau hinh mac dinh la `camera_config.json`:

```json
{
  "line_a": "0.20,0.67,0.58,0.67",
  "line_b": "0.00,0.81,0.62,0.81",
  "distance_meters": 10.0,
  "speed_limit_kmh": 60.0,
  "max_speed_kmh": 150.0,
  "confidence": 0.25,
  "tracker": "bytetrack.yaml",
  "detect_roi_only": false
}
```

Co the tao file config tu tham so hien tai:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --save-config camera_config.json --line-a 0.20,0.67,0.58,0.67 --line-b 0.00,0.81,0.62,0.81 --distance-meters 10 --speed-limit-kmh 60
```

Khi chay voi config, van co the ghi de tung gia tri:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --config camera_config.json --speed-limit-kmh 50
```

Neu chi muon test nhanh 30 frame dau va khong mo cua so video:

```powershell
..\.venv\Scripts\python.exe detect_vehicles.py --source ..\Camera_giao_thong --no-show --max-frames 30
```

## Luu video ket qua

```powershell
python detect_vehicles.py --source videos\road.mp4 --output outputs\detected.mp4
```

## Ghi chu

- Lan dau chay, `ultralytics` se tu tai model `yolov8n.pt`.
- Cac loai xe dang nhan dien: `car`, `motorbike`, `bus`, `truck`.
- `--conf` mac dinh la `0.25` de bat xe may nho tot hon. Neu bi nhan sai nhieu, tang len `0.35`.
- Neu muon AI chi nhin vung hinh thang, them `--detect-roi-only`, nhung cach nay co the bo sot xe may nho.
- Neu config dang bat `detect_roi_only`, dung `--full-frame-detect` de ep YOLO nhin toan khung hinh.
- Tracker mac dinh la ByteTrack qua `--tracker bytetrack.yaml`.
- Dung `--config camera_config.json` de nap cau hinh.
- Dung `--save-config camera_config.json` de luu cau hinh hien tai.
- Dung `--line-a` va `--line-b` de dat 2 vach vao dung mat duong trong video.
- Dung `--distance-meters` de nhap khoang cach that giua vach A va B.
- Dung `--max-speed-kmh` de loc toc do bat thuong. Dat `0` neu muon tat loc.
- Dung `--speed-limit-kmh` de dat nguong canh bao qua toc do. Dat `0` neu muon tat canh bao.
- Dung `--csv` de chon file luu ket qua toc do. Dat `--csv ""` neu khong muon luu CSV.
- Nhan phim `q` de tat cua so video.
