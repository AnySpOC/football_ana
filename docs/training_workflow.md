# Training Workflow

このプロジェクトでは、持っている試合動画からYOLO学習用データを作ります。

## 1. フレーム抽出

```powershell
.\.venv\Scripts\python backend\scripts\extract_training_frames.py `
  C:\Users\foxho\Downloads\fusalIMG_4626.MOV `
  --dataset data\datasets\futsal `
  --fps 2 `
  --max-seconds 120
```

- `--fps 2`: 1秒あたり2枚だけ抽出します。
- `--max-seconds 120`: まず2分だけ作ります。
- `--val-every 5`: 5枚に1枚を検証用に回します。

## 2. 既存YOLOで仮ラベル作成

```powershell
.\.venv\Scripts\python backend\scripts\autolabel_yolo_dataset.py `
  --dataset data\datasets\futsal `
  --model yolo11n.pt `
  --person-conf 0.25 `
  --ball-conf 0.08 `
  --imgsz 1280
```

これで以下ができます。

```text
data/datasets/futsal/images/train
data/datasets/futsal/images/val
data/datasets/futsal/labels/train
data/datasets/futsal/labels/val
```

## 3. 人間が修正

CVAT、Roboflow、Label Studio、makesense.ai などで、特にボールの漏れ・誤検出を修正します。

最初に重点的に直すもの:

- 小さいボール
- ブレたボール
- 選手に隠れたボール
- 床と似た色のボール
- 人物の誤検出

## 4. YOLO追加学習

```powershell
.\.venv\Scripts\yolo detect train `
  model=yolo11n.pt `
  data=ml\datasets\futsal\data.yaml `
  epochs=50 `
  imgsz=1280 `
  batch=4
```

学習後のモデルは通常ここにできます。

```text
runs/detect/train/weights/best.pt
```

## 5. 学習済みモデルで動画生成

```powershell
.\.venv\Scripts\python backend\scripts\create_overlay.py `
  data\uploads\fusalIMG_4626.MOV `
  --model runs\detect\train\weights\best.pt `
  --output data\results\fusalIMG_4626_trained_overlay.mp4 `
  --seconds 30 `
  --conf 0.25 `
  --ball-conf 0.08 `
  --imgsz 1280 `
  --swap-teams
```

## 進め方

最初から全動画を全部ラベル付けしません。

1. まず100〜300枚だけ作る
2. ボール中心に修正する
3. 追加学習する
4. 新モデルでまた仮ラベルを作る
5. 間違いだけ直す

この繰り返しで精度を上げます。
