# Fooball Ana

フットサル、ソサイチ、サッカーの試合動画から、選手・チーム・ボール・イベントを推定してサマリデータを作るためのプロトタイプです。

## 目標

- 試合動画を読み込む
- ユニフォーム色からチームを分類する
- 可能な範囲で選手を追跡する
- ボール位置を推定する
- 移動距離、ポゼッション、シュート数、パス成功率などを推定する
- 動画上に解析結果をオーバーレイ表示する
- サッカー動画データセットと機械学習で精度を上げる

## 現在のプロトタイプ

- 動画アップロード
- OpenCVによる動画メタ情報取得
- 再生中の動画にCanvasオーバーレイ表示
- 味方選手、敵選手を色付きの四角で表示
- 再生時間に同期したパス数、ポゼッション、ボール位置のデモ表示
- サンプル再生モード
- APIログ出力

## 構成

```text
backend/      動画解析API
frontend/     ブラウザUI
ml/           学習・評価・データセット管理
data/         ローカル動画・解析結果
docs/         設計メモ
```

## 設計・ライセンス

- [システム設計書 v2](docs/system_design_v2.md)
- [学習データ作成手順](docs/training_workflow.md)
- [第三者ソフトウェアと著作権上の注意](THIRD_PARTY_NOTICES.md)

## 開発ロードマップ

1. 動画アップロードと解析ジョブ作成
2. OpenCVで動画メタ情報を取得
3. YOLOで人物・ボール検出
4. ByteTrackまたはDeepSORTで選手追跡
5. ユニフォーム色でチーム分類
6. 軌跡、ヒートマップ、イベント推定
7. モデル学習・評価パイプライン
8. Webアプリからスマートフォンアプリへ展開

## 起動

バックエンド:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

OpenCVを使った動画メタ情報取得や検出処理まで試す場合は、Python 3.11/3.12系の仮想環境が安定です。以下を使います。

```powershell
pip install -r backend\requirements-vision.txt
```

フロントエンド:

```powershell
cd frontend
python -m http.server 5174
```

ブラウザで `http://localhost:5174` を開きます。

サンプル確認:

1. `サンプルを表示` を押す
2. `サンプル再生` を押す
3. ピッチ上に味方/敵の枠、パス数、ポゼッション、ボール位置が表示されます

実動画確認:

1. 試合動画を選択する
2. `解析を開始` を押す
3. 動画メタ情報がサマリに表示されます
4. 動画再生中、仮の味方/敵枠とパス数オーバーレイが再生時間に同期して表示されます

ログ:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/logs?lines=50
```
