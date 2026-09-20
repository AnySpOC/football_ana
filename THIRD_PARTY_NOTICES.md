# Third-Party Software and Copyright Notes

更新日: 2026-09-20

このリポジトリ固有のアプリケーションコードは、外部の類似製品またはOSSのソースコードを複製せず、このプロジェクト向けに実装しています。設計検討では公開されている製品説明、論文、ドキュメントから一般的な構成と要件を参照しています。

SoccerNet、Roboflow Sports、Veo、Metrica Sportsのソースコード、画像、文章、UI素材、モデル重みをこのリポジトリへ転載していません。

## 直接依存するソフトウェア

以下は開発環境で確認したバージョンです。依存関係を更新した場合は、配布前に再確認してください。

| Package | Version | License | Usage |
| --- | --- | --- | --- |
| FastAPI | 0.136.1 | MIT | Web API |
| Pydantic | 2.13.4 | MIT | API data validation |
| Uvicorn | 0.47.0 | BSD-3-Clause | ASGI server |
| python-multipart | 0.0.29 | Apache-2.0 | Video upload parsing |
| OpenCV Python | 4.13.0.92 | Apache-2.0 | Video I/O and drawing |
| NumPy | 2.4.6 | Multiple permissive licenses | Numerical processing |
| Ultralytics | 8.4.149 | AGPL-3.0 or commercial license | YOLO inference, training and tracking API |
| lap | 0.5.13 | BSD-2-Clause | Assignment solver used by tracking |

## Ultralyticsに関する重要事項

現在の実装は`ultralytics`パッケージとYOLOモデルを使用します。Ultralyticsの公式説明では、オープンソース利用はAGPL-3.0、非公開・商用・SaaSなどの利用はEnterprise License等の商用ライセンスが選択肢として示されています。

- https://www.ultralytics.com/license
- https://www.ultralytics.com/legal/agpl-3-0-software-license

本アプリを非公開サービス、社内システム、スマートフォン製品または商用サービスとして提供する前に、利用形態がライセンス条件を満たすか確認してください。AGPL-3.0の条件で提供しない場合は、Ultralyticsの商用ライセンス取得、または互換ライセンスの別モデル・推論基盤への置換を検討してください。

## OpenCVとFastAPI

- OpenCV 4.5.0以降はApache-2.0: https://opencv.org/license/
- FastAPIはMIT: https://fastapi.tiangolo.com/

各パッケージの完全なライセンス本文と著作権表示は、配布物に含まれるパッケージの`LICENSE`、`METADATA`、公式配布元を参照してください。バイナリやコンテナを配布する場合は、依存パッケージが要求するライセンス本文とNOTICEを配布物へ同梱してください。

## 外部データとモデル

試合動画、学習画像、アノテーション、事前学習済みモデルには、ソフトウェアとは別の利用条件が存在します。

- 自分が撮影した動画でも、選手・観客の肖像、会場規約、チームの同意を確認する。
- 外部データセットは、商用利用、再配布、派生モデルに関する条件を個別に記録する。
- データセットごとに出典、バージョン、取得日、ライセンス、同意範囲を台帳化する。
- ライセンス不明の動画や画像を学習・デモ・配布物へ混在させない。

## このリポジトリ自体のライセンス

現時点では、このリポジトリ固有コードに対するルート`LICENSE`は選択されていません。公開・共同開発・商用提供を行う前に、権利者がプロジェクトライセンスを決定する必要があります。Ultralyticsを継続利用する場合、その選択はUltralytics側のライセンス条件と整合させてください。

この文書は開発上の確認事項であり、法的助言ではありません。
