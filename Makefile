# 定点観測ストレス評価スクリプトの開発用ショートカット
# 使い方: make run / make clean / make clean-profiles / make clean-all / make show-profiles
# ※ Windows で make が無い場合は run.ps1 を使う: .\run.ps1 <target>

PYTHON ?= python

.PHONY: run clean clean-profiles clean-all show-profiles help

help:
	@echo "targets: run | clean | clean-profiles | clean-all | show-profiles"

run:
	$(PYTHON) peroson_detect.py

# セッション出力（ログ・グラフ・キャッシュ）だけ消す
# 削除は rm 非依存にするため python 経由（Windowsのmake環境でも動くように）
clean:
	-$(PYTHON) -c "import os,shutil;[os.remove(f) for f in ('stress_log.csv','stress_graph.png') if os.path.exists(f)];shutil.rmtree('__pycache__',ignore_errors=True)"

# 学習した平常状態・人物ID（定点観測データ）を消す＝開発中のリセット
clean-profiles:
	-$(PYTHON) -c "import os;[os.remove(f) for f in ('person_profiles.json','session_history.jsonl') if os.path.exists(f)]"

# すべて消す
clean-all: clean clean-profiles

# 保存済みプロファイルを整形表示
show-profiles:
	$(PYTHON) -c "import json;print(json.dumps(json.load(open('person_profiles.json',encoding='utf-8')),indent=2,ensure_ascii=False))"
