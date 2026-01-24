# =========================
# Pipeline: Steps 01 to 06
# =========================

.PHONY: all clean step01 step02 step03 step04 step05 step06

# Default target
all: step06

step01:
	@echo "Running Step 01: Txt to JSON conversion"
	python 01_txt_to_json_all.py

step02: step01
	@echo "Running Step 02: TF-IDF signal extraction"
	python 02_tfidf_event_signals.py

step03: step02
	@echo "Running Step 03: BERT topic modeling"
	python 03_bert_topic.py

step04: step03
	@echo "Running Step 04: Multi-company event alignment"
	python 04_align_events_multi_company.py

step05: step04
	@echo "Running Step 05: Model training"
	python 05_train_model.py

step06: step05
	@echo "Running Step 06: Logistic Regression event model"
	python 06_logistic_regression_event_model.py