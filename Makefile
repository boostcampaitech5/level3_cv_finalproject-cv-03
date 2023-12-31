run_server:
	uvicorn src.scratch.main:app --host 0.0.0.0 --port 8001 --reload

nohup_run_server:
	nohup uvicorn src.scratch.main:app --host 0.0.0.0 --port 8001 --reload > ./FastAPI_Uvicorn.log 2>&1 &

nohup_run_dreambooth_worker:
	nohup celery -A src.scratch.worker_dreambooth.dream_app worker -l info -E -Q dreambooth --concurrency=8 --prefetch-multiplier 1 > ./celery_worker_dreambooth.log 2>&1 &

nohup_run_sdxl_worker:
	nohup celery -A src.scratch.worker_sdxl.celery_app worker -l info -E -Q sdxl --concurrency=8 --prefetch-multiplier 1 > ./celery_worker_sdxl.log 2>&1 &

nohup_run_sd_worker:
	nohup celery -A src.scratch.worker_sd.celery_app worker -l info -E > ./celery_worker_sd.log 2>&1 &
