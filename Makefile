run_server:
	uvicorn src.scratch.main:app --host 0.0.0.0 --port 8001 --reload
