run_server:
	uvicorn src.scratch.main:app --reload

run_client:
	streamlit run streamlit_frontend.py --server.port 30009 --server.fileWatcherType none --browser.gatherUsageStats False

run_gateway_server:
	uvicorn src.redis_celery.main:app --reload

run_gateway_client:
	streamlit run src/redis_celery/streamlit_frontend.py --server.fileWatcherType none --browser.gatherUsageStats False

run_app: run_server run_client

run_gateway_app: run_gateway_server run_gateway_client
