run_server:
	uvicorn src.scratch.main:app --reload

run_client:
	streamlit run src/scratch/streamlit_frontend.py --server.port 30009 --server.fileWatcherType none --browser.gatherUsageStats False

run_app: run_server run_client
