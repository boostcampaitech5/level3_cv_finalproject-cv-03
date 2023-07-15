run_server:
	uvicorn main:app --reload

run_client:
	streamlit run streamlit_frontend.py --server.port 30009 --server.fileWatcherType none --browser.gatherUsageStats False

run_app: run_server run_client
