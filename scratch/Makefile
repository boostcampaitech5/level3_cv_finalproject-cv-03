run_server:
	uvicorn fastapi-server:app --reload

run_client:
	streamlit run streamlit-frontend-kr.py --server.port 30009

run_app: run_server run_client
