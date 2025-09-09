up:
	docker-compose up --build

down:
	docker-compose down

logs:
	docker-compose logs -f

api:
	docker-compose exec api sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000"
