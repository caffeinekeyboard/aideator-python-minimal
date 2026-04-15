.PHONY: dev-api dev-web install-api install-web test

install-api:
	./venv/bin/pip install -r requirements.txt -r application/api/requirements-api.txt

install-web:
	cd application/frontend && npm install

dev-api:
	./venv/bin/uvicorn application.api.main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	cd application/frontend && npm run dev

test:
	./venv/bin/pytest tests/ -q
