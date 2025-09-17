FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

RUN pip install --no-cache-dir flask requests curl_cffi werkzeug loguru python-dotenv playwright

RUN python -m playwright install --with-deps chrome

COPY . .

ENV PORT=5200
ENV PYTHONUNBUFFERED=1

EXPOSE 5200

CMD ["python", "app.py"]