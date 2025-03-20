FROM python:3.9-slim
LABEL authors="albus"

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir flask paddlex PyMuPDF
EXPOSE 5000
CMD ["python", "app.py"]