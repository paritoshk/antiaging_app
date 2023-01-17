FROM python:3.10.8-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]