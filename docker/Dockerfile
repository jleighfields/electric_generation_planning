FROM python:3.7
WORKDIR /electric_generation_planning
COPY requirements.txt requirements.txt
COPY ./src ./src
COPY ./csv ./csv


RUN python3 -m venv venv
RUN . venv/bin/activate
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["src/app.py"]
