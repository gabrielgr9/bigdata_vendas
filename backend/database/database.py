import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres123@postgres:5432/projeto_vendas"
)

engine = create_engine(DATABASE_URL)