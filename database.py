import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from contextlib import contextmanager
import json

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode={os.getenv('DB_SSLMODE')}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

@contextmanager
def get_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    return db

def close_db(db):
    try:
        if db:
            db.close()
    except Exception as e:
        print(f"Error closing database session: {e}")
        # Force close if normal close fails
        try:
            db.bind.dispose()
        except:
            pass

def get_all_tables():
    query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")

    result = db.execute(query)
    tables = [row[0] for row in result]

def get_table_columns(db, table_name):
    query = text(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = :table_name;
    """)
    
    result = db.execute(query, {"table_name": table_name})
    columns = [{"name": row[0], "type": row[1]} for row in result]

def get_table_data(db, table_name, id):
    if id is None:
        query = text(f"SELECT * FROM {table_name};")
    else:
        query = text(f"SELECT * FROM {table_name} WHERE ID = {id};")
    
    result = db.execute(query)

    try:
        return result.fetchall()
    except:
        return "Executed without results"

def clear_table(db, table_name, id):
    if id is None:
        query = text(f"DELETE FROM {table_name};")
    else:
        query = text(f"DELETE FROM {table_name} WHERE id = {id};")
    
    result = db.execute(query)
    db.commit()
    
    try:
        return result.fetchall()
    except:
        return "Executed without results"

def remove_one_entry(db, table_name, entry_id):
    query = text(f"DELETE FROM {table_name} WHERE id = :entry_id;")
    
    db.execute(query, {"entry_id": entry_id})
    db.commit()

def create_new_table(db, table_name, columns):
    column_definitions = ", ".join([f"{col['name']} {col['type']}" for col in columns])
    query = text(f"CREATE TABLE {table_name} ({column_definitions});")
    
    db.execute(query)
    db.commit()

def drop_table(db, table_name):
    query = text(f"DROP TABLE IF EXISTS {table_name};")
    
    db.execute(query)
    db.commit()

def insert_into_table(db, table_name, data, return_id):
    try:
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{key}" for key in data.keys()])
        query = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING ID;")
        
        result = db.execute(query, data)
        
        if return_id:
            row = result.fetchone()
            db.commit()
            return row[0] if row else None
        else:
            db.commit()
            return None
    except Exception as e:
        try:
            db.rollback()
        except:
            pass
        raise e

def run_custom_query(db, query):
    query = text(query)

    result = db.execute(query)

    db.commit()

    try:
        return result.fetchall()
    except:
        return "Executed without results"

functions = {
    "get_all_tables": get_all_tables,
    "get_table_columns": get_table_columns,
    "create_new_table": create_new_table,
    "drop_table": drop_table,
    "get_db": get_db,
}