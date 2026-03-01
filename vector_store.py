import sqlite3
import sqlite_vec

def InitializeDB():
    conn = sqlite3.connect("memory.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories USING vec0(\
        vector FLOAT[1536],\
        plain_text TEXT,\
        category TEXT)\
    ")
    conn.commit()
    
    return conn, cursor