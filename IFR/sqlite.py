import sqlite3
from sqlite3 import Error

# global variables
sqlite_db = 'loki.sqlite'

person_table_def =   """CREATE TABLE IF NOT EXISTS Person (
                        person_id   INTEGER     NOT NULL    PRIMARY KEY,    -- unique ID
                        name        TEXT        NOT NULL,
                        note        TEXT,                                   -- optional
                        );"""

face_representation_table_def =   """CREATE TABLE IF NOT EXISTS FaceRepresentation (
                        rep_id      INTEGER     PRIMARY KEY,    -- unique ID
                        person_id   INTEGER     NOT NULL        -- join to Person table
                        image_name  TEXT        NOT NULL,       -- original image's filename
                        image_path  TEXT        NOT NULL,       -- oroginal LOCAL image path
                        region      TEXT        NOT NULL,       -- list of 4 integers specifying the rectangular face's corners inside the image
                        embedding   TEXT        NOT NULL,       -- vector representation 
                        FOREIGN KEY (rep_id)    REFERENCES Person(person_id)
                        );"""


def create_connection(db_file):
    """
    create a database connection to a SQLite database
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)
    # finally:
    #     if conn:
    #         conn.close()


def init_db():
    """
    init the DB (if it doesn't exist)
    and create the table structures
    """
    conn = create_connection(sqlite_db)
    cursor = conn.cursor()
    cursor.execute(person_table_def)
    cursor.execute(face_representation_table_def)
    return conn


def close_db(conn):
    conn.close()


def person_add(conn, person):
    """
    Add a new person into the Person table
    :param conn:
    :param person:
    :return: person_id
    """
    sql = ''' INSERT INTO Person(person_id,name,note)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, person)
    conn.commit()
    return cur.lastrowid

def person_remove(conn, person_id):
    """
    Add a new person into the Person table
    :param conn:
    :param person_id:
    """
    sql = 'DELETE FROM Person WHERE person_id = ?'
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()

def person_update_name(conn, name, person_id):
    """
    Update an existing person's name
    :param conn:
    :param name:
    :param person_id:
    """
    sql = '''   UPDATE Person
                SET name = ?
                WHERE person_id = ?
        '''
    cur = conn.cursor()
    cur.execute(sql, name, person_id)
    conn.commit()