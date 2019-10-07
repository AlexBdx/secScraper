import psycopg2
from tqdm import tqdm
import ast
from datetime import datetime

def delete_table(connector, name_table):
    cur = connector.cursor()
    deletion_request = "DROP TABLE IF EXISTS {};".format(name_table)
    cur.execute(deletion_request)
    connector.commit()
    print("[INFO] Deleted table", name_table)


# Create the header of the table and specify what goes in it
def create_postgres_table(connector, name_table, header):
    create_table = "CREATE TABLE {}".format(name_table)
    sql_header = "(IDX integer PRIMARY KEY,"
    for column in header:
        sql_header += "{} {},".format(*column)  # Name type
    sql_header = sql_header[:-1] + ')'
    create_table += sql_header
    
    print("[INFO] Creating the following table:")
    print(create_table)
    
    cur = connector.cursor()
    cur.execute(create_table)
    connector.commit()


def insert_row(connector, name_table, row):
    sql_query = "INSERT INTO {} VALUES (".format(name_table)
    for element in row:
        sql_query += "%s, "
    sql_query = sql_query[:-2]
    sql_query += ")"
    # print(sql_query)
    cur = connector.cursor()
    cur.execute(sql_query, row)
    connector.commit()


def settings_to_postgres(connector, s):
    delete_table(connector, 'settings')
    create_postgres_table(connector, 'settings', [('KEY', 'text'), ('VALUE', 'text')])
    idx = 0
    for k, v in tqdm(s.items()):
        row = [idx, k, str(v)]
        insert_row(connector, 'settings', row)
        idx += 1


def pf_values_to_postgres(connector, pf_values, header, s):
    delete_table(connector, 'pf_values')
    create_postgres_table(connector, 'pf_values', header)
    idx = 0
    for m in tqdm(s['metrics'][:-1]):
        for l in s['bin_labels']:
            for qtr in s['list_qtr'][s['lag']:]:
                insert_row(connector, 'pf_values', [idx, m, qtr, l, *pf_values[m][l][qtr]])
                idx += 1


def lookup_to_postgres(connector, lookup, header):
    delete_table(connector, 'lookup')
    create_postgres_table(connector, 'lookup', header)
    idx = 0
    for k, v in tqdm(lookup.items()):
        row = [idx, k, str(v)]  # Technically, v is always an int
        insert_row(connector, 'lookup', row)
        idx += 1


def cik_scores_to_postgres(connector, cik_scores, header, s):
    delete_table(connector, 'cik_scores')
    create_postgres_table(connector, 'cik_scores', header)
    idx = 0
    for cik in tqdm(cik_scores.keys()):
        for qtr in s['list_qtr'][s['lag']:]:
            for m in s['metrics']:
                try:
                    md = cik_scores[cik][qtr]['0']  # Metadata
                    insert_row(connector, 'cik_scores', 
                               (idx, cik, qtr, m, cik_scores[cik][qtr][m], md['type'], md['published']))
                    idx += 1
                except KeyError:  # There is no data for this qtr, CIK not listed/delisted
                    continue

def csv_to_postgres(connector, table_name, header, path):
    delete_table(connector, table_name)
    create_postgres_table(connector, table_name, header)
    with open(path, 'r') as f:
        cur = connector.cursor()
        next(f) # Skip the header row.
        cur.copy_from(f, table_name, sep=';')
        connector.commit()


# Build the plot based on a PostGres query
def retrieve_pf_values(connector, table_name, s):
    sql_query = "SELECT * FROM {};".format(table_name)
    print(sql_query)
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    
    # Re-initialize pf_values
    pf_values = {m: 0 for m in s['metrics'][:-1]}
    for m in s['metrics'][:-1]:
        pf_values[m] = {q: {qtr: [0, s['tax_rate'], 0] for qtr in s['list_qtr'][1:]} for q in s['bin_labels']}
    
    # Re-build pf_values knowing their type - hacky
    for e in data:
        pf_values[e[1]][e[3]][ast.literal_eval(e[2])] = [*e[4:]]
    return pf_values


def retrieve_settings(connector):
    sql_query = "SELECT * FROM settings;"
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    
    # Re-build the dictionary
    s = {e[1]: e[2] for e in data}
    print("[WARNING] Verify that the following are meant to stay str:")
    for k in s:
        try:
            s[k] = ast.literal_eval(s[k])
        except (SyntaxError, ValueError):
            print(k, ":", s[k])
    return s


def retrieve_lookup(connector):
    sql_query = "SELECT * FROM lookup;"
    print(sql_query)
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    
    lookup = {e[1]: e[2] for e in data}
    reverse_lookup = {e[2]: e[1] for e in data}
    return lookup, reverse_lookup


def retrieve_cik_scores(connector, cik, s):
    sql_query = "SELECT * FROM cik_scores WHERE cik = '{}';".format(cik)
    print(sql_query)
    
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    #print(data)
    # Initialize
    result = {cik: {qtr: {} for qtr in s['list_qtr']}}
    for e in data:
        result[cik][ast.literal_eval(e[2])][e[3]] = e[4]
        result[cik][ast.literal_eval(e[2])]['0'] = {
            'type': e[5],
            'published': e[6],
            'qtr': ast.literal_eval(e[2])
        }
    return result


def retrieve_all_stock_data(connector, table_name):
    sql_query = "SELECT * FROM {};".format(table_name)
    print(sql_query)
    
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    
    # Rebuild stock_data
    stock_data = dict()
    for e in tqdm(data):
        try:
            stock_data[e[1]][e[2]] = [*e[3:]]
        except KeyError:  # That ticker does not exist yet
            stock_data[e[1]] = dict()
            stock_data[e[1]][e[2]] = [*e[3:]]
        
    return stock_data

def retrieve_stock_data(connector, ticker):
    sql_query = "SELECT * FROM stock_data WHERE ticker = '{}';".format(ticker)
    print(sql_query)
    
    cur = connector.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    #print(data)
    # Initialize
    result = {cik: {qtr: {} for qtr in s['list_qtr']}}
    for e in data:
        result[cik][ast.literal_eval(e[2])][e[3]] = e[4]
        result[cik][ast.literal_eval(e[2])]['0'] = {
            'type': e[5],
            'published': e[6],
            'qtr': ast.literal_eval(e[2])
        }
    return result
    
    
    
def does_ticker_exist(connector, ticker):
    """
    Check if a given ticker exists in the stock database.
    
    """
    cur = connector.cursor()
    cur.execute("SELECT TICKER FROM stock_data WHERE TICKER = %s", (ticker,))
    return cur.fetchone() is not None
