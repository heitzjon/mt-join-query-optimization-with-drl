import psycopg2
from queryoptimization.QueryGraph import Query, Relation

"""
    cm1(): 
        calculates the cost of the given query object.
    card and card_nochache():
        gets the cardinality estimations from PostgreSQL.
"""
̣__author__ = "Jonas Heitz"

cardinality = {}

def cm1(query,cursor,lambda_factor=1):
    r = 0.2

    # cost of table scan (1 Relation)
    if type(query) is Relation:
        costs = r * card(query,cursor)

    # cost of index join
    elif (type(query.right) is Relation) and (any({*query.right.indices} & {*query.join_condition})): # checks if query.right.indices as a mutual element with query.join_condition
        costs = cm1(query.left,cursor) + lambda_factor * card(query.left,cursor) * max(
            card(query,cursor) / card(query.left,cursor), 1)

    # cost of cross join
    elif len(query.join_condition) is 0:
        costs = cm1(query.left,cursor) * cm1(query.right,cursor)+card(query.left,cursor) * card(query.right,cursor)

    # cost of hash join
    else:
        costs = card(query,cursor) + cm1(query.left,cursor) + cm1(query.right,cursor)
    return costs


def card_nochache(query,cursor):
    #query.toSql(0)
    cursor.execute("""EXPLAIN """ + query.toSql(0))
    rows = cursor.fetchall()
    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")
    return float(estimatedRows)

#caches already used cardinalities
def card(query,cursor):
    global cardinality
    #print(query.name)
    if query.name in cardinality and ((type(query) is Relation) or (''.join(sorted(query.joined_columns)) in cardinality[query.name])):
        if type(query) is Relation:
            return int(cardinality[query.name]['card'])
        elif ''.join(sorted(query.joined_columns)) in cardinality[query.name]:
            return int(cardinality[query.name][''.join(sorted(query.joined_columns))]['card'])
    else:

        try:
            cursor.execute("""EXPLAIN """ + query.toSql(0))
            rows = cursor.fetchall()
            row0 = rows[0][0].split("(cost=")[1].split(' ')
            estimatedRows = row0[1].replace("rows=", "")
        except:
            print(query.toSql(0))

        if query.name not in cardinality:
            cardinality[query.name] = {}
        if type(query) is Relation:
            cardinality[query.name]['card'] = estimatedRows
        else:
            cardinality[query.name][''.join(sorted(query.joined_columns))]={'card':estimatedRows}
                                                                         
        return float(estimatedRows)
