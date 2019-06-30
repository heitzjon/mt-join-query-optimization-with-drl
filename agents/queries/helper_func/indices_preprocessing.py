import csv

# Open the CSV
f = open('indices.txt', 'rU')
# Change each fieldname to the appropriate field name. I know, so difficult.
reader = csv.DictReader(f)#, fieldnames=("schema", "name", "type", "owner", "table"))

keys = []
for r in reader:
    keys.append(r["table_name"].replace("_","") + "." + r["column_name"])
    keys.append(r["table_name"].replace("_","") + "2." + r["column_name"])
    print(r["table_name"].replace("_","") + "." + r["column_name"])
print(keys)

'''
select
    t.relname as table_name,
    i.relname as index_name,
    a.attname as column_name
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
order by
    t.relname,
    i.relname;
'''