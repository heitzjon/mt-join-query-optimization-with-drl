import numpy as np
join_conditions = {}

"""
    Query_Init:
        Parses simple join queries into the object structure for the gym environment
        
    Relation:
        Defines the functions and abilities of a base relation
    
    Empty_Query:
        Is a placeholder for the empty spots during the query optimization process in the gym environment
        
    Query:
        Defines the functions and abilities of the queries and subqueries
        
"""
̣__author__ = "Jonas Heitz"

class Query_Init(object):
    actions = []
    global join_conditions
    join_conditions = {}
    mask = []

    def __init__(self, sqlquery, schema, indices):
        global join_conditions
        self.query = object

        schema_one_hot = np.zeros((len(schema),sum(len(col) for val, col in schema.items())), dtype=int)
        relations = {}
        pointer = 0
        for val, col in schema.items():
            relations[val] = list(schema_one_hot[0].copy())
            for i in range(0, len(col)):
                relations[val][pointer] = 1
                pointer += 1


        self.actions, join_conditions = self.sqlToActionSpace(sqlquery, indices, relations, schema)
        for i in range(len(self.actions),len(list(schema_one_hot))):
            self.actions.append(EmptyQuery(schema_one_hot[0]))


        #sorted action space
        sorted_actions = []
        for key,value in schema.items():
            flag = True
            for action in self.actions:
                name = action.name
                if " AS " in name:
                    name = name.split(" AS ")[1]
                if name == key.replace("_",""):
                    sorted_actions.append(action)
                    flag = False
                    break
            if flag: sorted_actions.append(EmptyQuery(schema_one_hot[0]))
        self.actions = sorted_actions




    def sqlToActionSpace(self,sql,indices,masks, schema):
        action_space = []
        join_conditions = {}
        try:
            relations = sql.split('FROM')[1].split('WHERE')[0].replace(" ", "").split(',')
        except Exception:
            relations = sql.split('FROM')[1].replace(" ", "")
            pass
        for r in relations:
            r = r.replace("AS"," AS ")
            r_split = r.split(" AS ")
            action_space.append(Relation(r,indices,masks[r_split[0]],schema[r_split[0]]))

        try:
            all_conditions = sql.split('WHERE')[1].split('AND')
        except Exception:
            all_conditions = []
            pass

        for condition in all_conditions:
            if "=" in condition:
                clauses = condition.split("=")
                element = []
                for clause in clauses:
                    element.append(clause.replace("\n", "").replace(" ", "").split("."))
                try:
                    cond = [element[0][0].replace('_','') + '.' + element[0][1], element[1][0].replace('_','') + '.' + element[1][1]]
                    join_conditions['_'.join(sorted([element[0][0].replace('_','')]+[element[1][0].replace('_','')]))] = cond

                except Exception:
                    pass

        return action_space, join_conditions

class Relation(object):
    name = ''
    indices = []
    joined_columns = []
    mask = []
    columns = []

    def __init__(self, name,indices,mask,columns):
        self.name = name
        self.indices = []
        self.mask = mask
        self.columns = []
        if " AS " in self.name:
            self.name = self.name.split(" AS ")[0]+" AS "+self.name.split(" AS ")[1].replace('_','')
        elif "_" in self.name:
            self.name = self.name+" AS "+self.name.replace('_','')
        for column in columns:
            if " AS " in self.name:
                self.columns.append(self.name.split(" AS ")[1] + "." + column)
            else:
                self.columns.append(name+"."+column)
        for i in indices:
            if " AS " in self.name:
                table = self.name.split(" AS ")[1]
            else:
                table = self.name
            if i.split('.')[0] == table:
                self.indices.append(i)

    def toSql(self,level):
        return "SELECT * FROM "+self.name+";"
    def __str__(self):
        return self.name


class EmptyQuery(object):
    name = 'EmptyQuery'
    mask = []
    def __init__(self, mask):
        self.mask = mask
    def __str__(self):
        return 'empty'



class Query(object):
    left = None
    right = None
    name = ''
    join_condition = {}
    joined_columns = []
    mask =  []
    columns = []
    aliasflag = True#False

    def __init__(self, left, right):
        global join_conditions
        self.joined_columns = []

        self.left = left
        self.right = right
        if " AS " in self.left.name:
            lname = self.left.name.split(" AS ")[1]
            lname_list = [lname]
        else:
            lname = self.left.name
            lname_list = lname.split('_')
        if " AS " in self.right.name:
            rname = self.right.name.split(" AS ")[1]
            rname_list = [rname]
        else:
            rname = self.right.name
            rname_list = rname.split('_')
        '''
        if type(self.left) is Relation:
            lname=lname.replace("_","")
        if type(self.right) is Relation:
            rname = rname.replace("_","")
        '''
        self.name = '_'.join(sorted(lname_list+rname_list))
        self.mask = [x | y for (x, y) in zip(left.mask, right.mask)]

        if self.name in join_conditions:
            self.join_condition = join_conditions[self.name]
            for i in self.join_condition:
                self.joined_columns.append(i.split('.')[1])
        if type(self.left) is Query:
            self.joined_columns=self.joined_columns+self.left.joined_columns
        if type(self.right) is Query:
            self.joined_columns=self.joined_columns+self.right.joined_columns

        self.columns=[]
        tmpcolumns = []
        for c in left.columns:
            if " AS " in c:
                self.columns.append(lname+"."+c.split(" AS ")[1])
                tmpcolumns.append(c.split(" AS ")[1])
            else:

                self.columns.append(lname+"."+c.split(".")[1])
                tmpcolumns.append(c.split('.')[1])
        for c in right.columns:
            if " AS " in c:
                c = rname+"."+c.split(" AS ")[1]
            if c.split('.')[1] in tmpcolumns:
                new_column = rname+"."+str(c.split('.')[0].split("_")[0]+c.split('.')[0].split("_")[-1])+"_"+c.split('.')[1] #hack to keep columns length short
                while new_column.split('.')[1] in tmpcolumns:
                    new_column=new_column+"_tmp"
                tmpcolumns.append(new_column.split('.')[1])
                self.columns.append(rname+"."+c.split('.')[1]+" AS "+new_column.split('.')[1])
                for key, val in join_conditions.items():
                    newval=[]
                    for v in val:
                        if v == rname+"."+c.split('.')[1]:
                            newval.append(v.replace(rname+"."+c.split('.')[1],new_column))
                            if "kind_type" in new_column: print(new_column)
                        else:
                            newval.append(v)
                    join_conditions[key]=newval
            else:
                self.columns.append(rname+"."+c.split(".")[1])
                tmpcolumns.append(c.split('.')[1])

        join_conditions = self.deleteJoinCondition(lname, rname, join_conditions)
        join_conditions = self.changeJoinConditions(lname, rname, self.name, join_conditions)


    def __str__(self):
        return '(' + self.left.__str__() + ' ' + self.right.__str__() + ')'

    def getJoinConditions(self, relA, relB, join_conditions):
        try:
            condition = join_conditions[relA + ":" + relB]

        except Exception:
            condition = None
            pass
        return condition

    def changeJoinConditions(self, relA, relB, relnew, join_conditions):
        conditions = {}
        if "_" in relB:
            relB = relB.split('_')
        else:
            relB = [relB]
        if "_" in relA:
            relA = relA.split("_")
        else:
            relA = [relA]
        for key, value in join_conditions.items():
            if set(relB).issubset(key.split('_')):
                new_key = '_'.join(np.unique(sorted(key.split('_')+relnew.split('_'))))
                value2 = []
                for v in value:
                    #value2.append('_'.join(np.unique(sorted(v.split('.')[0].replace(relB,relnew).split('_'))))+'.'+v.split('.')[1])
                    if set(relB).issubset(v.split('.')[0].split('_')):
                        value2.append('_'.join(np.unique(sorted(v.split('.')[0].split('_')+relnew.split('_'))))+ '.' +v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                else:
                    conditions[new_key] = value2
            elif set(relA).issubset(key.split('_')):
                #new_key = '_'.join(np.unique(sorted(key.replace(relA, relnew).split('_'))))
                new_key = '_'.join(np.unique(sorted(key.split('_') + relnew.split('_'))))
                value2 = []
                for v in value:
                    #value2.append('_'.join(np.unique(sorted(v.split('.')[0].replace(relA, relnew).split('_'))))+'.'+v.split('.')[1])
                    if set(relA).issubset(v.split('.')[0].split('_')):
                        value2.append('_'.join(np.unique(sorted(v.split('.')[0].split('_')+relnew.split('_'))))+ '.' +v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                else:
                    conditions[new_key] = value2
            else:
                if key in conditions:
                    conditions[key] = conditions[key] + value
                else:
                    conditions[key] = value
        return conditions

    def deleteJoinCondition(self, relA, relB, jc):
        conditions = dict(jc)
        try:
            del conditions['_'.join(np.unique(sorted(relA.split('_') + relB.split('_'))))]
            #del conditions[relB + relA]
        except Exception:
            pass
        return conditions

    def toSql(self,level):
        if self.aliasflag:
            sql = 'SELECT '
            i = False
            for c in self.columns:
                if i:
                    sql+=", "
                i = True
                #sql+=self.name+"."+c.split('.')[1]
                sql += c
            sql+=' FROM '

        else:
            sql = 'SELECT * FROM '

        if type(self.left) is Relation and type(self.right) is Relation:
            subsql_left = self.left.name
            subsql_right = self.right.name

        elif type(self.left) is Relation and type(self.right) is Query:
            subsql_left = self.left.name
            subsql_right = self.right.toSql(1)
           # subsql_right = '(' + subsql_right + ')'

        elif type(self.left) is Query and type(self.right) is Relation:
            subsql_left = self.left.toSql(1)
            subsql_right = self.right.name
            #subsql_left = '(' + subsql_left + ')'

        elif type(self.left) is Query and type(self.right) is Query:
            subsql_left = self.left.toSql(1)
            subsql_right = self.right.toSql(1)
            #subsql_left = '(' + subsql_left + ')'
            #subsql_right = '(' + subsql_right + ')'
        else:
            return ""

        if len(self.join_condition) is not 0:
            sql_join_condition = self.join_condition[0] + '=' + self.join_condition[1]
            if len(self.join_condition) > 2:
                for i in range(2,len(self.join_condition),2):
                    sql_join_condition = sql_join_condition+' AND '+self.join_condition[i]+'='+self.join_condition[i+1]
            sql += subsql_left + ' INNER JOIN ' + subsql_right + ' ON (' + sql_join_condition + ')'
        else:
            sql += subsql_left + ' CROSS JOIN ' + subsql_right

        if level is 1: sql = '('+sql+') AS ' + self.name + ' '

        return sql

def getJoinConditions():
    return join_conditions
