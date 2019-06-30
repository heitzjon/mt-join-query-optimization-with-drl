import gym
import psycopg2
import sys
import time

from queryoptimization.QueryGraph import Query_Init, Query, Relation
from queryoptimization.cm1_postgres_card import cm1, cardinality
from itertools import permutations, combinations

"""
    Tradition Query Optimization Algorithms:
        DP left-deep, DP right-deep and greedy selection.
        For comparison of DRL approaches

"""
̣__author__ = "Jonas Heitz"


def dynamic_programming_left_deep(sql, schema, primary,lambda_f=1):
    try:
        conn = psycopg2.connect(host="localhost", database="imdbload", user="postgres", password="admin")
        #conn = psycopg2.connect(host="localhost", database="imdbload", user="docker", password="docker")
    except:
        print("I am unable to connect to the database")
    #print(query)
    cursor = conn.cursor()

    query_env = Query_Init(sql, schema, primary)
    num_of_checks=0

    actions = []
    for rel in query_env.actions:
        if type(rel) is Relation:
            actions.append(rel)
    query_env.actions = actions

    #Step 1
    # generate all possible 2-way joins
    # select cheapest of same outcome
    queries={}
    for query_list in permutations(query_env.actions,2):
        num_of_checks+=1
        query = Query(query_list[0], query_list[1])
        if query.join_condition is not []: #no cross-join rule!
            cost = cm1(query,cursor,lambda_f)
            query_name = query.name
            Query_Init(sql, schema, primary) #to reset global variables
            if query_name in queries:
                if cost<queries[query_name]['cost']:
                    queries[query_name]['cost']=cost
                    queries[query_name]['rel']=[query_list[0],query_list[1]]
                    queries[query_name]['obj'] = query
            else:
                queries[query_name]={}
                queries[query_name]['cost'] = cost
                queries[query_name]['rel'] = [query_list[0], query_list[1]]
                queries[query_name]['obj'] = query

    for i in range(1,len(query_env.actions)-1):
        new_queries={}
        print("STEP: "+str(i)+" | #candidates: "+str(len(queries)))
        #Step 2-x
        # generate (3-x)-way joins
        # throw away equivalent queries which are more expensive
        for relation in query_env.actions:
            #print("rel: "+str(relation))
            #print("_______")
            for key, subquery in queries.items():
                #for x in subquery['rel']: print(x)
                #print("_______")
                if relation not in subquery['rel']:
                    num_of_checks += 1
                    query=Query(subquery['rel'][0],subquery['rel'][1])
                    for j in range(2,len(subquery['rel'])):
                        query=Query(query,subquery['rel'][j])
                    query = Query(query,relation)
                    if query.join_condition is not []:  #True: ## no cross-join rule!
                        cost=cm1(query,cursor,lambda_f)
                        query_name = query.name
                        Query_Init(sql, schema, primary)  # to reset global variables
                        #print(query_name,cost)
                        if query_name in new_queries:
                            if cost < new_queries[query_name]['cost']:
                                new_queries[query_name]['cost'] = cost
                                new_queries[query_name]['rel'] = subquery['rel']+[relation]
                                new_queries[query_name]['obj'] = query
                        else:
                            new_queries[query_name]={}
                            new_queries[query_name]['cost'] = cost
                            new_queries[query_name]['rel'] = subquery['rel']+[relation]
                            new_queries[query_name]['obj'] = query
        queries = new_queries
    cursor.close()
    print("STEP: " + str(i+1) + " | #candidates: " + str(len(queries)))
    for key, val in queries.items():
        display('dynamic prog. left deep', val['obj'].__str__(), num_of_checks, val['cost'])
        return [val['obj'].__str__(), num_of_checks, val['cost']]


def dynamic_programming_right_deep(sql, schema, primary,lambda_f=1):
    try:
        conn = psycopg2.connect(host="localhost", database="imdbload", user="postgres", password="admin")
    except:
        print("I am unable to connect to the database")
    # print(query)
    cursor = conn.cursor()

    query_env = Query_Init(sql, schema, primary)
    num_of_checks = 0

    actions = []
    for rel in query_env.actions:
        if type(rel) is Relation:
            actions.append(rel)
    query_env.actions = actions
    # Step 1
    # generate all possible 2-way joins
    # select cheapest of same outcome
    queries = {}
    for query_list in permutations(query_env.actions, 2):
        num_of_checks += 1
        query = Query(query_list[0], query_list[1])
        if query.join_condition is not []:  # no cross-join rule!
            cost = cm1(query, cursor,lambda_f)
            query_name = query.name
            Query_Init(sql, schema, primary)  # to reset global variables
            if query_name in queries:
                if cost < queries[query_name]['cost']:
                    queries[query_name]['cost'] = cost
                    queries[query_name]['rel'] = [query_list[0], query_list[1]]
                    queries[query_name]['obj'] = query
            else:
                queries[query_name] = {}
                queries[query_name]['cost'] = cost
                queries[query_name]['rel'] = [query_list[0], query_list[1]]
                queries[query_name]['obj'] = query

    for i in range(1, len(query_env.actions) - 1):
        print("STEP: " + str(i) + " | #candidates: " + str(len(queries)))
        new_queries = {}
        # Step 2-x
        # generate (3-x)-way joins
        # throw away equivalent queries which are more expensive
        for relation in query_env.actions:
            for key, subquery in queries.items():
                if relation not in subquery['rel']:
                    num_of_checks += 1
                    query = Query(subquery['rel'][0], subquery['rel'][1])
                    for j in range(2, len(subquery['rel'])):
                        query = Query(subquery['rel'][j],query)
                    query = Query(relation, query)
                    if query.join_condition is not []:  # no cross-join rule!
                        cost = cm1(query, cursor,lambda_f)
                        query_name=query.name
                        Query_Init(sql, schema, primary)  # to reset global variables
                        if query_name in new_queries:
                            if cost < new_queries[query_name]['cost']:
                                new_queries[query_name]['cost'] = cost
                                new_queries[query_name]['rel'] = subquery['rel'] + [relation]
                                new_queries[query_name]['obj'] = query
                        else:
                            new_queries[query_name] = {}
                            new_queries[query_name]['cost'] = cost
                            new_queries[query_name]['rel'] = subquery['rel'] + [relation]
                            new_queries[query_name]['obj'] = query
        queries = new_queries
    print("STEP: " + str(i + 1) + " | #candidates: " + str(len(queries)))
    for key, val in queries.items():
        display('dynamic prog. right deep', val['obj'].__str__(), num_of_checks, val['cost'])
        return [val['obj'].__str__(), num_of_checks, val['cost']]

def greedy_left_deep(sql, schema, primary):
    try:
        conn = psycopg2.connect(host="localhost", database="imdbload", user="postgres", password="admin")
    except:
        print("I am unable to connect to the database")
    # print(query)
    cursor = conn.cursor()

    query_env = Query_Init(sql, schema, primary)
    num_of_checks = 0

    actions = []
    for rel in query_env.actions:
        if type(rel) is Relation:
            actions.append(rel)
    query_env.actions = actions
    # Step 1
    # generate all possible 2-way joins
    # select cheapest
    queries = {}
    queries['cost'] = float('inf')
    for query_list in permutations(query_env.actions, 2):
        num_of_checks += 1
        query = Query(query_list[0], query_list[1])
        if query.join_condition is not []:  # no cross-join rule!
            cost = cm1(query, cursor)
            query_name = query.name
            Query_Init(sql, schema, primary)  # to reset global variables
            if cost < queries['cost']:
                queries['cost'] = cost
                queries['rel'] = [query_list[0], query_list[1]]
                queries['obj'] = query
    for i in range(1,len(query_env.actions)-1):
        new_queries={}
        new_queries['cost'] = float('inf')
        print("STEP: "+str(i)+" | #candidates: "+str(len(queries)))
        #Step 2-x
        # generate (3-x)-way joins
        # throw away equivalent queries which are more expensive
        for relation in query_env.actions:
            #print("rel: "+str(relation))
            #print("_______")
            subquery = queries
            #for x in subquery['rel']: print(x)
            #print("_______")
            if relation not in subquery['rel']:
                num_of_checks += 1
                query=Query(subquery['rel'][0],subquery['rel'][1])
                for j in range(2,len(subquery['rel'])):
                    query=Query(query,subquery['rel'][j])
                query = Query(query,relation)
                if query.join_condition is not []:  #True: ## no cross-join rule!
                    cost=cm1(query,cursor)
                    Query_Init(sql, schema, primary)  # to reset global variables
                    #print(query_name,cost)
                    if cost < new_queries['cost']:
                        new_queries['cost'] = cost
                        new_queries['rel'] = subquery['rel']+[relation]
                        new_queries['obj'] = query
        queries = new_queries
    cursor.close()
    print("STEP: " + str(i + 1) + " | #candidates: " + str(len(queries)))
    val = queries
    #for key, val in queries.items():
    display('greedy left deep', val['obj'].__str__(), num_of_checks, val['cost'])
    return [val['obj'].__str__(), num_of_checks, val['cost']]


def display(model,result,i,reward):
    print("\n****** Execution Plan ******")
    print("model: "+model)
    print("plan: "+result)
    print("steps/reward: "+str(i)+"/"+str(reward))
    print("*****************************\n")


def createAllTrees(numOfLeaves):
    result = []
    if numOfLeaves == 1:
        result.append("i")
        return result

    for lenLeftSubtree in range(1,numOfLeaves):
        possibleLeftSubtrees = createAllTrees(lenLeftSubtree)
        possibleRightSubtrees = createAllTrees(numOfLeaves-lenLeftSubtree)
        for lt in possibleLeftSubtrees:
            for rt in possibleRightSubtrees:
                result.append([lt,rt])
    return result

def createQuery(tree,list_iter):
    query=object
    if tree[0] is 'i' and tree[1] is 'i':
        query = Query(next(list_iter),next(list_iter))
    elif tree[0] is 'i':
        query=Query(next(list_iter),createQuery(tree[1],list_iter))
    elif tree[1] is 'i':
        query = Query(createQuery(tree[0],list_iter),next(list_iter))
    else:
        query = Query(createQuery(tree[0],list_iter),createQuery(tree[1],list_iter))
    return query


global cardinality
env = gym.make('CM1-postgres-card-job-one-v0')
schema = {"aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"], "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"], "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"], "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"], "comp_cast_type": ["id", "kind"], "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"], "company_type": ["id", "kind"], "complete_cast": ["id", "movie_id", "subject_id", "status_id"], "info_type": ["id", "info"], "keyword": ["id", "keyword", "phonetic_code"], "kind_type": ["id", "kind"], "link_type": ["id", "link"], "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"], "movie_info": ["id", "movie_id", "info_type_id", "info", "note"], "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"], "movie_keyword": ["id", "movie_id", "keyword_id"], "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"], "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"], "person_info": ["id", "person_id", "info_type_id", "info", "note"], "role_type": ["id", "role"], "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]}
primary = ['akaname.md5sum', 'akaname2.md5sum', 'akaname.name', 'akaname2.name', 'akaname.surname_pcode', 'akaname2.surname_pcode', 'akaname.name_pcode_cf', 'akaname2.name_pcode_cf', 'akaname.name_pcode_nf', 'akaname2.name_pcode_nf', 'akaname.person_id', 'akaname2.person_id', 'akaname.id', 'akaname2.id', 'akatitle.episode_of_id', 'akatitle2.episode_of_id', 'akatitle.kind_id', 'akatitle2.kind_id', 'akatitle.md5sum', 'akatitle2.md5sum', 'akatitle.movie_id', 'akatitle2.movie_id', 'akatitle.phonetic_code', 'akatitle2.phonetic_code', 'akatitle.title', 'akatitle2.title', 'akatitle.production_year', 'akatitle2.production_year', 'akatitle.id', 'akatitle2.id', 'castinfo.person_role_id', 'castinfo2.person_role_id', 'castinfo.movie_id', 'castinfo2.movie_id', 'castinfo.person_id', 'castinfo2.person_id', 'castinfo.role_id', 'castinfo2.role_id', 'castinfo.id', 'castinfo2.id', 'charname.imdb_id', 'charname2.imdb_id', 'charname.md5sum', 'charname2.md5sum', 'charname.name', 'charname2.name', 'charname.surname_pcode', 'charname2.surname_pcode', 'charname.name_pcode_nf', 'charname2.name_pcode_nf', 'charname.id', 'charname2.id', 'compcasttype.kind', 'compcasttype2.kind', 'compcasttype.id', 'compcasttype2.id', 'companyname.country_code', 'companyname2.country_code', 'companyname.imdb_id', 'companyname2.imdb_id', 'companyname.md5sum', 'companyname2.md5sum', 'companyname.name', 'companyname2.name', 'companyname.name_pcode_nf', 'companyname2.name_pcode_nf', 'companyname.name_pcode_sf', 'companyname2.name_pcode_sf', 'companyname.id', 'companyname2.id', 'companytype.kind', 'companytype2.kind', 'companytype.id', 'companytype2.id', 'completecast.movie_id', 'completecast2.movie_id', 'completecast.subject_id', 'completecast2.subject_id', 'completecast.id', 'completecast2.id', 'infotype.info', 'infotype2.info', 'infotype.id', 'infotype2.id', 'keyword.keyword', 'keyword2.keyword', 'keyword.phonetic_code', 'keyword2.phonetic_code', 'keyword.id', 'keyword2.id', 'kindtype.kind', 'kindtype2.kind', 'kindtype.id', 'kindtype2.id', 'linktype.link', 'linktype2.link', 'linktype.id', 'linktype2.id', 'moviecompanies.company_id', 'moviecompanies2.company_id', 'moviecompanies.company_type_id', 'moviecompanies2.company_type_id', 'moviecompanies.movie_id', 'moviecompanies2.movie_id', 'moviecompanies.id', 'moviecompanies2.id', 'movieinfo.info_type_id', 'movieinfo2.info_type_id', 'movieinfo.movie_id', 'movieinfo2.movie_id', 'movieinfo.id', 'movieinfo2.id', 'movieinfoidx.id', 'movieinfoidx2.id', 'moviekeyword.keyword_id', 'moviekeyword2.keyword_id', 'moviekeyword.movie_id', 'moviekeyword2.movie_id', 'moviekeyword.id', 'moviekeyword2.id', 'movielink.linked_movie_id', 'movielink2.linked_movie_id', 'movielink.link_type_id', 'movielink2.link_type_id', 'movielink.movie_id', 'movielink2.movie_id', 'movielink.id', 'movielink2.id', 'name.gender', 'name2.gender', 'name.imdb_id', 'name2.imdb_id', 'name.md5sum', 'name2.md5sum', 'name.name', 'name2.name', 'name.surname_pcode', 'name2.surname_pcode', 'name.name_pcode_cf', 'name2.name_pcode_cf', 'name.name_pcode_nf', 'name2.name_pcode_nf', 'name.id', 'name2.id', 'personinfo.info_type_id', 'personinfo2.info_type_id', 'personinfo.person_id', 'personinfo2.person_id', 'personinfo.id', 'personinfo2.id', 'roletype.id', 'roletype2.id', 'roletype.role', 'roletype2.role', 'title.episode_nr', 'title2.episode_nr', 'title.episode_of_id', 'title2.episode_of_id', 'title.imdb_id', 'title2.imdb_id', 'title.kind_id', 'title2.kind_id', 'title.md5sum', 'title2.md5sum', 'title.phonetic_code', 'title2.phonetic_code', 'title.season_nr', 'title2.season_nr', 'title.title', 'title2.title', 'title.production_year', 'title2.production_year', 'title.id', 'title2.id']

sql_query_9d = "SELECT * FROM cast_info, info_type, info_type AS info_type2, movie_info, movie_info_idx, name, title WHERE title.id = movie_info.movie_id AND title.id = movie_info_idx.movie_id AND title.id = cast_info.movie_id AND cast_info.movie_id = movie_info.movie_id AND cast_info.movie_id = movie_info_idx.movie_id AND movie_info.movie_id = movie_info_idx.movie_id AND name.id = cast_info.person_id AND info_type.id = movie_info.info_type_id AND info_type2.id = movie_info_idx.info_type_id"
sql_query_5b = "SELECT * FROM company_name, keyword, movie_companies, movie_keyword, title WHERE company_name.id = movie_companies.company_id AND movie_companies.movie_id = title.id AND title.id = movie_keyword.movie_id AND movie_keyword.keyword_id = keyword.id AND movie_companies.movie_id = movie_keyword.movie_id"
sql_query_24b = "SELECT * FROM keyword, movie_info, movie_keyword, title WHERE title.id = movie_info.movie_id AND title.id = movie_keyword.movie_id AND movie_keyword.movie_id = movie_info.movie_id AND keyword.id = movie_keyword.keyword_id"




start = time.time()
query_file = open('queries/job_queries_simple_label.txt','r')
res_file = open('queries/DP/res_left_deep.txt','w')
res_file.write("nr|label|query|resquery|steps|costs|planningtime\n")
i = 0
for query_l in query_file.readlines():
    x = query_l.split("|")
    query = x[1].replace(';','')
    label = x[0]
    qstart = time.time()
    res = dynamic_programming_left_deep(query, schema, primary,2)
    delta = time.time()-qstart
    res_file.write(str(i)+'|'+label+'|'+query.replace('\n','')+'|'+res[0]+'|'+str(res[1])+'|'+str(res[2])+'|'+str(delta)+"\n")
    i+=1

delta = time.time()-start
print(delta)
res_file.close()
query_file.close()
