


f = open("~//PycharmProjects/mt-join-queryoptimization-with-drl/agents/queries/job_queries_label.txt", "r")
w = open("~//PycharmProjects/mt-join-queryoptimization-with-drl/agents/queries/job_queries_simple_labled.txt", "w")
qselect="SELECT * "
for x in f:
  y=x.split('WHERE')
  qwhere = "WHERE"
  e=0
  for i in y[1].split('AND'):
      if not(("'" in i) or ('%' in i) or ("LIKE" in i) or ("<" in i) or (">" in i) or ("BETWEEN" in i) or ("OR" in i) or ("=" not in i)):
          #print(i)
          if e is 0:
              qwhere=qwhere+i
              e = 1
          else:
              qwhere = qwhere +"AND"+ i

  z = y[0].split('FROM')
  alias = {}
  relations = []
  qfrom = 'FROM '
  e = 0
  for i in z[1].split(','):
      a = i.split(' AS ')
      rel = a[0].replace(' ', '')
      if rel in alias.values():
          alias[a[1].replace(' ', '')] = rel+"2"
          rel = rel+" AS "+rel+"2"
      else:
          alias[a[1].replace(' ', '')] = rel
      if e is 0:
          qfrom = qfrom + rel
          e = 1
      else:
          qfrom = qfrom + ", " + rel
  qfrom = qfrom
  for key, val in alias.items():
      qwhere = qwhere.replace(' '+key+'.', ' '+val+'.')

  #print(relations)
  #print(alias)
  print(qselect)
  print(qfrom)
  print(qwhere)
  w.write(x.split("|")[0]+"|"+qselect+qfrom+' '+qwhere)
  #print(j)
