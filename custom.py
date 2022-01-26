import dill
with open('trained_data.pkl', 'rb') as f:
       data= dill.load(f)
for cluster in data.clusters :
    print (cluster.points[0].text)



