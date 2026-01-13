import pandas as pd

hop2_source = "./2hop.csv"
direct_source = "./direct.csv"
destination = "./raw_relations.csv"

hop2 = pd.read_csv(hop2_source)
direct = pd.read_csv(direct_source)

rows = []

for i, row in hop2.iterrows():
    c1_qid = row['company1'].split('/')[-1]
    c2_qid = row['company2'].split('/')[-1]
    relation = row['p1'].split('/')[-1] + '-' + row['p2'].split('/')[-1]
    rows.append((c1_qid, c2_qid, relation))

for i, row in direct.iterrows():
    c1_qid = row['company1'].split('/')[-1]
    c2_qid = row['company2'].split('/')[-1]
    relation = row['relation'].split('/')[-1]
    rows.append((c1_qid, c2_qid, relation))

relations = pd.DataFrame(rows, columns=['Source', 'Target', 'Relation_Type'])
print(f"Got {len(relations['Relation_Type'].unique())} unique relation types...")
print(f"Totally got {len(relations)} relations between companies...")
print(f"Raw relations saved to {destination}")
relations.to_csv(destination, index=False)