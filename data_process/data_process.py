import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np

f_microbe=open('selected_microbe.txt')
f_drug=open('All_CIDs_matched_SMILES.smi')
f_disease=open('selected_disease.txt')
f_lineage=open('lineage.txt')
list_microbe=[i.strip('\n') for i in f_microbe]
list_drug=[i.strip('\n').split('\t')[1] for i in f_drug]
list_disease=[i.strip('\n') for i in f_disease]

NPs = []
SMIs = []
for lines in f_drug:
    lines = lines.strip().split('\t')
    smi = lines[0]
    np_id = lines[1]
    NPs.append(np_id)
    SMIs.append(smi)
S = np.zeros((len(NPs),len(NPs)))
fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(i)) for i in SMIs]
for i in range(len(fps)):
    for j in range(len(fps)):
        S[i][j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
np.savetxt('drug_similarity.txt',S)

microbe_dag = nx.DiGraph()
for i in f_lineage:
    i=i.strip('\n').split('\t')
    tax_list=i[2].split(';')
    for n in range(len(tax_list)-1):
        microbe_dag.add_edge(tax_list[n], tax_list[n+1])
        microbe_dag.add_edge(tax_list[n+1], tax_list[n])
microbe_list=[]
for i in f_microbe:
    micro=i.strip('\n')
    if micro in microbe_dag.nodes():
        microbe_list.append(micro)
microbe_distance=np.zeros((len(microbe_list),len(microbe_list)))
for i in range(len(microbe_list)):
    for j in range(len(microbe_list)):
        micro1=microbe_list[i]
        micro2=microbe_list[j]
        distance=nx.shortest_path_length(microbe_dag, source=micro1, target=micro2)
        microbe_distance[i][j]=distance
microbe_similarity=1-microbe_distance/np.max(microbe_distance)
np.savetxt('microbe_similarity.txt',microbe_similarity)

def parse_mesh_data(mesh_file_path):
    tree = ET.parse(mesh_file_path)
    root = tree.getroot()

    mesh_terms = {}

    for descriptor_record in root.findall('.//DescriptorRecord'):
        descriptor_ui = descriptor_record.find('./DescriptorUI').text
        descriptor_name = descriptor_record.find('./DescriptorName/String').text

        mesh_terms[descriptor_ui] = {
            'name': descriptor_name,
            'children': []
        }

        for child in descriptor_record.findall('.//TreeNumberList/TreeNumber'):
            child_ui = child.text.split('/')[0]
            mesh_terms[descriptor_ui]['children'].append(child_ui)

    return mesh_terms

def build_dag(mesh_terms):
    dag = nx.DiGraph()

    for term, data in mesh_terms.items():

        for child in data['children']:
            child_list=child.split('.')
            father_nodes=[]
            for i in range(len(child_list)):
                father_node='.'.join(child_list[:i+1])
                father_nodes.append(father_node)
            for n in range(len(father_nodes)-1):
                dag.add_edge(father_nodes[n], father_nodes[n+1])
                dag.add_edge(father_nodes[n+1], father_nodes[n])
            dag.add_edge(father_nodes[-1], term)
            dag.add_edge(term, father_nodes[-1])

    return dag

mesh_file_path = './desc2024.xml'
mesh_terms = parse_mesh_data(mesh_file_path)
diseae_dag = build_dag(mesh_terms)
dis_distance=np.zeros((len(list_disease),len(list_disease)))
for i in range(len(list_disease)):
    for j in range(len(list_disease)):
        dis1=list_disease[i]
        dis2=list_disease[j]
        distance=nx.shortest_path_length(diseae_dag, source=dis1, target=dis2)
        dis_distance[i][j]=distance
dis_similarity=1-dis_distance/np.max(dis_distance)
np.savetxt('dis_similarity.txt',dis_similarity)
