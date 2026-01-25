# GO guided Deep Learning Model For biomarker Identification (GO - DMBI)
GO-DMBI is a deep learning model for identifying biomarkers from a PPI Network.

# Training model
 Training the model requires the following data
    1. GEOKG_PATH = "DATA/Data_process_output/disease_node_features_geokg.npy" # mapped using data_process.py
    2. GO_TERMS_PATH = "DATA/Data_process_output/disease_node_features_go_terms.npy" # embedding generated go_terms_embedding.py and mapping and numpy conversion done            sing data_process.py
    3. NODES_PATH = "DATA/Data_process_output/disease_nodes_with_uniprot_and_embidx.csv" # Nodes curated from data_process.py i.e. nodes that has GO terms and GEOKG             mapped
    4. EDGES_PATH = "DATA/Raw/Glioblastoma_microarray/glio_edges.csv" # edges form the PPI network
    5. OUTPUT_DIR = "update1_disease_outputs" # Output directory specified to save outputs
    Training the data on requires labels i.e, biomarkers 
    

