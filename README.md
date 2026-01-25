<h1>GO-Guided Deep Learning Model for Biomarker Identification (GO-DMBI)</h1>
<p>
GO-DMBI is a graph-based deep learning framework for identifying disease biomarkers
from PPI network by integrating Gene Ontology (GO)–guided node embeddings with network topology.
</p>
<hr>
<h2>Overview</h2>
<p>
The GO-DMBI pipeline combines:
</p>
<ul>
  <li>Protein–Protein Interaction (PPI) network structure</li>
  <li>GO-term–based gene embeddings</li>
  <li>Deep learning models for supervised biomarker identification</li>
</ul>
<hr>
<h2>Training Data Requirements</h2>
<p>Training the model requires the following inputs:</p>
<ul>
  <li>
    <b>GEOKG Embeddings</b><br>
    <code>DATA/Data_process_output/disease_node_features_geokg.npy</code><br>
    Gene embeddings mapped using <code>data_process.py</code>.
  </li>
  <li>
    <b>GO Term Embeddings</b><br>
    <code>DATA/Data_process_output/disease_node_features_go_terms.npy</code><br>
    Generated using <code>go_terms_embedding.py</code> and mapped via
    <code>data_process.py</code>.
  </li>
  <li>
    <b>Node Metadata</b><br>
    <code>DATA/Data_process_output/disease_nodes_with_uniprot_and_embidx.csv</code><br>
    Curated disease-specific nodes with UniProt IDs and embedding indices.
  </li>
  <li>
    <b>PPI Network Edges</b><br>
    <code>DATA/Raw/Glioblastoma_microarray/glio_edges.csv</code><br>
    Edge list defining the disease-specific PPI network.
  </li>
  <li>
    <b>Output Directory</b><br>
    <code>update1_disease_outputs</code><br>
    Directory used to store model outputs and results.
  </li>
</ul>
<hr>
<h2>Labels for Supervised Training</h2>
<p>
Supervised training requires biomarker annotations:
</p>
<ul>
  <li>Binary labels indicating biomarker / non-biomarker status</li>
  <li>Labels must correspond to nodes listed in the node metadata file</li>
  <li>Label alignment is handled during model training</li>
</ul>
<hr>
<h2>Expected Project Structure</h2>
<pre>
GO-DMBI/
├── DATA/
│   ├── Raw/
│   └── Data_process_output/
├── src/
│   ├── data_process.py
│   ├── go_terms_embedding.py
│   └── model/
├── update1_disease_outputs/
├── README.md
└── requirements.txt
</pre>
<hr>

<h2>Notes</h2>
<ul>
  <li>Large datasets, embeddings, and trained models are not tracked in Git</li>
  <li>Only source code and documentation are version-controlled</li>
  <li>External storage should be used for data and model artifacts</li>
</ul>
<hr>
<h2>Citation</h2>
<p>
Citation details will be added upon publication.
</p>
