<h1>CABIgo: A gated fusion framework of graph convolution networks and Multi-Layer Perceptron for Context-Aware Biomarker Identification Using Gene Ontology Embeddings</h1>
<p> CABIgo is a graph-based deep learning framework for identifying disease biomarkers
from PPI network by integrating Gene Ontology (GO)–guided node embeddings with network topology.
</p>
<h2>Overview</h2>
<p>
The CABIgo pipeline combines:
</p>
  <li>Protein–Protein Interaction (PPI) network structure</li>
  <li>GO-term–based protein embeddings</li>
  <li>Deep learning models for biomarker identification</li>
<h2>Training Data Requirements</h2>
<p>Training the model requires the following inputs:</p>
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
    <b>Node data</b><br>
    <code>DATA/Data_process_output/disease_nodes_with_uniprot_and_embidx.csv</code><br>
    Curated disease-specific nodes with UniProt IDs and embedding indices.
  </li>
  <li>
    <b>PPI Network Edges</b><br>
    <code>disease_edges.csv</code><br>
    Edge list defining the disease-specific PPI network.
  </li>
  <li>
    <b>Output Directory</b><br>
    <code>disease_outputs</code><br>
    Directory used to store model outputs and results.
  </li>
<h2>Labels for Supervised Training</h2>
  <li>Binary labels indicating biomarker </li>
<h2>Notes</h2>
<ul>
  <li>Some Large datasets, embeddings, and trained models are not tracked in Git</li>
  <li>The fine tuned BioBert model weight is aviliable at <href> https://doi.org/10.5281/zenodo.19972967 </href> </li
  <li> The trained model weights for each disease are at the respective directories model-outputs_disease/disease</li>
</ul>
<h2>Citation</h2>
<p>
Citation details will be added upon publication.
</p>
