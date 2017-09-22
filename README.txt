data_collection: from pcap to labelled jsons
1. parse_ctu_label: parse the groundtruth of CTU dataset
2. pcap2json: extract flow information from pcap and store them into jsons.
ML: the machine learning part
1. Learner: the interface for learning and testing
2. CtuCCAnalyzer: train and testing for CTU CC flows
3. CtuAdAnalyzer: train and testing for CTU Ad flows