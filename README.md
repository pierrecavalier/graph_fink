# Graph on fink
[Fink](https://fink-broker.org/) is a community driven project, open to anyone, that processes time-domains alert streams and connects them with follow-up facilities and science teams. Fink broker has been selected as a community broker to process the full stream of transient alerts from the Vera C. Rubin Observatory. Since 2020, we are processing the alert stream from the Zwicky Transient Facility (ZTF).

During a 4-month internship supervised by Julien Peloton and Julius Hrivnac, we tried to explore the analysis of graphs for the broker's alerts, mainly for anomaly detection and a new similarity measure.


## Usage 
Data can be found [here](https://fink-portal.org/) and once put in a folder data, one can create his own graph, there are severals Neural Network notebook where the only notable change is the size of the layers and the depth of the network.

The neural network will simply check for every pair of element in the dataset, a number between 0 and 1, that can be interpreted as a similarity measurement will be associated. A arbitrary threshold at 0.9 will help to select will elements will be connected in the graph.

The report details the code and how it is used (but in french). Presentation slides are in english however.

## Example of graphs

<p align="center">
<img src=https://github.com/pierrecavalier/graph_fink/blob/main/figures/graph.png width="500">
</p>


