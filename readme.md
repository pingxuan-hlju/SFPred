### SFPred

***

**Subgraph topology and dynamic graph topology
enhanced graph learning and pairwise feature
context relationship integration for predicting
disease-related miRNAs**

### Operating environment

***

- python 3.8.0

- PyTorch 2.0.0

- numpy 1.23.5

- scipy 1.10.1 

- scikit-learn  1.2.0      
                  




## Dataset

| File_name        | Data_type       | Description                                                                                                                                                                                                                                        | Source                                                                                               |
|------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| mi_name.txt      | miRNA           | The file contains names of 793 miRNAs.                                                                                                                                                                                                             | [HMDD](https://www.cuilab.cn/hmdd)                                                                   |
| dis_name.txt     | disease         | The file contains names of 341 diseases.                                                                                                                                                                                                           | [HMDD](https://www.cuilab.cn/hmdd)                                                                   |
| mi_sim.txt       | miRNA-miRNA     | The file contains the functional similarity among 793 miRNAs. The value in _i_-th row and _j_-th column is the similarity between the _i_-th miRNA _m<sub>i</sub>_ and the _j_-th miRNA _m<sub>j</sub>_.                                           | [Wang *et al.*](https://academic.oup.com/bioinformatics/article/26/13/1644/200577?login=false)$^{1}$        |
| dis_sim.txt      | disease-disease | The file contains the semantic similarity among 341 diseases. The value in _i_-th row and _j_-th column is the similarity between the _i_-th disease _d<sub>i</sub>_ and the _j_-th disease _d<sub>j</sub>_.                                       | [Wang *et al.*](https://academic.oup.com/bioinformatics/article/26/13/1644/200577?login=false)$^{1}$                                                             |
| mi_dis.txt       | miRNA-disease   | The file includes the known 7,908 associations between 793 miRNAs and 341 diseases. The value in _i_-th row and _j_-th column is 1 when the _i_-th miRNA _m<sub>i</sub>_ is associated with the _j_-th disease _d<sub>j</sub>_, otherwise it is 0. | [HMDD](https://www.cuilab.cn/hmdd)   |
(1) Wang, D.; Wang, J.; Lu, M.; Song, F.; Cui, Q. Inferring the human microRNA functional similarity and functional network based on microRNA-associated diseases. Bioinformatics 2010, 26, 1644–1650.


### How to run the code
1.Environment Preparing:
        create the environment according to the above requirements and download the required dependencies.


2.Train the model:
        run main.py.
***



