The doc_enc library is devoted to the computation of cross-lingual vector representations of long texts' embeddings applicable in information retrieval and classification tasks.

Full documentation in [Russian](https://semvectors-doc-enc.readthedocs.io/ru/latest/index.html)

### Quick start:
First of all, you should download a small dataset of documents and a pre-trained model:

``` sh
curl -O dn11.isa.ru:8080/doc-enc-data/datasets.docs.mini.v1.tar.gz
tar xf datasets.docs.mini.v1.tar.gz
find docs-mini/texts/ -name "*.txt" > files.txt
curl -O http://dn11.isa.ru:8080/doc-enc-data/models.def.pt
```

The data is available for downloading by the [link](https://mega.nz/folder/vkghwIDY#3WIHndJIiti5HHrncR8IVg), in case the above links are not working. 
To start the conversion process, it is convenient to use a pre-built Docker image. 
Before doing so, ensure that you have installed the NVIDIA Container Toolkit by following the instructions provided in the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).

``` sh
docker run  --gpus=1  --rm  -v $(pwd):/temp/ -w /temp  \
  semvectors/doc_enc:0.1.2 \
  docenccli docs -i /temp/files.txt -o /temp/vecs -m /temp/models.def.pt
```

Vectors will be stored in the `vecs` directory alongside their corresponding file names using the `numpy.savez` function.
Below is an  example of how to load the vectors from these files:

``` python
import numpy as np

obj = np.load('vecs/0000.npz')
print(obj['ids'][:2])
print(obj['embs'][:2])
```

