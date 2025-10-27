# llama-index-rag
RAG implementation using Llama Index and Qdrant doc storage.

## **Credit**
This software is distributed with GPLv3 license. Largely based on software reported in this GitHub repository:    

- https://github.com/Otman404/local-rag-llamaindex

## **Installation**    

### Install llama-index & other dependencies
* Create and activate a virtual environment, e.g. ```llama-index-rag```, under a desired path ```VENV_DIR```     
  ```
  $ python3 -m venv $VENV_DIR/llama-index-rag
  $ source $VENV_DIR/llama-index-rag/bin/activate
  ```   
* Install dependencies inside venv:   
  ```
  (llama-index-rag)$ pip install -r $SRC_DIR/requirements.txt
  ```

### Install & run Qdrant service
* Download the latest Qdrant image from Dockerhub:   
```
docker pull qdrant/qdrant
```
* Set up a storage directory, e.g. `/scratch/qdrant/storage`, and run the service adding the storage directory as a volume:
```
docker run -p 6333:6333 -p 6334:6334 \
    -v "/scratch/qdrant/storage:/qdrant/storage:z" \
    qdrant/qdrant
```
* Qdrant is now accessible at these endpoints:   
    - REST API: ```localhost:6333```
    - Web UI: ```localhost:6333/dashboard```
    - GRPC API: ```localhost:6334```

### Install & run Qdrant service


## **Usage**  
WRITE ME   
