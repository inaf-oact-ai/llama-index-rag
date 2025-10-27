# llama-index-rag
RAG implementation using Llama Index and Qdrant doc storage.

## **Credit**
This software is distributed with GPLv3 license. Largely based on software reported in this GitHub repository:    

- https://github.com/Otman404/local-rag-llamaindex

## **Installation**    

### Install llama-index & other dependencies
* Create and activate a virtual environment, e.g. ```llama-index-rag```, under a desired path ```VENV_DIR```     
  ```
  python3 -m venv $VENV_DIR/llama-index-rag
  source $VENV_DIR/llama-index-rag/bin/activate
  ```   
* Install dependencies inside venv:   
  ```
  pip install -r $SRC_DIR/requirements.txt
  ```

### Install & run Qdrant service
* Install Docker:    
```
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo systemctl status docker
```
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

### Install & run Ollama service
* To install Ollama, run the following command as a root user:
```
curl -fsSL https://ollama.com/install.sh | sh
```
* Create a directory for ollama models, e.g. `/scratch/ollama/models`, and edit ollama service configuration `/etc/systemd/system/ollama.service` as follows:
```
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=$USER
Group=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/scratch/ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"

[Install]
WantedBy=default.target
```

* Run the ollama service:
```
sudo systemctl start ollama.service
```

* Pull some LLM models, e.g. ```llama3.1```:
```
ollama pull llama3.1:8b
ollama list
```

### Download HF models    
* Download embedding models from HuggingFace. Below, we list possible options, some currently highly ranked in the leaderboard (https://huggingface.co/spaces/mteb/leaderboard):   
```
export HF_HOME="/scratch/huggingface"
hf download sentence-transformers/all-mpnet-base-v2
hf download mixedbread-ai/mxbai-embed-large-v1
hf download Qwen/Qwen3-Embedding-8B
hf download nvidia/llama-embed-nemotron-8b
```

## **Usage**  

### **Upload docs to Qdrant storage**
Use the provided script `ingest.doc.py` to upload docs or scientific papers in the Qdrant storage. Below, we show a bash script example:    

```
#!/bin/bash

############################
##   ENV
############################
# - Source env
source /home/riggi/software/venvs/llama-index-rag/bin/activate

############################
##  OPTIONS
############################
DATA_PATH="/home/riggi/Documents/papers"
COLLECTION="papers"
CHUNK_SIZE=1024
MODEL="Qwen/Qwen3-Embedding-0.6B"
QDRANT_URL="http://localhost:6333"

############################
##   RUN
############################
echo "INFO: Start doc upload ..."
date

python scripts/ingest_doc.py \
  --data_path=$DATA_PATH \
  --collection_name=$COLLECTION \
  --chunk_size=$CHUNK_SIZE \
  --file_exts=".pdf" --recursive \
  --embedding_model=$MODEL \
  --qdrant_url=$QDRANT_URL

date
echo "INFO: End doc upload"
```
