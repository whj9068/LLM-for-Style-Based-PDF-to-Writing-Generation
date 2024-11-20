### Utilizing LLM for Style-Based PDF to Writing Generation

1. Download Ollama from https://ollama.com/download

2.

```
sudo ollama run llama3.2
```

3. 
```
ollama pull llama3.2
```
4. 
```
pip install -r requirements.txt
```
4. 
```
python context.py
```

### Opensource Datasets for style transfer training

1. [Modern-to-shakesperean-translation](https://huggingface.co/datasets/harpreetsahota/modern-to-shakesperean-translation)
```
ds = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
```

3. 
