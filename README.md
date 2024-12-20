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

1. [modern-to-shakesperean-translation](https://huggingface.co/datasets/harpreetsahota/modern-to-shakesperean-translation) - Trained
```
ds = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
```
2. [English-to-shakespeare-parallel](https://huggingface.co/datasets/ayaan04/english-to-shakespeare)
```
ds = load_dataset("ayaan04/english-to-shakespeare")
```
3. [book-text-style-transfer](https://huggingface.co/datasets/jdpressman/retro-text-style-transfer-v0.1)
```
ds = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
```
4. [formal-casual](https://huggingface.co/datasets/Mehaki/formal_casual) - Trained
```
ds = load_dataset("Mehaki/formal_casual")
```
5. [Poetry-Modern_Resaissance](https://huggingface.co/datasets/merve/poetry) - Trained
```
ds = load_dataset("merve/poetry")
```
