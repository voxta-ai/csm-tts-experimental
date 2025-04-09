# Sesame CSM TTS Engine

This repository shares experimental performance gains and streaming for [Sesame Conversational Speech Model](https://github.com/SesameAILabs/csm)

Runs on CUDA 12.4 and Python 3.12.9.

You also need to download:

- [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [CSM-1b](https://huggingface.co/sesame/csm-1b)

```bash
# Recommended: Create a venv and activate it
python -m venv env
venv\Scripts\activate 
# Install torch 2.6 if you don't already have it
pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install the requirements
pip3 install -r requirements.txt
# Run the test
python test.py --model-path path/to/model.gguf --snac-path path/to/snac-folder --output-path path/to-outputs
```

For arguments, see [test.py](./test.py)

## Credits

Credits go to CSM creators: [Sesame Conversational Speech Model](https://github.com/SesameAILabs/csm)
Also thanks to @davidbrowne17 for sharing their streaming implementation: [davidbrowne17/csm-streaming](https://github.com/davidbrowne17/csm-streaming)

## License

Inherited from Sesame CSM.

[Apache-2.0 License](https://github.com/SesameAILabs/csm?tab=Apache-2.0-1-ov-file)