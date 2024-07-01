# Google AI Python SDK for the Gemini API

[![PyPI version](https://badge.fury.io/py/google-generativeai.svg)](https://badge.fury.io/py/google-generativeai)
![Python support](https://img.shields.io/pypi/pyversions/google-generativeai)
![PyPI - Downloads](https://img.shields.io/pypi/dd/google-generativeai)

The Google AI Python SDK is the easiest way for Python developers to build with the Gemini API. The Gemini API gives you access to Gemini [models](https://ai.google.dev/models/gemini) created by [Google DeepMind](https://deepmind.google/technologies/gemini/#introduction). Gemini models are built from the ground up to be multimodal, so you can reason seamlessly across text, images, and code. 

## Get started with the Gemini API
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Login with your Google account.
3. [Create](https://aistudio.google.com/app/apikey) an API key. Note that in Europe the free tier is not available.
4. Try a Python SDK [quickstart](https://github.com/google-gemini/gemini-api-cookbook/blob/main/quickstarts/Prompting.ipynb) in the [Gemini API Cookbook](https://github.com/google-gemini/gemini-api-cookbook/).
5. For detailed instructions, try the 
[Python SDK tutorial](https://ai.google.dev/tutorials/python_quickstart) on [ai.google.dev](https://ai.google.dev).

## Usage example
See the [Gemini API Cookbook](https://github.com/google-gemini/gemini-api-cookbook/) or [ai.google.dev](https://ai.google.dev) for complete code.

1. Install from [PyPI](https://pypi.org/project/google-generativeai).

`pip install -U google-generativeai`

2. Import the SDK and configure your API key.

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

3. Create a model and run a prompt.

```python
model = genai.GenerativeModel('gemini-1.0-pro-latest')
response = model.generate_content("The opposite of hot is")
print(response.text)
```

## Documentation

See the [Gemini API Cookbook](https://github.com/google-gemini/gemini-api-cookbook/) or [ai.google.dev](https://ai.google.dev) for complete documentation.

## Contributing

See [Contributing](https://github.com/google/generative-ai-python/blob/main/CONTRIBUTING.md) for more information on contributing to the Google AI Python SDK.

## License

The contents of this repository are licensed under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
