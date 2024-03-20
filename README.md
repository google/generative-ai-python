# Google Python SDK for the Gemini API

[![PyPI version](https://badge.fury.io/py/google-generativeai.svg)](https://badge.fury.io/py/google-generativeai)
![Python support](https://img.shields.io/pypi/pyversions/google-generativeai)
![PyPI - Downloads](https://img.shields.io/pypi/dd/google-generativeai)

The Google AI Python SDK enables developers to use Google's state-of-the-art generative AI
models (like Gemini and PaLM) to build AI-powered features and applications. This SDK
supports use cases like:

- Generate text from text-only input
- Generate text from text-and-images input (multimodal) (for Gemini only)
- Build multi-turn conversations (chat)
- Embedding

For example, with just a few lines of code, you can access Gemini's multimodal
capabilities to generate text from text-and-image input:

```python
model = genai.GenerativeModel('gemini-pro-vision')

cookie_picture = {
    'mime_type': 'image/png',
    'data': Path('cookie.png').read_bytes()
}
prompt = "Give me a recipe for this:"

response = model.generate_content(
    contents=[prompt, cookie_picture]
)
print(response.text)
```


## Try out the API

Install from PyPI.

`pip install google-generativeai`

[Obtain an API key from AI Studio](https://makersuite.google.com/app/apikey),
then configure it here.

Import the SDK and load a model.

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel('gemini-pro')
```

Use `GenerativeModel.generate_content` to have the model complete some initial text.

```python
response = model.generate_content("The opposite of hot is")
print(response.text)  # cold.
```

Use `GenerativeModel.start_chat` to have a discussion with a model.

```python
chat = model.start_chat()
response = chat.send_message('Hello, what should I have for dinner?')
print(response.text) #  'Here are some suggestions...'
response = chat.send_message("How do I cook the first one?")
```



## Installation and usage

Run [`pip install google-generativeai`](https://pypi.org/project/google-generativeai).

For detailed instructions, you can find a
[quickstart](https://ai.google.dev/tutorials/python_quickstart) for the Google AI
Python SDK in the Google documentation.

This quickstart describes how to add your API key and install the SDK in your app,
initialize the model, and then call the API to access the model. It also describes some
additional use cases and features, like streaming, embedding, counting tokens, and
controlling responses.


## Documentation

Find complete documentation for the Google AI SDKs and the Gemini model in the Google
documentation: https://ai.google.dev/docs


## Contributing

See [Contributing](https://github.com/google/generative-ai-python/blob/main/CONTRIBUTING.md) for more information on contributing to the Google AI Python SDK.

## Developers who use the PaLM API

### Migrate to use the Gemini API

Check our [migration guide](https://ai.google.dev/docs/migration_guide) in the Google
documentation.

### Installation and usage for the PaLM API

Install from PyPI.

`pip install google-generativeai`

[Obtain an API key from AI Studio](https://makersuite.google.com/app/apikey), then
configure it here.

```python
import google.generativeai as palm

palm.configure(api_key=os.environ["PALM_API_KEY"])
```

Use `palm.generate_text` to have the model complete some initial text.

```python
response = palm.generate_text(prompt="The opposite of hot is")
print(response.result)  # cold.
```

Use `palm.chat` to have a discussion with a model.

```python
response = palm.chat(messages=["Hello."])
print(response.last) #  'Hello! What can I help you with?'
response.reply("Can you tell me a joke?")
```

### Documentation for the PaLM API

- [General PaLM documentation](https://ai.google.dev/docs/palm_api_overview)

- [Text quickstart](https://github.com/google/generative-ai-docs/blob/main/site/en/palm_docs/text_quickstart.ipynb)

- [Chat quickstart](https://github.com/google/generative-ai-docs/blob/main/site/en/palm_docs/chat_quickstart.ipynb)

- [Tuning quickstart](https://github.com/google/generative-ai-docs/blob/main/site/en/palm_docs/tuning_quickstart_python.ipynb)

### Colab magics

```
%pip install -q google-generativeai
%load_ext google.generativeai.notebook
```

Once installed, use the Python client via the `%%llm` Colab magic. Read the full guide [here](https://developers.generativeai.google/tools/notebook_magic).

```python
%%llm
The best thing since sliced bread is
```

## License

The contents of this repository are licensed under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
