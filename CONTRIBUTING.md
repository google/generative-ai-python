# How to become a contributor and submit your own code

**Table of contents**

* [Contributor License Agreements](#contributor-license-agreements)
* [Contributing a patch](#contributing-a-patch)
* [Running the tests](#running-the-tests)
* [Releasing the library](#releasing-the-library)

## Contributor License Agreements

We'd love to accept your sample apps and patches! Before we can take them, we
have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement
(CLA).

  * If you are an individual writing original source code and you're sure you
    own the intellectual property, then you'll need to sign an [individual CLA](https://developers.google.com/open-source/cla/individual).
  * If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a [corporate CLA](https://developers.google.com/open-source/cla/corporate).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your pull requests.

## Contributing a Patch

1.  Submit an issue describing your proposed change to the repo in question.
1.  The repo owner will respond to your issue promptly.
1.  If your proposed change is accepted, and you haven't already done so, sign a
    Contributor License Agreement (see details above).
1.  Fork the desired repo, develop and test your code changes.
1.  Ensure that your code adheres to the existing style in the code to which
    you are contributing.
1.  Ensure that your code has an appropriate set of tests which all pass.
1.  Title your pull request following [Conventional Commits](https://www.conventionalcommits.org/) styling.
1.  Submit a pull request.

### Before you begin

1.  [Select or create a Cloud Platform project][projects].
1.  [Enable billing for your project][billing]. 
1.  [Enable the Generative Language API][enable_api]. 
1.  [Set up authentication with a service account][auth] so you can access the
    API from your local workstation.

You can use an API-key, but remember never to save it in your source files.


## Development

### Local install

Install the source in "editable" mode, with testing requirements:

```
pip install -e .[dev]
```

This "editable" mode lets you edit the source without needing to reinstall the package.

### Testing

Use the builtin unittest package:

```
 python -m unittest
```

Or to debug, use:

```commandline
nose2 --debugger
```

### Type checking 

Use `pytype` (configured in `pyproject.toml`)

```
pip install pytype
touch google/__init__.py   #  https://github.com/google/pytype/issues/464
pytype
rm google/__init__.py
```

### Formatting:

Use black:

```
pip install black 
black .
```

### Generate api reference

```
python docs/build_docs.py
```


[setup]: https://cloud.google.com/nodejs/docs/setup
[projects]: https://console.cloud.google.com/project
[billing]: https://support.google.com/cloud/answer/6293499#enable-billing
[enable_api]: https://console.cloud.google.com/flows/enableapi?apiid=generativelanguage.googleapis.com
[auth]: https://cloud.google.com/docs/authentication/getting-started
