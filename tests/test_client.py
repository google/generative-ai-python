import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from google.api_core import client_options
import google.ai.generativelanguage as glm
from google.generativeai import client


class ClientTests(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        client._client_manager = client._ClientManager()

    def test_api_key_passed_directly(self):
        client.configure(api_key="AIzA_direct")

        client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_direct")

    def test_api_key_passed_via_client_options(self):
        client_opts = client_options.ClientOptions(api_key="AIzA_client_opts")
        client.configure(client_options=client_opts)

        client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_client_opts")

    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzA_env"})
    def test_api_key_from_environment(self):
        # Default to API key loaded from environment.
        client.configure()
        client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_env")

        # But not when a key is provided explicitly.
        client.configure(api_key="AIzA_client")
        client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_client")

    def test_api_key_cannot_be_empty_passed_directly(self):
        with self.assertRaisesRegex(ValueError, "api key can't be empty."):
            client.configure(api_key="")

    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": ""})
    def test_api_key_cannot_be_empty_from_environment(self):
        with self.assertRaisesRegex(ValueError, "api key can't be empty."):
            client.configure()

    @parameterized.parameters(
        client.get_default_discuss_client,
        client.get_default_text_client,
        client.get_default_discuss_async_client,
        client.get_default_model_client,
        client.get_default_operations_client,
    )
    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": ""})
    def test_configureless_client_with_empty_api_key(self, factory_fn):
        with self.assertRaisesRegex(ValueError, "api key can't be empty."):
            _ = factory_fn()

    def test_empty_api_key_cannot_be_set_twice(self):
        client_opts = client_options.ClientOptions(api_key="")

        with self.assertRaisesRegex(ValueError, "api key can't be empty."):
            client.configure(api_key="", client_options=client_opts)

    def test_api_key_cannot_be_set_twice(self):
        client_opts = client_options.ClientOptions(api_key="AIzA_client_opts")

        with self.assertRaisesRegex(ValueError, "You can't set both"):
            client.configure(api_key="AIzA_client", client_options=client_opts)

    def test_api_key_and_client_options(self):
        # Client options should merge with an API key, as long as they are both
        # do not have the key set.
        client_opts = client_options.ClientOptions(api_endpoint="web.site")
        client.configure(api_key="AIzA_client", client_options=client_opts)

        actual_client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(actual_client_opts.api_key, "AIzA_client")
        self.assertEqual(actual_client_opts.api_endpoint, "web.site")

    @parameterized.parameters(
        client.get_default_discuss_client,
        client.get_default_text_client,
        client.get_default_discuss_async_client,
        client.get_default_model_client,
        client.get_default_operations_client,
    )
    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzA_env"})
    def test_configureless_client_with_key(self, factory_fn):
        _ = factory_fn()

        # And ensure that it has set the default options.
        actual_client_opts = client._client_manager.client_config["client_options"]
        self.assertEqual(actual_client_opts.api_key, "AIzA_env")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, metadata=None):
            self.metadata = metadata

        not_a_function = 7

        def _hidden(self):
            self.called_hidden = True

        @staticmethod
        def static():
            pass

        @classmethod
        def classm(cls):
            cls.called_classm = True

    @mock.patch.object(glm, "TextServiceClient", DummyClient)
    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzA_env"})
    def test_default_metadata(self):
        # The metadata wrapper injects this argument.
        metadata = [("hello", "world")]
        client.configure(default_metadata=metadata)

        text_client = client.get_default_text_client()
        text_client.generate_text()

        self.assertEqual(metadata, text_client.metadata)

        self.assertEqual(text_client.not_a_function, ClientTests.DummyClient.not_a_function)

        # Since these don't have a metadata arg, they'll fail if the wrapper is applied.
        text_client._hidden()
        self.assertTrue(text_client.called_hidden)

        text_client.static()

        text_client.classm()
        self.assertTrue(ClientTests.DummyClient.called_classm)

    def test_same_config(self):
        cm1 = client._ClientManager()
        cm1.configure(api_key="abc")

        cm2 = client._ClientManager()
        cm2.configure(client_options=dict(api_key="abc"))

        self.assertEqual(
            cm1.client_config["client_info"].__dict__, cm2.client_config["client_info"].__dict__
        )
        self.assertEqual(
            cm1.client_config["client_options"].__dict__,
            cm2.client_config["client_options"].__dict__,
        )
        self.assertEqual(cm1.default_metadata, cm2.default_metadata)


if __name__ == "__main__":
    absltest.main()
