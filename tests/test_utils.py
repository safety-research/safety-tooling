import os
from unittest.mock import patch

from safetytooling.utils import utils


def test_setup_environment_treats_blank_api_keys_as_unset():
    with patch("safetytooling.utils.utils.dotenv.load_dotenv", return_value=True):
        os.environ["TOGETHER_API_KEY"] = ""
        try:
            utils.setup_environment()
            # Should be removed so downstream code gets None, not ""
            assert "TOGETHER_API_KEY" not in os.environ
        finally:
            os.environ.pop("TOGETHER_API_KEY", None)


def test_setup_environment_treats_whitespace_only_api_keys_as_unset():
    with patch("safetytooling.utils.utils.dotenv.load_dotenv", return_value=True):
        os.environ["TOGETHER_API_KEY"] = "   \t  "
        try:
            utils.setup_environment()
            assert "TOGETHER_API_KEY" not in os.environ
        finally:
            os.environ.pop("TOGETHER_API_KEY", None)


def test_setup_environment_normalizes_any_var_ending_in_api_key():
    with patch("safetytooling.utils.utils.dotenv.load_dotenv", return_value=True):
        os.environ["SOME_FUTURE_API_KEY"] = ""
        try:
            utils.setup_environment()
            assert "SOME_FUTURE_API_KEY" not in os.environ
        finally:
            os.environ.pop("SOME_FUTURE_API_KEY", None)
