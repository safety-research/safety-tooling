from safetytooling.utils.utils import load_secrets


def test_secrets_with_comments():
    try:
        # Load the test secrets file
        secrets = load_secrets("SECRETS.test")

        # Verify only valid key-value pairs were loaded
        expected_keys = {"OPENAI_API_KEY1", "ANTHROPIC_API_KEY"}
        actual_keys = set(secrets.keys())

        # Check that we got exactly the expected keys
        if actual_keys != expected_keys:
            print(f"❌ Test failed: Expected keys {expected_keys}, got {actual_keys}")
            return False

        # Verify values were loaded correctly
        if secrets["OPENAI_API_KEY1"] != "test-key-1":
            print(f"❌ Test failed: Expected OPENAI_API_KEY1=test-key-1, got {secrets['OPENAI_API_KEY1']}")
            return False
        if secrets["ANTHROPIC_API_KEY"] != "test-key-2":
            print(f"❌ Test failed: Expected ANTHROPIC_API_KEY=test-key-2, got {secrets['ANTHROPIC_API_KEY']}")
            return False

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    test_secrets_with_comments()
