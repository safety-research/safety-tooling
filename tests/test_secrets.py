from safetytooling.utils.utils import get_repo_root, load_secrets

TEST_SECRETS_CONTENT = """# This is a comment
OPENAI_API_KEY1=TEST_PLACEHOLDER_KEY_1
# Another comment

# Empty line above
ANTHROPIC_API_KEY=TEST_PLACEHOLDER_KEY_2
invalid line without equals
# Comment at the end"""


def test_secrets_with_comments():
    """Test that load_secrets properly handles comments and blank lines.

    Creates a temporary SECRETS.test file, runs the test, and cleans up afterwards.
    """
    test_file = get_repo_root() / "SECRETS.test"
    try:
        # Create test secrets file
        with open(test_file, "w") as f:
            f.write(TEST_SECRETS_CONTENT)

        # Load and verify the secrets
        secrets = load_secrets("SECRETS.test")

        # Verify only valid key-value pairs were loaded
        expected_keys = {"OPENAI_API_KEY1", "ANTHROPIC_API_KEY"}
        actual_keys = set(secrets.keys())

        # Check that we got exactly the expected keys
        if actual_keys != expected_keys:
            print(f"❌ Test failed: Expected keys {expected_keys}, got {actual_keys}")
            return False

        # Verify values were loaded correctly
        if secrets["OPENAI_API_KEY1"] != "TEST_PLACEHOLDER_KEY_1":
            print(f"❌ Test failed: Expected OPENAI_API_KEY1=TEST_PLACEHOLDER_KEY_1, got {secrets['OPENAI_API_KEY1']}")
            return False
        if secrets["ANTHROPIC_API_KEY"] != "TEST_PLACEHOLDER_KEY_2":
            print(
                f"❌ Test failed: Expected ANTHROPIC_API_KEY=TEST_PLACEHOLDER_KEY_2, got {secrets['ANTHROPIC_API_KEY']}"
            )
            return False

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_secrets_with_comments()
