# research-tooling
Contains code useful for AI safety research.

# Repository setup

### Environment

To set up the development environment for this project,
follow the steps below:

1. First install `micromamba` if you haven't already.
You can follow the installation instructions here:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html. The Homebrew / automatic installation method is recommended.
Or just run:
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
2. Next, you need to run the following from the root of the repository:
    ```bash
    micromamba env create -n safetytooling python=3.11.7
    micromamba activate safetytooling
    pip install -r requirements.txt
    ```

### Secrets

You should create a file called `SECRETS` at the root of the repository
with the following contents:
```
# Required
OPENAI_API_KEY1=<your-key>
OPENAI_API_KEY2=<your-key>
ANTHROPIC_API_KEY=<your-key>
RUNPOD_API_KEY=<your-key>
HF_API_KEY=<your-key>
GOOGLE_API_KEY=<your-key>
GOOGLE_PROJECT_ID=<GCP-project-name>
GOOGLE_PROJECT_REGION=<GCP-project-region>
ELEVENLABS_API_KEY=<your-key>

# Optional: Where to log prompt history by default.
# If not set or empty, prompt history will not be logged.
# Path should be relative to the root of the repository.
PROMPT_HISTORY_DIR=<path-to-directory>
```

### Linting and formatting
We lint our code with [Ruff](https://docs.astral.sh/ruff/) and format with black.
This tool is automatically installed when you run set up the development environment.

If you use vscode, it's recommended to install the official Ruff extension.

In addition, there is a pre-commit hook that runs the linter and formatter before you commit.
To enable it, run `make hooks` but this is optional.

### Testing
Run `pytest -n auto` to run tests in parallel.
You can run all tests (including some slow flaky ones) with
`RUN_SLOW_TESTS=1 pytest -n auto`.

### Updating dependencies
We only pin top-level dependencies only to make cross-platform development easier.

- To add a new python dependency, add a line to `requirements.txt`.
- To upgrade a dependency, bump the version number in `requirements.txt`.

Run `pip install -r requirements.txt` after edits have been made to the requirements file.

To check for outdated dependencies, run `pip list --outdated`.

# Usage

### Running Finetuning

To launch a finetuning job, run the following command:
```bash
python -m safetytooling.apis.finetuning.run --model 'gpt-3.5-turbo-1106' --train_file <path-to-train-file> --n_epochs 1
```
This should automatically create a new job on the OpenAI API, and also sync that run to wandb. You will have to keep the program running until the OpenAI job is complete.

You can include the `--dry_run` flag if you just want to validate the train/val files and estimate the training cost without actually launching a job.

### API Usage
To get OpenAI usage stats, run:
```bash
python -m safetytooling.apis.inference.usage.usage_openai
```
You can pass a list of models to get usage stats for specific models. For example:
```bash
python -m safetytooling.apis.inference.usage.usage_openai --models 'model-id1' 'model-id2' --openai_tags 'OPENAI_API_KEY1' 'OPENAI_API_KEY2'
```
And for Anthropic, to fine out the numbder of threads being used run:
```bash
python -m safetytooling.apis.inference.usage.usage_anthropic
```
