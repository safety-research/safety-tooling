# safety-tooling

- This is a repo designed to be shared across many AI Safety projects. It has built up since Summer 2023 during projects with Ethan Perez. The code primarily provides an LLM API with a common interface for OpenAI, Anthropic and Google models.
- The aim is that this repo continues to grow and evolve as more collaborators start to use it, ultimately speeding up new AI Safety researchers that join the cohort in the future. Furthermore, it provides an opportunity for those who have less of a software engineering background to upskill in contributing to a larger codebase that has many users.
- This repo is designed to be a git [submodule](https://git-scm.com/docs/git-submodule). This has the benefit of being able to use the code directly (without needing to install it) and having the ease of being able to commit changes that everyone can benefit from. When you `cd` into the submodule directory, you can `git pull` and `git push` to that repository.
    - An example repo that uses `safety-tooling` as a submodule is provided and explained in the second section.

## Repository setup

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
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate safetytooling
    pip install -r requirements.txt
    ```

### Redis Configuration (Optional)

To use Redis for caching instead of writing everything to disk, *[install Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)*, and make sure it's running by doing `redis-cli ping`.

```bash
export REDIS_CACHE=True  # Enable Redis caching (defaults to False)
export REDIS_PASSWORD=<your-password>  # Optional Redis password, in case the Redis instance on your machine is password protected
```

Default Redis configuration if not specified:
- Host: localhost
- Port: 6379
- No authentication

You can monitor what is being read from or written to Redis by running `redis-cli` and then `MONITOR`.

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
Run tests as a Python module (with 6 parallel workers, and verbose output) using:
```bash
python -m pytest -v -s -n 6
```

Certain tests are inherently slow, including all tests regarding the batch API. We disable them by default to avoid slowing down the CI pipeline. To run them, use:
```bash
SAFETYTOOLING_SLOW_TESTS=True python -m pytest -v -s -n 6
```

### Updating dependencies
We only pin top-level dependencies only to make cross-platform development easier.

- To add a new python dependency, add a line to `requirements.txt`.
- To upgrade a dependency, bump the version number in `requirements.txt`.

Run `pip install -r requirements.txt` after edits have been made to the requirements file.

To check for outdated dependencies, run `pip list --outdated`.

## Usage

### Running Inference

* See `examples/inference_api/inference_api.ipynb` for an example of how to use the Inference API.
* See `examples/anthropic_batch_api/run_anthropic_batch.py` for an example of how to use the Anthropic Batch API and how to set up command line input arguments using `simple_parsing` and `ExperimentConfigBase` (a useful base class we created for this project).

### Running Finetuning

To launch a finetuning job, run the following command:
```bash
python -m safetytooling.apis.finetuning.openai.run --model 'gpt-3.5-turbo-1106' --train_file <path-to-train-file> --n_epochs 1
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


## Features

- **LLM Inference API:**
    - Location:`safetytooling/apis/inference/api.py`
    - **Caching Mechanism:**
        - Caches prompt calls to avoid redundant API calls. Cache location defaults to `$exp_dir/cache`. This means you can kill your run anytime and restart it without worrying about wasting API calls.
        - **Redis Cache:** Optionally use Redis for caching instead of files by setting `REDIS_CACHE=true` in your environment. Configure Redis connection with:
            - `REDIS_PASSWORD`: Optional Redis password for authentication
            - Default Redis configuration: localhost:6379, no password
        - The cache implementation is automatically selected based on the `REDIS_CACHE` environment variable.
        - **No Cache:** Set `NO_CACHE=True` as an environment variable to disable all caching. This is equivalent to setting `use_cache=False` when initialising an InferenceAPI or BatchInferenceAPI object.
    - **Prompt Logging:** For debugging, human-readable `.txt` files are can be output in `$exp_dir/prompt_history` and timestamped for easy reference (off by default). You can also pass `print_prompt_and_response` to the api object to print coloured messages to the terminal.
    - Ability to double the rate limit if you pass a list of models e.g. `["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]`
    - Manages rate limits efficiently for OpenAI bypassing the need for exponential backoff.
    - Number of concurrent threads can be customised when initialising the api object for each provider (e.g. `openai_num_threads`, `anthropic_num_threads`). Furthermore, the fraction of the OpenAI rate limit can be specified (e.g. only using 50% rate limit by setting `openai_fraction_rate_limit=0.5`.
    - When initialising the API it checks how much OpenAI rate limit is available and sets caps based on that.
    - Allows custom filtering of responses via `is_valid` function and will retry until a valid one is generated (e.g. ensuring json output).
    - Provides a running total of cost for OpenAI models and model timings for performance analysis.
    - Utilise maximum rate limit by setting `max_tokens=None` for OpenAI models.
    - Supports OpenAI moderation, embedding and Realtime (audio) API
    - Supports OpenAI chat/completion models, Anthropic, Gemini (all modalities), GraySwan, HuggingFace inference endpoints (e.g. for llama3).
- Prompt, LLMResponse and ChatMessage class
    - Location:`safetytooling/data_models/messages.py`
    - A unified class for prompts with methods to unpack it into the format expected by the different providers.
        - Take a look at the `Prompt` class to see how the messages are transformed into different formats. This is important if you ever need to add new providers or modalities.
    - Supports image and audio.
    - The objects are pydantic and inherit from a hashable base class (useful for caching since you can hash based on the prompt class)
- **Finetuning runs with Weights and Biases:**
    - Location: `safetytooling/apis/finetuning/run.py`
    - Logs finetuning runs with Weights and Biases for easy tracking of experiments.
- **Text to speech**
    - Location: `safetytooling/apis/tts/elevenlabs.py`
- **Usage Tracking:**
    - Location: `safetytooling/apis/inference/usage`
    - Tracks usage of OpenAI and Anthropic APIs so you know how much they are being utilised within your organisation.
- **Experiment base class**
    - Location: `safetytooling/utils/experiment_utils.py`
    - This dataclass is used as a base class for each experiment script. It provides a common set of args that always need to be specified (e.g. those that initialise the InferenceAPI class)
    - It creates the experiment directory and sets up logging to be output into log files there. It also sets random seeds and initialises the InferenceAPI so it is accessible easily by using `cfg.api`.
    - See examples of usage in the `examples` repo in next section.
- **API KEY management**
    - All API keys get stored in the SECRETS file (that is in the .gitignore)
    - `setup_environment()` in `safetytooling/uils/utils.py` loads these in so they are accessible by the code (and also automates exporting environment variables)
- **Utilities:**
    - Plotting functions for confusion matrices and setting up plotting in notebooks (`plotting_utils.py`)
    - Prompt loading with a templating library called Jinja (`prompt_utils.py`)
    - Image/audio processing (`image_utils.py` and `audio_utils.py`)
    - Human labelling framework which keeps a persistent store of human labels on disk based on the input/output pair of the LLM (`human_labeling_utils.py`)