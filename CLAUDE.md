# Claude Code Guide - Safety Tooling

This guide helps Claude Code work effectively with the safety-tooling repository. For general project information, refer to README.md.

## Quick Context

This is an AI safety research toolkit that provides a unified API for multiple LLM providers. The codebase prioritizes stability, caching efficiency, and ease of use across research projects.

## Key Principles When Working on This Codebase

1. **Always preserve backward compatibility** - This is used as a submodule by many projects
2. **Caching is critical** - Never bypass caching without explicit user request
3. **API keys are sacred** - Never log or expose API keys, always use the `.env` pattern
4. **Rate limits matter** - Respect the built-in rate limiting mechanisms

## Common Tasks and Patterns

### Adding Support for a New Model

When a user asks to add support for a new model:

1. **First check if it works with `force_provider`** - Many new models work without code changes:
   ```python
   # Try this first before modifying any code
   response = await API(
       model_id="new-model-name",
       prompt=prompt,
       force_provider="openai"  # or "anthropic" if compatible
   )
   ```

2. **Only modify code if force_provider doesn't work** - Look for provider detection logic in the inference API

### Working with Caching

**DO:**
- Use the existing cache_dir pattern
- Respect NO_CACHE environment variable
- Maintain cache key compatibility

**DON'T:**
- Create new caching mechanisms
- Modify cache key generation without careful consideration
- Clear caches without explicit user request

### Handling API Keys

**Pattern to follow:**
```python
# Always use utils.setup_environment()
utils.setup_environment()

# For multiple keys, use the tag system
utils.setup_environment(
    openai_tag="OPENAI_API_KEY_PROJECT_X",
    anthropic_tag="ANTHROPIC_API_KEY_PROJECT_X"
)
```

**Never:**
- Hardcode API keys
- Create new environment variable patterns
- Log API keys in any form

### Creating Examples

When writing example code:
1. Always use async/await patterns (the API is async)
2. Include proper error handling
3. Use Path objects for file paths: `Path(".cache")`
4. Follow the established pattern from existing examples

### Modifying Core Components

**Before modifying any core files, check:**
- Can this be done with existing extension points?
- Will this break existing cached responses?
- Does this maintain backward compatibility?

**Core files to be extra careful with:**
- `safetytooling/apis/inference/api.py` - The main API
- `safetytooling/data_models/messages.py` - Data structures
- Cache-related files - Breaking changes invalidate caches

## Quick Reference

### Repository Structure

For a comprehensive understanding of this repository's structure, codebase organization, and development guidelines, please refer to the [REPO_GUIDE.md](./REPO_GUIDE.md). This guide contains detailed information about:

- Project architecture and module organization
- Development workflows and best practices  
- Contributing guidelines
- Testing strategies

Read this file if you need to understand the structure of the repo in depth.

### Standard Inference Pattern
```python
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path

# Always start with this
utils.setup_environment()
API = InferenceAPI(cache_dir=Path(".cache"))

# Create prompts using the data models
prompt = Prompt(messages=[
    ChatMessage(content="Your message", role=MessageRole.user)
])

# Make API calls
response = await API(
    model_id="gpt-4o-mini",
    prompt=prompt,
    print_prompt_and_response=True,  # For debugging
)
```

### Debugging Issues

1. **Rate limit errors**: Check `openai_fraction_rate_limit` and `num_threads` settings
2. **Cache misses**: Verify cache_dir path and NO_CACHE environment variable
3. **API key issues**: Ensure `.env` file exists and `setup_environment()` is called
4. **Import errors**: User should install with `uv pip install -e .`

### Testing Changes

Always run tests before submitting:
```bash
# Quick tests
python -m pytest -v -s -n 6

# Full test suite (if modifying core functionality)
SAFETYTOOLING_SLOW_TESTS=True python -m pytest -v -s -n 6
```

## Important Conventions

### File Organization
- Examples go in `examples/` with descriptive subdirectories
- Keep provider-specific code isolated when possible
- Utilities should be general-purpose and well-tested

### Error Messages
- Be specific about what went wrong
- Suggest concrete fixes
- Point to relevant documentation or examples

### Documentation
- Update docstrings when modifying functions
- Add type hints to new code
- Include usage examples in docstrings for public APIs

## What to Avoid

1. **Breaking changes to Prompt or ChatMessage classes** - These are used everywhere
2. **Modifying cache key generation** - This invalidates existing caches
3. **Adding required parameters to existing functions** - Use optional parameters with defaults
4. **Creating new environment variable patterns** - Use the existing .env system
5. **Synchronous code in the API path** - Everything should be async

## When Stuck

1. Check existing examples in the `examples/` directory
2. Look for similar patterns in the codebase
3. Verify your approach maintains backward compatibility
4. Consider if the task can be done without modifying core files

## Special Considerations

### Working with Batch APIs
- Batch processing has separate examples and patterns
- Don't mix batch and regular inference patterns
- Respect batch API specific rate limits

### Multi-provider Support
- Each provider has different message format requirements
- The Prompt class handles conversion - don't bypass it
- Test with multiple providers when modifying core functionality

### Redis Caching
- Redis is optional - code must work with file-based caching too
- Check REDIS_CACHE environment variable
- Don't assume Redis is available

## Remember

This codebase is used by multiple AI safety research projects. Stability and reliability are more important than new features. When in doubt, ask the user for clarification rather than making breaking changes.
