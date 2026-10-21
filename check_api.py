"""One short API request. Run with .venv/Scripts/python.exe check_api.py."""
import getpass
import os
import sys

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def main():
    key = os.getenv("OPENAI_API_KEY") or getpass.getpass("API key (hidden input): ")
    if not key.strip():
        print("No API key entered.")
        return 1
    base_url = os.getenv("OPENAI_BASE_URL") or "https://changellenge.up.railway.app/v1"
    model = os.getenv("OPENAI_MODEL") or "qwen/qwen3.8-27b"
    print(f"Sending one short request. Model: {model}")
    try:
        with OpenAI(api_key=key.strip(), base_url=base_url, timeout=60, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with only: OK"}],
                max_tokens=128,
            )
        print("SUCCESS: API returned a response.")
        print("Response:", response.choices[0].message.content or "(empty text)")
        if response.usage:
            print("Tokens:", response.usage.total_tokens)
        print("The key was not saved. This check does not configure solution.py.")
        return 0
    except APITimeoutError:
        print("Request timed out after 60 seconds.")
    except APIConnectionError:
        print("Connection failed. Check network access and OPENAI_BASE_URL.")
    except APIStatusError as error:
        print(f"API returned HTTP {error.status_code}.")
        # Do not print raw server errors or request headers containing credentials.
        if error.status_code == 401:
            print("The API key was rejected. Check the key.")
        elif error.status_code == 429:
            print("Request or budget limit reached. Check the gateway budget.")
        elif error.status_code == 402:
            print("Check the gateway budget/billing limit.")
        elif error.status_code >= 500:
            print("Gateway or model server error. Retry later.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
