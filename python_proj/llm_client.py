import requests
import config
import time
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# Provider Delay Handling
# -------------------------------
def get_provider_delay(provider):
    # Only Godfather supported now
    return config.GODFATHER_DELAY_SECONDS

# -------------------------------
# Main LLM Call Function
# -------------------------------
def call_llm(model_spec, prompt, system_prompt=None):
    """
    model_spec format: provider/model_name
    Supported providers: godfather
    """

    if "/" in model_spec:
        provider, model_name = model_spec.split("/", 1)
    else:
        provider = "godfather"
        model_name = model_spec

    
    # Force provider to be godfather if not specified (legacy safety, though all should be updated)
    if provider != "godfather":
        logger.warning(f"⚠️ Provider '{provider}' not supported, switching to 'godfather'")
        provider = "godfather"

    logger.info(f"🤖 Calling {provider}/{model_name} (prompt length: {len(prompt)} chars)")

    delay = get_provider_delay(provider)
    if delay > 0:
        time.sleep(delay)

    start_time = time.time()

    for attempt in range(config.MAX_RETRIES):
        try:
            # ==========================================================
            # GODFATHER (REMOTE CUSTOM API)
            # ==========================================================
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            payload = {
                "query": full_prompt,
                "model_name": model_name,
                "stream": False
            }
            
            response = requests.post(
                config.GODFATHER_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            # Simple error handling for non-200
            if response.status_code != 200:
                logger.error(f"❌ API Error: {response.status_code} - {response.text}")
                response.raise_for_status()

            # Result parsing
            try:
                result = response.json()
                # Try common keys
                text = result.get("response") or result.get("output") or result.get("text") or str(result)
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Response received in {elapsed:.1f}s")
                return text

            except ValueError:
                # If not JSON, return text
                return response.text

        except Exception as e:
            logger.error(f"❌ Error calling {provider}/{model_name}: {e}")
            if attempt < config.MAX_RETRIES - 1:
                wait = (config.RETRY_BACKOFF_FACTOR ** attempt) * 2
                logger.warning(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return ""