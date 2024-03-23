######################## Libraries ########################
### Python Modules 
import time
from functools import wraps
import logging


######################## Fucntions ########################
def retry_with_exponential_backoff(max_attempts=3, initial_delay_seconds=1, backoff_factor=2):
    """
    A decorator that retries a function up to max_attempts with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay_seconds
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_attempts:
                        logging.error(f"Final attempt ({attempt}) failed. No more retries.")
                        raise  # Re-raise the last exception
                    else:
                        logging.error(f"Attempt {attempt} failed with error: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
        return wrapper
    return decorator

def timeit(func):
    """
    A decorator that times the execution of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"Function {func.__name__} took {elapsed:.2f} seconds.")
        return result
    return wrapper