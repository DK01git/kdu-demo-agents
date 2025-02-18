import os
from dotenv import load_dotenv, find_dotenv
from functools import lru_cache

# tload env diel                                                                                                                    
def load_env():
    _ = load_dotenv(find_dotenv())

def get_groc_api_key():
    load_env()
    groc_api_key = os.getenv("AZURE_KEY")
    if not groc_api_key:
        raise ValueError("No GROQ_API_KEY set for Groc API")
    return groc_api_key

def get_serper_api_key():
    load_env()
    groc_api_key = os.getenv("SERPER_API_KEY")
    return groc_api_key

def get_azure_credentials():
    """Get Azure API key and endpoint from environment variables."""
    
    load_env()
    
    azure_key = os.getenv("AZURE_KEY")
    azure_endpoint = os.getenv("Azure_ENDPOINT")
    
    if not azure_key or not azure_endpoint:
        raise ValueError("Azure credentials not found in environment variables")
        
    return azure_key, azure_endpoint

# break to 50
# don't break in the middle 
def pretty_print_result(result):
  parsed_result = []
  for line in result.split('\n'):
      if len(line) > 50:
          words = line.split(' ')
          new_line = ''
          for word in words:
              if len(new_line) + len(word) + 1 > 50:
                  parsed_result.append(new_line)
                  new_line = word
              else:
                  if new_line == '':
                      new_line = word
                  else:
                      new_line += ' ' + word
          parsed_result.append(new_line)
      else:
          parsed_result.append(line)
  return "\n".join(parsed_result)

@lru_cache(maxsize=100)
def cached_search(query: str):
    # Your search logic here
    return result

class SearchTool:
    def __call__(self, search_query: str):
        return cached_search(search_query)

# Cache environment variables
class EnvironmentCache:
    _instance = None
    _credentials = None
    
    @classmethod
    def get_credentials(cls):
        if cls._credentials is None:
            cls._credentials = {
                'azure_key': os.getenv("AZURE_KEY"),
                'azure_endpoint': os.getenv("Azure_ENDPOINT"),
                'serper_key': os.getenv("SERPER_API_KEY")
            }
        return cls._credentials

# Use in your code
credentials = EnvironmentCache.get_credentials()
