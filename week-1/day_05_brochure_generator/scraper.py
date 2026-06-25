import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore

# Standard browser headers to prevent websites from blocking our requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def fetch_website_links(url):
    """Fetches a webpage and returns a list of all
    hyperlinks (URLs) found on it."""

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Raise error for bad responses (404, 500)

        soup = BeautifulSoup(response.content, "html.parser")

        links = []

        # find all <a> tags that have an 'href' attribute
        for a_tag in soup.find_all("a", href=True):
            links.append(a_tag["href"])

        return links

    except Exception as e:
        print(f"Error fetching links from {url} : {e}")
        return []


def fetch_website_content(url):
    """Fetches a webpage, strips HTML boilerplate, and
    returns cleaned text."""

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.title.string if soup.title else "No title found"

        # Remove script and style tags so we don't send code to the LLM
        if soup.body:
            for element in soup.body(["script", "style", "img", "input", "footer", "nav"]):
                element.decompose()
            # get the cleaned text content
            text = soup.get_text(separator="\n", strip=True)
        else:
            text=""

        return (title +"\n\n" + text)[:2_000]

    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""
