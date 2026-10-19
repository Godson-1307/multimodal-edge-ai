import json


def parse_json_response(response):

    response = response.strip()

    # Remove Markdown code block
    if response.startswith("```json"):
        response = response[7:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    return json.loads(response)