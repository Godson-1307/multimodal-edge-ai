from json_utils import parse_json_response
from validation import validate_extraction

response = '''```json
{
    "vendor_name": "SHABREZ ALI",
    "invoice_number": "725",
    "invoice_date": "11/07/2026",
    "gstin": "27AGFPN1247B1ZS",
    "state": "Maharashtra",
    "state_code": "27",
    "total_tax": null,
    "total_amount": null
}
```'''

data = parse_json_response(response)

missing = validate_extraction(data)

print("\nValidation:")

if missing:
    print("Missing fields:", missing)
else:
    print("All required fields are present.")

print("Parsed data:")
print(data)

print("\nIndividual fields:")
print("Vendor:", data["vendor_name"])
print("Invoice:", data["invoice_number"])
print("GSTIN:", data["gstin"])