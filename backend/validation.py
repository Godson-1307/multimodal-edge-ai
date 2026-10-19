REQUIRED_FIELDS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "gstin",
    "state",
    "state_code",
    "total_tax",
    "total_amount"
]


def validate_extraction(data):

    missing_fields = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            missing_fields.append(field)

    return missing_fields