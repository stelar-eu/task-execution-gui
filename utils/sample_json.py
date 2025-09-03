sample_task = {
    "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
    "name": "Agri Products Match",
    "image": "petroud/agri-products-match:latest",
    "inputs": {
        "npk_values": ["325fb7c7-b269-4a1e-96f6-a861eb2fe325"],
        "fertilizer_dataset": ["41da3a81-3768-47db-b7ac-121c92ec3f6d"],
    },
    "datasets": {"d0": "2f8a651b-a40b-4edd-b82d-e9ea3aba4d13"},
    "parameters": {"mode": "fertilizers"},
    "outputs": {
        "matched_fertilizers": {
            "url": "s3://abaco-bucket/MATCH/matched_fertilizers.csv",
            "dataset": "d0",
            "resource": {
                "name": "Matched Fertilizers based on NPK values",
                "relation": "matched",
            },
        }
    },
}
