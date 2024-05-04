import os
import boto3

# Create a boto3 session and initialize the Bedrock client
boto3_session = boto3.session.Session()
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

# Retrieve the Knowledge Base ID from environment variables
kb_id = os.environ.get("KNOWLEDGE_BASE_ID")

def retrieve(input_text, kbId):
    print("Retrieving information for:", input_text, "from KB:", kbId)
    # Correct the request to match the API's expected parameters
    response = bedrock_agent_runtime_client.retrieve(
        knowledgeBaseId=kbId,
        retrievalQuery={
            'text': input_text
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                # Include default values or configure as necessary
                'numberOfResults': 5,  # Example: set to 10 results, adjust as needed
                # 'filter': {},  # Uncomment and set up if there is a specific filter needed
                # 'overrideSearchType': 'YOUR_TYPE_HERE'  # Uncomment and adjust if needed
            }
        }
    )
    return response

def lambda_handler(event, context):
    if 'question' not in event:
        return {
            'statusCode': 400,
            'body': 'No question provided.'
        }

    query = event['question']
    response = retrieve(query, kb_id)
    print(response)

    return {
        'statusCode': 200,
        'body': {
            "question": query.strip(),
            "answer": response
        }
    }
