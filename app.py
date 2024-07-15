import chainlit as cl
import pandas as pd
import json
import requests
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models import Model
from tabulate import tabulate

# Load environment variables from .env file
load_dotenv()

# IBM LLM Model configuration
parameters = {
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.REPETITION_PENALTY: 1
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("IBM_API_KEY")
}

project_id = os.getenv("PROJECT_ID")
model_id = ModelTypes.LLAMA_3_8B_INSTRUCT

model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id
)

# Transform to JSON function
def transform_to_json(file_path):
    fields = [
        "CUSTOMER_ID", "OVERDUE_BALANCE", "BASE_USAGE", "CREDIT_HISTORY",
        "ALTERNATE_USAGE", "STANDING_CHARGE", "BASE_CHARGE", "ALTERNATE_CHARGE",
        "LEVY", "TAX", "TOTAL_NET", "TOTAL_TO_PAY", "AGE", "IS_REGISTERED_FOR_ALERTS",
        "OWNS_HOME", "COMPLAINTS", "HAS_THERMOSTAT", "HAS_HOME_AUTOMATION",
        "PV_ZONING", "WIND_ZONING", "SMART_METER_COMMENTS", "IS_CAR_OWNER",
        "HAS_EV", "HAS_PV", "HAS_WIND", "EBILL", "IN_WARRANTY", "CITY",
        "MARITAL_STATUS", "EDUCATION", "SEGMENT", "EMPLOYMENT", "BUILDING_TYPE",
        "BILLING_MONTH", "RATIO_THIS_MONTH_BASE_USAGE_VS_LAST_MONTH",
        "RATIO_THIS_MONTH_BASE_USAGE_VS_AVG_LOOKBACK_WINDOW",
        "RATIO_THIS_MONTH_ALTERNATE_USAGE_VS_LAST_MONTH",
        "RATIO_THIS_MONTH_ALTERNATE_USAGE_VS_AVG_LOOKBACK_WINDOW",
        "RATIO_THIS_MONTH_TOTAL_TO_PAY_VS_LAST_MONTH",
        "RATIO_THIS_MONTH_TOTAL_TO_PAY_VS_AVG_LOOKBACK_WINDOW",
        "NUM_MISSED_PAYMENTS_LOOKBACK_WINDOW", "BILLING_MONTH_NUMBER"
    ]
    
    df = pd.read_csv(file_path)
    values = df[fields].values.tolist()
    
    json_data = {
        "input_data": [
            {
                "fields": fields,
                "values": values
            }
        ]
    }
    return json_data

# Call Predictor API function
def call_predictor_api(json_data):
    API_KEY = os.getenv("IBM_API_KEY")
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    print("token_response:", token_response)
    mltoken = token_response.json()["access_token"]
    print("token:", mltoken)

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # Convert the json_data dictionary to a JSON string
    json_payload = json.dumps(json_data)

    response = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e8d4bf0e-1852-4711-8825-ca562ba8289a/predictions?version=2021-05-01', data=json_payload, headers=headers)
    print("response", response)
    print("response json", response.json())
    return response.json()

# Generate Report function
def generate_report(prediction_result):
    print("prediction_result:", json.dumps(prediction_result, indent=2))
    instruction = "Based on the following prediction results, generate a detailed report."
    prompt1 = "\n".join([instruction, "Report:", json.dumps(prediction_result, indent=2)])
    print(prompt1)
    
    model_details = model.get_details()
    # print("model_details:", model_details)

    response = model.generate_text(prompt=prompt1)
    print("Raw response from generate_text:", response)
    
    if not response:
        print("Error: Empty response from generate_text")
        return "Error: No response from the model."

    try:
        response_json = json.loads(response)
        return response_json.get('generated_text', response)
    except json.JSONDecodeError as e:
        print(f"Error parsing the response JSON: {e}")
        return response

def generate_report_and_table(prediction_result, csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Extract customer IDs and relevant fields from the DataFrame
    customer_ids = df["CUSTOMER_ID"].tolist()
    overdue_balances = df["OVERDUE_BALANCE"].tolist()
    base_usages = df["BASE_USAGE"].tolist()
    credit_histories = df["CREDIT_HISTORY"].tolist()
    
    # Extract predictions and probabilities from the prediction result
    predictions = prediction_result["predictions"][0]["values"]
    
    # Generate the report
    prediction_groups = {}
    for i, customer_id in enumerate(customer_ids):
        prediction = predictions[i][0]
        if prediction not in prediction_groups:
            prediction_groups[prediction] = []
        prediction_groups[prediction].append(customer_id)
    
    report_lines = []
    for prediction, customers in prediction_groups.items():
        customer_str = ", ".join(map(str, customers))
        if len(customers) == 1:
            if prediction == "Missed Payment":
                report_lines.append(f"The model is predicting that customer {customer_str} is likely to miss the payments.")
            else:
                report_lines.append(f"The model is predicting that customer {customer_str} is likely to pay on time the payments.")
        else:
            if prediction == "Missed Payment":
                report_lines.append(f"The model is predicting that customers {customer_str} are likely to miss the payments.")
            else:
                report_lines.append(f"The model is predicting that customers {customer_str} are likely to pay on time the payments.")
    
    report = " ".join(report_lines)
    
    # Create a DataFrame for the table
    table_data = {
        "Customer": customer_ids,
        "Prediction": [predictions[i][0] for i in range(len(customer_ids))],
        "Probability": [max(predictions[i][1]) for i in range(len(customer_ids))],
        "Overdue Balance": overdue_balances,
        "Base Usage": base_usages,
        "Credit History": credit_histories
    }
    
    table_df = pd.DataFrame(table_data)
    
    return report, table_df

# Convert DataFrame to Markdown table
def dataframe_to_markdown(df):
    return tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

# Chainlit app
@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a CSV file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a CSV file with customer information to begin!", accept=["text/csv"]
        ).send()

    csv_file = files[0]

    # Transform CSV file to JSON
    json_data = transform_to_json(csv_file.path)
    print("JSON data:", json.dumps(json_data, indent=2))

    # Call Predictor API
    prediction_result = call_predictor_api(json_data)
    print("Prediction result:", json.dumps(prediction_result, indent=2))

    # Generate Report and Table
    report, table_df = generate_report_and_table(prediction_result, csv_file.path)
    print("Report:", report)
    print("Table:")
    print(table_df)

    # Convert table to Markdown
    markdown_table = dataframe_to_markdown(table_df)
    
    # Store the report and table in the user session
    cl.user_session.set("report", report)
    cl.user_session.set("table_df", table_df)

    # Let the user know that the system is ready and provide the prediction result
    await cl.Message(
        content=f"`{csv_file.name}` uploaded and processed."
    ).send()

    # Provide the generated report
    await cl.Message(
        content=f"Generated Report: {report}\n\n{markdown_table}"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    report = cl.user_session.get("report")
    table_df = cl.user_session.get("table_df")
    print("report:", report)

    if not report:
        await cl.Message(
            content="No report found. Please upload a CSV file and generate a report first."
        ).send()
        return

    # Convert table to Markdown
    markdown_table = dataframe_to_markdown(table_df)

    # Generate a response to the user's question based on the report
    prompt = f"Using the following report: {report}, and the table: {markdown_table}, answer the question: {message.content}"
    response = model.generate_text(prompt=prompt)
    print("Response from LLM:", response)

    # Send the answer to the user
    await cl.Message(
        content=response
    ).send()

# Main function to handle user queries
@cl.on_message
async def main(message: cl.Message):
    report = cl.user_session.get("report")
    table_df = cl.user_session.get("table_df")
    print("report:", report)

    if not report:
        await cl.Message(
            content="No report found. Please upload a CSV file and generate a report first."
        ).send()
        return

    # Convert table to Markdown
    markdown_table = dataframe_to_markdown(table_df)

    # Generate a response to the user's question based on the report
    prompt = f"Using the following report: {report}, and the table: {markdown_table}, answer the question: {message.content}"
    response = model.generate_text(prompt=prompt)
    print("Response from LLM:", response)

    # Send the answer to the user
    await cl.Message(
        content=response
    ).send()


