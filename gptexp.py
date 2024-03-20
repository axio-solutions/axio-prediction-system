import pandas as pd
import openai
import os

# Set your OpenAI API key here
openai.api_key = "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"

def read_csv(file_path):
    """Reads a CSV file and returns its contents as a pandas DataFrame."""
    data = pd.read_excel(file_path)
    return data

def generate_insights(data):
    openai.api_key = "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"
    """Generates insights from the DataFrame using OpenAI's GPT."""
    # Convert your DataFrame into a summary or a series of questions you want insights on
    summary = data.describe().to_string()
    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",  # You may want to use the latest model here
            prompt=f"Tell me what you understand from this data and talk about missing values. Answer in 100 words. Here is the data\n:{data}",
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating insights: {e}")
        return None

def main():
    # Specify the path to your CSV file
    file_path = 'temp/DecCampaign.xlsx'
    
    # Read the CSV file
    data = read_csv(file_path)
    
    # Generate insights
    insights = generate_insights(data)
    
    if insights:
        print("Generated Insights:")
        print(insights)
    else:
        print("Failed to generate insights.")

if __name__ == "__main__":
    main()
