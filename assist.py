import os
OPENAI_API_KEY= "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def upload_file(file_path):
    file_to_upload = client.files.create( file=open(file_path, "rb"), purpose='assistants')
    return file_to_upload 

 
transformer_paper_path = "./data/transformer_paper.pdf"
file_to_upload = upload_file(transformer_paper_path)

def create_assistant(assistant_name,
                 	my_instruction,
                 	uploaded_file,
                 	model="gpt-3.5-turbo-instruct"):
    
	my_assistant = client.beta.assistants.create(
	name = assistant_name,
	instructions = my_instruction,
	model="gpt-3.5-turbo-instruct",
	tools=[{"type": "retrieval"}],
	file_ids=[uploaded_file.id]
	)
    
	return my_assistant