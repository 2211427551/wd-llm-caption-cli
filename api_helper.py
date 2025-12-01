import sys
import json
import base64
from openai import OpenAI

def list_models(api_endpoint, api_key):
    try:
        client = OpenAI(
            api_key=api_key if api_key != "None" else None,
            base_url=api_endpoint,
            timeout=30.0,
        )
        models_response = client.models.list()
        model_names = [model.id for model in models_response.data if hasattr(model, 'id')]
        print(json.dumps(model_names))
    except Exception as e:
        error_data = {
            "error": f"An exception of type {type(e).__name__} occurred: {str(e)}"
        }
        print(json.dumps(error_data))
        sys.exit(1)

def get_caption(api_endpoint, api_key, model, system_prompt, user_prompt, image_base64, temperature, max_tokens):
    try:
        client = OpenAI(
            api_key=api_key if api_key != "None" else None,
            base_url=api_endpoint,
            timeout=180.0, # Increased timeout for potentially long-running streams
        )
        
        messages = []
        if system_prompt and system_prompt != "None":
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        if image_base64 and image_base64 != "None":
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens) if int(max_tokens) > 0 else 1024,
            "stream": False # Explicitly disable streaming
        }
        
        if float(temperature) > 0:
            params["temperature"] = float(temperature)
        
        response = client.chat.completions.create(**params)
        
        full_content = response.choices[0].message.content
        
        print(json.dumps({"caption": full_content}))

    except Exception as e:
        error_data = {
            "error": f"An exception of type {type(e).__name__} occurred: {str(e)}"
        }
        print(json.dumps(error_data))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "A command is required: 'list_models' or 'get_caption'"}))
        sys.exit(1)

    command = sys.argv[1]
    if command == "list_models":
        if len(sys.argv) != 4:
            print(json.dumps({"error": "Usage: python api_helper.py list_models <api_endpoint> <api_key>"}))
            sys.exit(1)
        list_models(sys.argv[2], sys.argv[3])
    
    elif command == "get_caption":
        if len(sys.argv) != 3:
            print(json.dumps({"error": "Usage: python api_helper.py get_caption <path_to_json_input>"}))
            sys.exit(1)
        
        input_file_path = sys.argv[2]
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            get_caption(
                input_data['api_endpoint'],
                input_data['api_key'],
                input_data['model'],
                input_data['system_prompt'],
                input_data['user_prompt'],
                input_data['image_base64'],
                input_data['temperature'],
                input_data['max_tokens']
            )
        except Exception as e:
            print(json.dumps({"error": f"Failed to read from file or execute get_caption: {str(e)}"}))
            sys.exit(1)
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)
