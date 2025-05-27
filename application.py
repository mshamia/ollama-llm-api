from flask import Flask, request, jsonify, Response
from ollama import chat, pull, list, show
from ollama import ChatResponse
import re
import os
import threading
import time
import requests
import json
# For tracking download status
model_download_status = {}

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running properly.
    Returns status information about the API and Ollama service.
    """
    try:
        # Check if Ollama service is available by trying to list models
        models_response = list()
        models_count = len(models_response.get('models', []))
        
        # Get uptime
        uptime = time.time() - app.start_time if hasattr(app, 'start_time') else 0
        
        return jsonify({
            'status': 'healthy',
            'service': 'ollama-api',
            'uptime_seconds': uptime,
            'ollama_status': 'connected',
            'models_available': models_count,
            'timestamp': time.time()
        })
    except Exception as e:
        # If we can't connect to Ollama, return a degraded status
        return jsonify({
            'status': 'degraded',
            'service': 'ollama-api',
            'ollama_status': 'disconnected',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@app.route('/chat', methods=['POST'])
def chat_with_llm():
    try:
        # Get request data
        data = request.json
        
        # Extract parameters from request
        model = data.get('model', 'llama3.2')  # Default model if not specified
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Prepare messages for ollama
        messages = [
            {
                'role': 'user',
                'content': user_message
            }
        ]
        
        # Get chat history if provided
        history = data.get('history', [])
        if history:
            messages = history + messages
        
        # Call the Ollama API
        response: ChatResponse = chat(model=model, messages=messages)
        
        # Return the response
        # Convert the response to a serializable format
        serializable_response = {
            'message': {
                'role': response.message.role,
                'content': response.message.content
            },
            'model': response.model,
            'created_at': response.created_at,
            'done': response.done
        }
        
        return jsonify({
            'response': response.message.content,
            'model': model,
            'full_response': serializable_response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance-prompt', methods=['POST'])
def enhance_prompt():
    try:
        # Get request data
        data = request.json
        
        # Extract base prompt
        base_prompt = data.get('prompt', '')
        
        if not base_prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Prepare system instruction for prompt enhancement
        system_instruction = """You are a prompt enhancement assistant. Your task is to take a vague or low-detail user input and rewrite it into a high-quality, visually rich prompt suitable for AI image or video generation models.\n\nYour Goal:\nTransform the input into a natural, fluent, and richly descriptive prompt that includes:\n- A clear subject and action (only if the subject is mentioned in the input)\n- A vivid setting or environment\n- Descriptive stylistic and mood cues\n- Optional camera or motion directions IF I added this after the input user prompt: (VIDEO)\n\nThe goal is to help AI models generate stunning, visually coherent content based on a strong mental image, while being faithful to the user's input and respectful of cultural values.\n\nInstructions:\n1. Identify and clarify the main subject.\n   - Rewrite vague references into specific, concrete characters, objects, or scenes.\n   - Do NOT invent or add people, animals, or characters if not clearly mentioned.\n   - ✅ Example: “a man” → “a man wearing a white thobe, holding a Ramadan lantern while walking down a lantern-lit alley”\n   - ❌ Counter-example: “photo of Mecca” → do not add “a man walking through the streets”\n\n2. Add relevant context and setting.\n   - ✅ Example: “in a desert” → “in a vast Arabian desert at sunrise, golden light glows over the sand dunes”\n   - ✅ Example: “in a mosque” → “inside an ornate Ottoman-style mosque, sunlight streams through stained glass windows”\n\n3. Enhance with visual and sensory detail.\n   - ✅ Example: “a mountain” → “a snow-covered mountain under a clear blue sky, with falcons soaring above the peak”\n\n4. Include stylistic and aesthetic cues.\n   - Use artistic or photographic styles such as: “digital painting,” “cinematic photo,” “watercolor,” “3D render,” etc.\n   - ✅ Example: “a city balcony” → “a stone balcony overlooking the narrow alleys of old Damascus, in the style of a realistic oil painting”\n\n5. For video prompts (if the input ends with (VIDEO)):\n   - ✅ Example: “an old alley (VIDEO)” → “the camera slowly pans through a narrow alley lit by hanging lanterns in the evening”\n\n6. Keep the prompt fluent and natural.\n   - Use full descriptive sentences in either English or Arabic, but keep examples in English.\n\n7. Be culturally aware and modest.\n   - ✅ Example: “a man on the beach” → “a man wearing a long robe with wide sleeves, standing peacefully on the shore at sunset”\n\n8. Focus on what to include, not what to avoid.\n   - ✅ Instead of saying “no blur,” say “sharp focus,” “clean background,” or “realistic lighting”\n\n9. Be creative but faithful to the user's intent.\n   - If the input is very vague (e.g., “a city,” “a bird”), expand it with creative visuals — but never add new characters or subjects unless explicitly mentioned
"""
        
        # Prepare messages for ollama
        messages = [
            {
                'role': 'system',
                'content': system_instruction
            },
            {
                'role': 'user',
                'content': f"Original prompt: \"{base_prompt}\"\n\nEnhance this prompt for image generation. YOUR RESPONSE MUST BE IN ENGLISH ONLY, regardless of the language of the original prompt."
            }
        ]
        
        # Get model preference if provided
        model = data.get('model', 'llama3.2')
        
        # Call the Ollama API
        response = chat(model=model, messages=messages)
        
        # Extract just the enhanced prompt text (remove any explanations)
        enhanced_prompt = response.message.content
        
        # More aggressive cleaning of the enhanced prompt
        clean_prompt = re.sub(r'```.*?```', '', enhanced_prompt, flags=re.DOTALL)
        clean_prompt = re.sub(r'"', '', clean_prompt)  # Remove quotes
        clean_prompt = re.sub(r'Here\'s an enhanced version( of the prompt)?:?', '', clean_prompt, flags=re.IGNORECASE)
        clean_prompt = re.sub(r'Enhanced prompt:?', '', clean_prompt, flags=re.IGNORECASE)
        clean_prompt = re.sub(r'Here is the enhanced prompt:?', '', clean_prompt, flags=re.IGNORECASE)
        clean_prompt = clean_prompt.strip()
        
        # Convert the response to a serializable format
        serializable_response = {
            'message': {
                'role': response.message.role,
                'content': response.message.content
            },
            'model': response.model,
            'created_at': response.created_at,
            'done': response.done
        }
        
        # Return the response with both original and enhanced prompts
        return jsonify({
            'original_prompt': base_prompt,
            'enhanced_prompt': clean_prompt,
            'model': model 
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    try:
        # Get models from Ollama
        models_response = list()
        models = [model['name'] for model in models_response['models']]
        return jsonify({'available_models': models})
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Failed to list models'}), 500

@app.route('/models/pull', methods=['POST'])
def pull_model():
    try: 
        data = request.json
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Set initial status
        model_download_status[model_name] = {
            'status': 'downloading',
            'progress': 0,
            'start_time': time.time(),
            'error': None
        }
        
        # Start a background thread to pull the model
        def pull_model_task():
            try:
                # Pull the model (without stream and handler parameters)
                pull(model_name)
                
                # Update status on completion
                model_download_status[model_name]['status'] = 'completed'
                model_download_status[model_name]['progress'] = 100
                model_download_status[model_name]['end_time'] = time.time()
            except Exception as e:
                error_msg = str(e)
                print(f"Error pulling model {model_name}: {error_msg}")
                model_download_status[model_name]['status'] = 'failed'
                model_download_status[model_name]['error'] = error_msg
        
        thread = threading.Thread(target=pull_model_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'pulling',
            'message': f'Started pulling model {model_name} in the background',
            'model': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Failed to pull model'}), 500

@app.route('/models/status', methods=['GET'])
def get_model_status():
    """Get the status of all model downloads"""
    return jsonify({
        'downloads': model_download_status
    })

@app.route('/models/status/<model_name>', methods=['GET'])
def get_specific_model_status(model_name):
    """Get the status of a specific model download"""
    if model_name in model_download_status:
        return jsonify(model_download_status[model_name])
    
    # If model not found in status, check if it's already downloaded
    try:
        model_info = show(model_name)
        if model_info:
            return jsonify({
                'status': 'installed',
                'progress': 100,
                'model_info': model_info
            })
    except:
        pass
    
    return jsonify({
        'status': 'unknown',
        'message': f'No download information for model {model_name}'
    })








@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api_to_ollama(path):
    """
    Direct API proxy route that forwards requests to the Ollama API server running on localhost:11434
    This allows clients to access Ollama's native API endpoints through the same path structure
    """
    # Build the target URL
    target_url = f'http://localhost:11434/api/{path}'
    
    # Get the request headers
    headers = {key: value for key, value in request.headers if key != 'Host'}
    
    try:
        # Forward the request to the Ollama API with the same method, headers, and body
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            stream=True
        )
        
        # Create a response object with the content from the forwarded request
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(name, value) for name, value in resp.raw.headers.items()
                           if name.lower() not in excluded_headers]
        
        # Stream the response back to the client
        def generate():
            for chunk in resp.iter_content(chunk_size=4096):
                yield chunk
        
        return Response(
            generate(),
            resp.status_code,
            response_headers
        )
        
    except requests.RequestException as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to connect to Ollama API server'
        }), 503
    
@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_to_ollama(path):
    """
    General proxy route that forwards requests to the Ollama API server running on localhost:11434
    This allows clients to access any Ollama endpoint through this service
    """
    # Build the target URL
    target_url = f'http://localhost:11434/{path}'
    
    # Get the request headers
    headers = {key: value for key, value in request.headers if key != 'Host'}
    
    try:
        # Forward the request to the Ollama API with the same method, headers, and body
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            stream=True
        )
        
        # Create a response object with the content from the forwarded request
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(name, value) for name, value in resp.raw.headers.items()
                           if name.lower() not in excluded_headers]
        
        # Stream the response back to the client
        def generate():
            for chunk in resp.iter_content(chunk_size=4096):
                yield chunk
        
        return Response(
            generate(),
            resp.status_code,
            response_headers
        )
        
    except requests.RequestException as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to connect to Ollama API server'
        }), 503








if __name__ == '__main__':
    # Record start time for uptime tracking
    app.start_time = time.time()
    
    # Pull default models if specified
    default_models = os.environ.get('DEFAULT_MODELS', '')
    if default_models:
        models_to_pull = default_models.split(',')
        for model in models_to_pull:
            model = model.strip()
            print(f"Pulling default model: {model}")
            try:
                # Start in a non-blocking thread
                thread = threading.Thread(target=lambda: pull(model))
                thread.daemon = True
                thread.start()
            except Exception as e:
                print(f"Error pulling model {model}: {str(e)}")
    
    app.run(debug=True, host='0.0.0.0', port=4000)