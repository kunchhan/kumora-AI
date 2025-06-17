# app.py

import asyncio
import logging
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

# Import your Kumora engine and initialization function
# We adjust the import path because the modules are now in a sub-directory
from prompt_engineering_module.class_utils import *
from kumora_response_engine import *

# --- App Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global Kumora Engine Instance ---
# This is the single instance of Kumora engine that the app will use.
# It's initialized once when the application starts.
kumora_engine: KumoraResponseEngine

# --- API Endpoints ---

@app.route("/")
def index():
    """Serves the main chat page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat():
    """
    The main chat API endpoint. It receives a user message and returns
    Kumora's response. This is an async endpoint to properly handle
    the async nature of the response engine.
    """
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Invalid JSON body")

        user_message = data.get("message")
        user_id = data.get("userId")
        session_id = data.get("sessionId")
        # Pass conversation history for better context
        conversation_history = data.get("history", [])

        if not all([user_message, user_id, session_id]):
            raise BadRequest("Missing required fields: message, userId, sessionId")

        # Await the response from the globally initialized engine
        response_data = await kumora_engine.generate_response(
            user_message=user_message,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history
        )

        return jsonify(response_data)

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred in /chat: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    health_status = await kumora_engine.health_check()
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


# --- Main Execution ---

if __name__ == "__main__":
    # Initialize the Kumora Engine before starting the app
    # We run the async initialization function using asyncio.run()
    logging.info("Initializing Kumora Engine for Flask app...")
    kumora_engine = asyncio.run(initialize_kumora_engine())
    logging.info("Kumora Engine initialized. Starting Flask server...")

    # For development, use debug=True.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(host="0.0.0.0", port=5000, debug=True)