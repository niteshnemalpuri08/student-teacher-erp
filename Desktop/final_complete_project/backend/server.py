@app.route("/api/students", methods=["GET"])
=======
# Health check endpoint for Render
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

# --- API endpoints ---
@app.route("/api/students", methods=["GET"])
